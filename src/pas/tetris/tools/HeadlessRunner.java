package src.pas.tetris.tools;

import edu.bu.pas.tetris.agents.Agent;
import edu.bu.pas.tetris.game.Game;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.losses.MeanSquaredError;
import edu.bu.pas.tetris.nn.optimizers.AdamOptimizer;
import edu.bu.pas.tetris.training.data.Dataset;
import net.sourceforge.argparse4j.inf.Namespace;
import src.pas.tetris.agents.TetrisQAgent;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/** Deterministic, headless fitted-Q learning and separate score evaluation. */
public final class HeadlessRunner {
    private HeadlessRunner() {}

    public static void main(String[] args) throws Exception {
        Locale.setDefault(Locale.US);
        if (args.length == 0 || "--help".equals(args[0]) || "help".equals(args[0])) {
            usage();
            return;
        }
        String mode = args[0];
        if (!mode.equals("train") && !mode.equals("evaluate") && !mode.equals("smoke")) {
            throw new IllegalArgumentException("Expected train, evaluate, or smoke.");
        }
        Options options = new Options(args, mode);
        if (mode.equals("train")) train(options);
        else evaluate(options, mode.equals("smoke"));
    }

    private static void usage() {
        System.out.println("HeadlessRunner train|evaluate|smoke [--key value]\n"
            + "Common: --seed 42 --max-pieces 1000 --model models/tetris.model --csv runs/evaluation.csv\n"
            + "Train: --cycles 10 --games-per-cycle 10 --eval-games 10 --eval-seed 1000042\n"
            + "       --replay-capacity 10000 --sample-size 4000 --epochs 3 --batch-size 64\n"
            + "       --learning-rate 0.001 --gamma 0.95 --gradient-clip 5\n"
            + "       --epsilon-start 0.8 --epsilon-end 0.05 --epsilon-decay 200\n"
            + "Evaluate: --games 20 --eval-seed 2000042 --policy q|random|heuristic|all\n"
            + "Evaluation reports the engine's score, not shaped training reward.\n"
            + "The heuristic is an explicitly separate, hand-written baseline.\n"
            + "Smoke defaults to 2 games and 100 pieces; absent model means an untrained Q policy.");
    }

    public static TetrisQAgent newAgent(long seed) {
        return newAgent(seed, 0.8, 0.05, 200, 5);
    }

    private static TetrisQAgent newAgent(long seed, double start, double end, double decay, double clip) {
        TetrisQAgent agent = new TetrisQAgent("TetrisQAgent",
            new TetrisQAgent.Config(seed, start, end, decay, clip));
        Map<String, Object> values = new HashMap<String, Object>();
        values.put("seed", seed);
        values.put("inFile", "");
        agent.initialize(new Namespace(values));
        return agent;
    }

    /** Terminal transitions have no bootstrap term; time-limit truncations do. */
    public static double bellmanTarget(double reward, double gamma, double bestNextQ, boolean terminal) {
        if (!Double.isFinite(reward) || !Double.isFinite(gamma) || gamma < 0 || gamma >= 1
                || (!terminal && !Double.isFinite(bestNextQ))) {
            throw new IllegalArgumentException("Invalid Bellman target inputs");
        }
        double target = terminal ? reward : reward + gamma * bestNextQ;
        if (!Double.isFinite(target)) throw new IllegalStateException("Non-finite Bellman target");
        return target;
    }

    public static final class Episode {
        public final long seed;
        public final int score, pieces;
        public final boolean terminated, truncated;
        public final double shapedReward;

        Episode(long seed, int score, int pieces, boolean terminated, double reward) {
            this.seed = seed;
            this.score = score;
            this.pieces = pieces;
            this.terminated = terminated;
            this.truncated = !terminated;
            this.shapedReward = reward;
        }
    }

    /** Plays actual engine games; random actions use a stream independent of piece generation. */
    public static Episode play(TetrisQAgent agent, String policy, long seed, int maxPieces) throws Exception {
        return play(agent, policy, seed, maxPieces, 0, null);
    }

    private static Episode play(TetrisQAgent agent, String policy, long seed, int maxPieces,
                                double epsilon, Replay replay) throws Exception {
        if (maxPieces <= 0 || maxPieces > 1000000) throw new IllegalArgumentException("Invalid piece limit");
        PlayingAgent wrapper = new PlayingAgent(agent, policy, seed ^ 0x5DEECE66DL, epsilon);
        // Extra preview pieces prevent the engine's queue exhaustion from ending the capped episode early.
        Game game = new Game(seed, maxPieces + 4, wrapper);
        int pieces = 0;
        double shapedReward = 0;
        while (!game.isOver() && pieces < maxPieces) {
            int oldScore = game.getTotalScore();
            game.update();
            pieces++;
            boolean terminal = game.isOver();
            Game.GameView next = game.getView();
            // scoreThisTurn can be stale when the engine exits before scoring on top-out.
            double reward = TetrisQAgent.rewardFor(TetrisQAgent.occupancy(game.getBoard()),
                game.getTotalScore() - oldScore, terminal);
            shapedReward += reward;
            if (replay != null) {
                double[][] nextInputs = terminal ? new double[0][] : inputs(agent, next);
                replay.add(new Transition(wrapper.lastInput, reward, nextInputs, terminal));
            }
        }
        return new Episode(seed, game.getTotalScore(), pieces, game.isOver(), shapedReward);
    }

    private static List<Mino> moves(Game.GameView view) {
        return TetrisQAgent.orderedMoves(view);
    }

    private static final class PlayingAgent extends Agent {
        final TetrisQAgent delegate;
        final String policy;
        final Random random;
        final double epsilon;
        double[] lastInput;

        PlayingAgent(TetrisQAgent delegate, String policy, long seed, double epsilon) {
            super(policy);
            if (!policy.equals("q") && !policy.equals("random") && !policy.equals("heuristic")) {
                throw new IllegalArgumentException("Unknown policy: " + policy);
            }
            this.delegate = delegate;
            this.policy = policy;
            this.random = new Random(seed);
            this.epsilon = epsilon;
        }

        public void initialize(Namespace args) {}
        public void onGameEnd(Game.GameView view) {}

        public Mino makeMove(Game.GameView view) {
            List<Mino> candidates = moves(view);
            if (candidates.isEmpty()) throw new IllegalStateException("No legal moves");
            Mino action;
            if (policy.equals("random") || (epsilon > 0 && random.nextDouble() < epsilon)) {
                action = candidates.get(random.nextInt(candidates.size()));
            } else if (policy.equals("heuristic")) {
                action = heuristic(view, candidates);
            } else {
                action = delegate.getBestActionAndQValue(view).getFirst();
            }
            lastInput = vector(delegate.getQFunctionInput(view, action));
            return action;
        }
    }

    /** A hand-written baseline only; these weights are never substituted for learned Q values. */
    private static Mino heuristic(Game.GameView view, List<Mino> candidates) {
        Mino selected = null;
        double best = Double.NEGATIVE_INFINITY;
        boolean[][] board = TetrisQAgent.occupancy(view.getBoard());
        for (Mino action : candidates) {
            TetrisQAgent.Afterstate after = TetrisQAgent.simulate(board, TetrisQAgent.actionCells(action));
            TetrisQAgent.BoardFeatures f = TetrisQAgent.analyze(after.occupancy());
            double value = 1.5 * after.linesCleared - 0.45 * f.aggregateHeight
                - 1.0 * f.holes - 0.25 * f.bumpiness - 0.2 * f.maxHeight
                - (after.topOut ? 1000 : 0);
            if (selected == null || value > best) {
                selected = action;
                best = value;
            }
        }
        return selected;
    }

    private static double[] vector(Matrix row) {
        double[] values = new double[TetrisQAgent.FEATURE_COUNT];
        for (int col = 0; col < values.length; col++) values[col] = row.get(0, col);
        return values;
    }

    private static double[][] inputs(TetrisQAgent agent, Game.GameView view) {
        List<Mino> candidates = moves(view);
        double[][] result = new double[candidates.size()][];
        for (int index = 0; index < result.length; index++) {
            result[index] = vector(agent.getQFunctionInput(view, candidates.get(index)));
        }
        return result;
    }

    private static final class Transition {
        final double[] input;
        final double reward;
        final double[][] nextInputs;
        final boolean terminal;
        Transition(double[] input, double reward, double[][] nextInputs, boolean terminal) {
            this.input = input.clone();
            this.reward = reward;
            this.nextInputs = nextInputs;
            this.terminal = terminal;
        }
    }

    private static final class Replay {
        final List<Transition> values = new ArrayList<Transition>();
        final int capacity;
        int cursor;
        Replay(int capacity) { this.capacity = capacity; }
        void add(Transition transition) {
            if (values.size() < capacity) values.add(transition);
            else values.set(cursor, transition);
            cursor = (cursor + 1) % capacity;
        }
        List<Transition> sample(int count, Random random) {
            List<Transition> copy = new ArrayList<Transition>(values);
            Collections.shuffle(copy, random);
            return copy.subList(0, Math.min(count, copy.size()));
        }
    }

    private static Matrix rows(double[][] rows) {
        Matrix matrix = Matrix.zeros(rows.length, TetrisQAgent.FEATURE_COUNT);
        for (int row = 0; row < rows.length; row++) {
            for (int col = 0; col < TetrisQAgent.FEATURE_COUNT; col++) matrix.set(row, col, rows[row][col]);
        }
        return matrix;
    }

    private static double bestNextQ(TetrisQAgent agent, double[][] candidates) throws Exception {
        if (candidates.length == 0) return 0;
        Matrix predictions = agent.getQFunction().forward(rows(candidates));
        double best = Double.NEGATIVE_INFINITY;
        for (int row = 0; row < candidates.length; row++) best = Math.max(best, predictions.get(row, 0));
        return best;
    }

    private static double mse(TetrisQAgent agent, Matrix x, Matrix y) throws Exception {
        return new MeanSquaredError().forward(agent.getQFunction().forward(x), y).item();
    }

    private static void train(Options options) throws Exception {
        int cycles = options.positive("cycles");
        int gamesPerCycle = options.positive("games-per-cycle");
        int evalGames = options.positive("eval-games");
        int maxPieces = options.positive("max-pieces");
        long seed = options.number("seed");
        long evalSeed = options.number("eval-seed");
        long totalGames = Math.multiplyExact((long) cycles, gamesPerCycle);
        if (overlap(seed, totalGames, evalSeed, evalGames)) {
            throw new IllegalArgumentException("Training and validation seed ranges must be disjoint");
        }
        TetrisQAgent agent = newAgent(seed, options.decimal("epsilon-start"), options.decimal("epsilon-end"),
            options.decimal("epsilon-decay"), options.decimal("gradient-clip"));
        double rate = options.decimal("learning-rate");
        double gamma = options.decimal("gamma");
        if (rate <= 0 || gamma < 0 || gamma >= 1) throw new IllegalArgumentException("Invalid learning rate or gamma");
        Optimizer optimizer = new AdamOptimizer(agent.getQFunction().getParameters(), rate, 0.9, 0.999);
        Replay replay = new Replay(options.positive("replay-capacity"));
        Random sampling = new Random(seed ^ 0x9E3779B97F4A7C15L);
        Path csv = output(options.value("csv"));
        Path model = output(options.value("model"));
        String modelName = model.toString();
        Path bestModel = output(modelName.endsWith(".model")
            ? modelName.substring(0, modelName.length() - 6) + ".best.model" : modelName + ".best.model");
        double bestScore = Double.NEGATIVE_INFINITY;
        int bestCycle = 0;
        writeMetadata(csv, options, "validation seeds reused each cycle; final test must use separate evaluation seeds");
        try (BufferedWriter writer = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            writer.write("cycle,training_games,replay_size,epsilon,loss_before,loss_after,validation_mean_score,validation_mean_pieces,validation_truncated\n");
            for (int cycle = 0; cycle < cycles; cycle++) {
                long firstGame = (long) cycle * gamesPerCycle;
                for (int game = 0; game < gamesPerCycle; game++) {
                    long index = firstGame + game;
                    play(agent, "q", Math.addExact(seed, index), maxPieces, agent.explorationRate(index), replay);
                }
                List<Transition> samples = replay.sample(options.positive("sample-size"), sampling);
                double[][] inputRows = new double[samples.size()][];
                Matrix targets = Matrix.zeros(samples.size(), 1);
                // Every target is computed before any optimizer update: a frozen target for this cycle.
                for (int row = 0; row < samples.size(); row++) {
                    Transition sample = samples.get(row);
                    inputRows[row] = sample.input;
                    targets.set(row, 0, bellmanTarget(sample.reward, gamma,
                        sample.terminal ? 0 : bestNextQ(agent, sample.nextInputs), sample.terminal));
                }
                Matrix x = rows(inputRows);
                double before = mse(agent, x, targets);
                agent.trainQFunction(new Dataset(x, targets, options.positive("batch-size"), sampling),
                    new MeanSquaredError(), optimizer, options.positive("epochs"));
                double after = mse(agent, x, targets);
                double score = 0, pieces = 0;
                int truncated = 0;
                for (int game = 0; game < evalGames; game++) {
                    Episode episode = play(agent, "q", Math.addExact(evalSeed, game), maxPieces);
                    score += episode.score;
                    pieces += episode.pieces;
                    if (episode.truncated) truncated++;
                }
                agent.getQFunction().save(model.toString());
                if (score / evalGames > bestScore) {
                    bestScore = score / evalGames;
                    bestCycle = cycle + 1;
                    agent.getQFunction().save(bestModel.toString());
                    String selection = String.format(Locale.US,
                        "{\n  \"selection\": \"highest validation mean engine score; earliest cycle wins ties\",\n"
                        + "  \"cycle\": %d,\n  \"mean_score\": %.8f,\n  \"validation_seed\": %d,\n"
                        + "  \"validation_games\": %d,\n  \"max_pieces\": %d\n}\n",
                        bestCycle, bestScore, evalSeed, evalGames, maxPieces);
                    Files.write(Paths.get(bestModel.toString() + ".json"), selection.getBytes(StandardCharsets.UTF_8));
                }
                String line = String.format(Locale.US, "%d,%d,%d,%.8f,%.8f,%.8f,%.6f,%.6f,%d%n",
                    cycle + 1, firstGame + gamesPerCycle, replay.values.size(),
                    agent.explorationRate(firstGame + gamesPerCycle), before, after,
                    score / evalGames, pieces / evalGames, truncated);
                writer.write(line);
                writer.flush();
                System.out.printf(Locale.US, "cycle=%d loss=%.5f -> %.5f validation_score=%.3f pieces=%.1f replay=%d%n",
                    cycle + 1, before, after, score / evalGames, pieces / evalGames, replay.values.size());
            }
        }
        if (!Files.isRegularFile(model) || Files.size(model) == 0) throw new IOException("Model save failed: " + model);
        System.out.println("Saved final model: " + model + "\nBest validation checkpoint: " + bestModel
            + " (cycle " + bestCycle + ", mean score " + bestScore + ")\nTraining metrics: " + csv);
    }

    private static boolean overlap(long first, long count, long other, long otherCount) {
        long end = Math.addExact(first, count - 1);
        long otherEnd = Math.addExact(other, otherCount - 1);
        return first <= otherEnd && other <= end;
    }

    private static void evaluate(Options options, boolean smoke) throws Exception {
        TetrisQAgent agent = newAgent(options.number("seed"));
        boolean loaded = Files.isRegularFile(Paths.get(options.value("model")));
        if (loaded) agent.getQFunction().load(options.value("model"));
        String policy = options.value("policy");
        if (!loaded && !smoke && (policy.equals("q") || policy.equals("all"))) {
            throw new IOException("Q evaluation requires an existing --model. Use smoke for an untrained check.");
        }
        String[] policies = policy.equals("all") ? new String[] {"q", "random", "heuristic"} : new String[] {policy};
        int games = options.positive("games");
        int maxPieces = options.positive("max-pieces");
        long evalSeed = options.number("eval-seed");
        Path csv = output(options.value("csv"));
        writeMetadata(csv, options, loaded ? "Q checkpoint loaded; scores are engine points" : "untrained Q smoke test");
        try (BufferedWriter writer = Files.newBufferedWriter(csv, StandardCharsets.UTF_8)) {
            writer.write("policy,seed,score,pieces,terminated,truncated,shaped_reward\n");
            for (String selected : policies) {
                String label = selected.equals("q") && !loaded ? "q-untrained" : selected;
                double total = 0, sumSquares = 0, pieces = 0;
                int truncated = 0;
                for (int index = 0; index < games; index++) {
                    Episode episode = play(agent, selected, Math.addExact(evalSeed, index), maxPieces);
                    total += episode.score;
                    sumSquares += (double) episode.score * episode.score;
                    pieces += episode.pieces;
                    if (episode.truncated) truncated++;
                    writer.write(String.format(Locale.US, "%s,%d,%d,%d,%s,%s,%.8f%n", label, episode.seed,
                        episode.score, episode.pieces, episode.terminated, episode.truncated, episode.shapedReward));
                    writer.flush();
                }
                double mean = total / games;
                double sd = games > 1 ? Math.sqrt(Math.max(0, (sumSquares - total * total / games) / (games - 1))) : 0;
                System.out.printf(Locale.US, "%s games=%d mean_score=%.4f sample_sd=%.4f mean_pieces=%.2f truncated=%d%n",
                    label, games, mean, sd, pieces / games, truncated);
            }
        }
        System.out.println("Evaluation rows: " + csv);
    }

    private static Path output(String value) throws IOException {
        Path path = Paths.get(value).toAbsolutePath();
        if (path.getParent() != null) Files.createDirectories(path.getParent());
        return path;
    }

    private static void writeMetadata(Path csv, Options options, String note) throws IOException {
        StringBuilder json = new StringBuilder("{\n  \"engine\": \"tetris-1.0.3\",\n  \"mode\": \"")
            .append(options.mode).append("\",\n  \"note\": \"").append(note).append("\",\n  \"options\": {\n");
        int index = 0;
        for (Map.Entry<String, String> entry : options.values.entrySet()) {
            if (index++ > 0) json.append(",\n");
            json.append("    \"").append(entry.getKey()).append("\": \"")
                .append(entry.getValue().replace("\\", "\\\\").replace("\"", "\\\"")).append("\"");
        }
        json.append("\n  }\n}\n");
        Files.write(Paths.get(csv.toString() + ".json"), json.toString().getBytes(StandardCharsets.UTF_8));
    }

    private static final class Options {
        final Map<String, String> values = new LinkedHashMap<String, String>();
        final String mode;
        Options(String[] args, String mode) {
            this.mode = mode;
            String[][] defaults = {{"seed", "42"}, {"max-pieces", mode.equals("smoke") ? "100" : "1000"},
                {"model", "models/tetris.model"}, {"csv", "runs/" + mode + ".csv"},
                {"cycles", "10"}, {"games-per-cycle", "10"}, {"eval-games", "10"},
                {"eval-seed", mode.equals("train") ? "1000042" : "2000042"},
                {"replay-capacity", "10000"}, {"sample-size", "4000"}, {"epochs", "3"},
                {"batch-size", "64"}, {"learning-rate", "0.001"}, {"gamma", "0.95"},
                {"gradient-clip", "5"}, {"epsilon-start", "0.8"}, {"epsilon-end", "0.05"},
                {"epsilon-decay", "200"}, {"games", mode.equals("smoke") ? "2" : "20"}, {"policy", "all"}};
            for (String[] pair : defaults) values.put(pair[0], pair[1]);
            for (int index = 1; index < args.length; index += 2) {
                if (!args[index].startsWith("--") || index + 1 == args.length) {
                    throw new IllegalArgumentException("Each option requires --key value");
                }
                String key = args[index].substring(2);
                if (!values.containsKey(key)) throw new IllegalArgumentException("Unknown option: " + key);
                values.put(key, args[index + 1]);
            }
        }
        String value(String key) { return values.get(key); }
        long number(String key) { return Long.parseLong(value(key)); }
        int positive(String key) {
            int value = Integer.parseInt(value(key));
            if (value <= 0) throw new IllegalArgumentException(key + " must be positive");
            return value;
        }
        double decimal(String key) {
            double value = Double.parseDouble(value(key));
            if (!Double.isFinite(value)) throw new IllegalArgumentException(key + " must be finite");
            return value;
        }
    }
}
