import edu.bu.pas.tetris.agents.Agent;
import edu.bu.pas.tetris.game.Game;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.losses.MeanSquaredError;
import edu.bu.pas.tetris.nn.optimizers.AdamOptimizer;
import edu.bu.pas.tetris.training.data.Dataset;
import src.pas.tetris.agents.TetrisQAgent;
import src.pas.tetris.tools.HeadlessRunner;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import net.sourceforge.argparse4j.inf.Namespace;

/** Dependency-free behavior tests; run with java -ea, never Java's disabled-by-default asserts alone. */
public final class TetrisQAgentTest {
    private static int passed;
    private interface Test { void run() throws Exception; }

    public static void main(String[] args) throws Exception {
        test("empty board measurements", TetrisQAgentTest::emptyBoard);
        test("holes, heights, bumpiness, and hole depth", TetrisQAgentTest::boardMeasurements);
        test("triangular well depth", TetrisQAgentTest::wells);
        test("line clears and eroded cells", TetrisQAgentTest::lineClears);
        test("hypothetical boards are immutable", TetrisQAgentTest::immutability);
        test("hidden-row top-out precedes line clearing", TetrisQAgentTest::topOut);
        test("invalid inputs fail explicitly", TetrisQAgentTest::validation);
        test("feature dimensions and normalization", TetrisQAgentTest::features);
        test("epsilon decay and reward semantics", TetrisQAgentTest::explorationAndReward);
        test("terminal and truncated Bellman targets", TetrisQAgentTest::bellman);
        test("seeded initialization and checkpoint round trip", TetrisQAgentTest::checkpoint);
        test("optimizer learns a synthetic regression task", TetrisQAgentTest::regression);
        test("real seeded games reproduce scores and lengths", TetrisQAgentTest::games);
        test("afterstate matches the actual engine through line clears", TetrisQAgentTest::engineAgreement);
        test("training CLI bounds replay and reproduces checkpoint", TetrisQAgentTest::trainingCli);
        System.out.println("PASS " + passed + " behavior tests");
    }

    private static void test(String name, Test test) throws Exception {
        test.run();
        passed++;
        System.out.println("PASS " + name);
    }

    private static boolean[][] empty() { return new boolean[22][10]; }
    private static void check(boolean ok, String message) {
        if (!ok) throw new AssertionError(message);
    }
    private static void close(double expected, double actual, double tolerance, String message) {
        check(Double.isFinite(actual) && Math.abs(expected - actual) <= tolerance,
            message + ": expected " + expected + ", actual " + actual);
    }
    private static void fails(Runnable action) {
        boolean rejected = false;
        try { action.run(); } catch (IllegalArgumentException expected) { rejected = true; }
        check(rejected, "Expected IllegalArgumentException");
    }

    private static void emptyBoard() {
        TetrisQAgent.BoardFeatures f = TetrisQAgent.analyze(empty());
        check(f.aggregateHeight == 0 && f.maxHeight == 0 && f.holes == 0 && f.filledCells == 0, "Empty counts");
        check(f.bumpiness == 0 && f.wells == 0 && f.holeDepth == 0, "Empty surface");
        check(f.rowTransitions == 44 && f.columnTransitions == 10, "Wall/floor transition convention");
    }

    private static void boardMeasurements() {
        boolean[][] board = empty();
        board[21][0] = true;
        board[19][1] = true;
        board[21][1] = true;
        TetrisQAgent.BoardFeatures f = TetrisQAgent.analyze(board);
        check(f.aggregateHeight == 4 && f.maxHeight == 3, "Heights include covered holes");
        check(f.holes == 1 && f.holeDepth == 1 && f.rowsWithHoles == 1, "One covered empty cell");
        check(f.bumpiness == 5 && f.filledCells == 3, "Surface and occupancy differ");
    }

    private static void wells() {
        boolean[][] board = empty();
        board[20][3] = board[21][3] = board[20][5] = board[21][5] = true;
        check(TetrisQAgent.analyze(board).wells == 3, "Two-cell well contributes 1+2");
    }

    private static void lineClears() {
        boolean[][] single = empty();
        for (int x = 0; x < 6; x++) single[21][x] = true;
        TetrisQAgent.Afterstate one = TetrisQAgent.simulate(single,
            new int[][] {{6,21},{7,21},{8,21},{9,21}});
        check(one.linesCleared == 1 && one.erodedPieceCells == 4, "Single-line erosion");
        check(TetrisQAgent.analyze(one.occupancy()).filledCells == 0, "Single perfect clear");

        boolean[][] doubleLine = empty();
        for (int y = 20; y < 22; y++) for (int x = 0; x < 8; x++) doubleLine[y][x] = true;
        TetrisQAgent.Afterstate two = TetrisQAgent.simulate(doubleLine,
            new int[][] {{8,20},{9,20},{8,21},{9,21}});
        check(two.linesCleared == 2 && two.erodedPieceCells == 8, "Double-line erosion");
        check(TetrisQAgent.analyze(two.occupancy()).filledCells == 0, "Double perfect clear");

        boolean[][] tetris = empty();
        for (int y = 18; y < 22; y++) for (int x = 0; x < 9; x++) tetris[y][x] = true;
        TetrisQAgent.Afterstate four = TetrisQAgent.simulate(tetris,
            new int[][] {{9,18},{9,19},{9,20},{9,21}});
        check(four.linesCleared == 4 && four.erodedPieceCells == 16, "Tetris erosion");
        check(TetrisQAgent.analyze(four.occupancy()).filledCells == 0, "Tetris perfect clear");
    }

    private static void immutability() {
        boolean[][] board = empty();
        board[21][0] = true;
        boolean[][] original = new boolean[22][];
        for (int row = 0; row < 22; row++) original[row] = board[row].clone();
        int[][] action = {{6,21},{7,21},{8,21},{9,21}};
        TetrisQAgent.Afterstate after = TetrisQAgent.simulate(board, action);
        TetrisQAgent.extractFeatures(board, action);
        check(Arrays.deepEquals(board, original), "Candidate scoring mutated the live board");
        boolean[][] exposed = after.occupancy();
        exposed[21][9] = false;
        check(after.occupancy()[21][9], "Afterstate leaked a mutable cell array");
        String[] names = TetrisQAgent.featureNames();
        names[0] = "changed";
        check(!TetrisQAgent.featureNames()[0].equals("changed"), "Feature names are not defensive");
    }

    private static void topOut() {
        boolean[][] board = empty();
        for (int x = 1; x < 10; x++) board[4][x] = true;
        TetrisQAgent.Afterstate result = TetrisQAgent.simulate(board,
            new int[][] {{0,1},{0,2},{0,3},{0,4}});
        check(result.topOut && result.linesCleared == 0, "Top-out must precede clearing the full row");
        check(result.occupancy()[4][9] && result.occupancy()[1][0], "Terminal board must retain placement");
    }

    private static void validation() {
        fails(() -> TetrisQAgent.analyze(new boolean[][] {new boolean[2], new boolean[3], new boolean[2]}));
        fails(() -> TetrisQAgent.simulate(empty(), new int[][] {{0,21},{0,21},{1,21},{2,21}}));
        fails(() -> TetrisQAgent.simulate(empty(), new int[][] {{0,22},{1,21},{2,21},{3,21}}));
        fails(() -> new TetrisQAgent.Config(1, 0.2, 0.3, 10, 5));
        fails(() -> new TetrisQAgent.Config(1, Double.NaN, 0.1, 10, 5));
        fails(() -> HeadlessRunner.bellmanTarget(0, 1, 2, false));
        fails(() -> HeadlessRunner.bellmanTarget(Double.NaN, .95, 0, true));
        fails(() -> HeadlessRunner.bellmanTarget(0, .95, Double.POSITIVE_INFINITY, false));
    }

    private static void features() {
        TetrisQAgent agent = HeadlessRunner.newAgent(19);
        Game game = new Game(42, 100, agent);
        boolean[][] original = TetrisQAgent.occupancy(game.getBoard());
        for (edu.bu.pas.tetris.game.minos.Mino action : game.getView().getFinalMinoPositions()) {
            Matrix feature = agent.getQFunctionInput(game.getView(), action);
            check(feature.getShape().getNumRows() == 1 && feature.getShape().getNumCols() == 18, "Feature shape");
            for (int col = 0; col < 18; col++) {
                check(Double.isFinite(feature.get(0,col)) && Math.abs(feature.get(0,col)) <= 1, "Feature bound");
            }
        }
        check(Arrays.deepEquals(original, TetrisQAgent.occupancy(game.getBoard())), "Feature extraction changed engine board");
        boolean[][] board = empty();
        for (int x = 0; x < 6; x++) board[21][x] = true;
        double[] f = TetrisQAgent.extractFeatures(board, new int[][] {{6,21},{7,21},{8,21},{9,21}});
        close(.25, f[9], 0, "Normalized clear count");
        close(1, f[15], 0, "Perfect-clear feature");
    }

    private static void explorationAndReward() {
        TetrisQAgent agent = HeadlessRunner.newAgent(9);
        close(.8, agent.explorationRate(0), 1e-12, "Initial epsilon");
        check(agent.explorationRate(200) < agent.explorationRate(10), "Exploration should decay");
        close(.05, agent.explorationRate(100000), 1e-12, "Exploration floor");
        close(2, TetrisQAgent.rewardFor(empty(),1,false) - TetrisQAgent.rewardFor(empty(),0,false), 1e-12, "Score reward");
        close(5, TetrisQAgent.rewardFor(empty(),0,false) - TetrisQAgent.rewardFor(empty(),0,true), 1e-12, "Terminal penalty");
    }

    private static void bellman() {
        close(-5, HeadlessRunner.bellmanTarget(-5,.95,Double.NaN,true), 0, "No bootstrap at terminal");
        close(4.8, HeadlessRunner.bellmanTarget(1,.95,4,false), 1e-12, "Bootstrap at nonterminal/truncation");
    }

    private static Matrix probe() {
        Matrix x = Matrix.zeros(8,18);
        Random random = new Random(88);
        for (int row = 0; row < 8; row++) for (int col = 0; col < 18; col++) x.set(row,col,random.nextDouble());
        return x;
    }

    private static void checkpoint() throws Exception {
        TetrisQAgent first = HeadlessRunner.newAgent(91);
        TetrisQAgent second = HeadlessRunner.newAgent(91);
        TetrisQAgent third = HeadlessRunner.newAgent(92);
        Matrix x = probe(), a = first.getQFunction().forward(x), b = second.getQFunction().forward(x);
        Matrix different = third.getQFunction().forward(x);
        boolean differs = false;
        for (int row = 0; row < 8; row++) {
            close(a.get(row,0), b.get(row,0), 0, "Seeded initialization");
            differs |= Math.abs(a.get(row,0) - different.get(row,0)) > 1e-8;
        }
        check(differs, "Distinct seeds should initialize different weights");
        Path temp = Files.createTempFile("tetris-checkpoint-", ".model");
        try {
            first.getQFunction().save(temp.toString());
            third.getQFunction().load(temp.toString());
            Matrix restored = third.getQFunction().forward(x);
            for (int row = 0; row < 8; row++) close(a.get(row,0), restored.get(row,0), 1e-12, "Checkpoint prediction");
        } finally { Files.deleteIfExists(temp); }
    }

    private static void regression() throws Exception {
        TetrisQAgent agent = HeadlessRunner.newAgent(11);
        Matrix x = Matrix.zeros(64,18), y = Matrix.zeros(64,1);
        Random random = new Random(12);
        for (int row = 0; row < 64; row++) {
            for (int col = 0; col < 18; col++) x.set(row,col,random.nextDouble() * 2 - 1);
            y.set(row,0,0.4 * x.get(row,0) - 0.2 * x.get(row,1) + 0.1);
        }
        MeanSquaredError loss = new MeanSquaredError();
        double before = loss.forward(agent.getQFunction().forward(x), y).item();
        agent.trainQFunction(new Dataset(x,y,16,new Random(13)),loss,
            new AdamOptimizer(agent.getQFunction().getParameters(),.005,.9,.999),80);
        double after = loss.forward(agent.getQFunction().forward(x), y).item();
        check(after < before * .1 && after < .005, "Loss failed to decrease: " + before + " -> " + after);
        System.out.printf("  synthetic half-MSE %.8f -> %.8f%n", before, after);
    }

    private static void games() throws Exception {
        TetrisQAgent agent = HeadlessRunner.newAgent(7);
        HeadlessRunner.Episode first = HeadlessRunner.play(agent,"random",12345,300);
        HeadlessRunner.Episode second = HeadlessRunner.play(agent,"random",12345,300);
        check(first.terminated && first.pieces > 0 && first.pieces < 300, "Real random game must end naturally");
        check(first.score == second.score && first.pieces == second.pieces, "Same piece/action seeds must reproduce");
        close(first.shapedReward,second.shapedReward,0,"Reproducible reward");
        HeadlessRunner.Episode capped = HeadlessRunner.play(agent,"heuristic",12345,10);
        check(capped.truncated && !capped.terminated && capped.pieces == 10, "Piece cap must be labeled truncation");
        HeadlessRunner.Episode q = HeadlessRunner.play(agent,"q",98765,100);
        check(q.pieces > 0 && q.score >= 0 && Double.isFinite(q.shapedReward), "Real Q policy executes legal actions");
    }

    private static final class FixedAgent extends Agent {
        Mino chosen;
        FixedAgent() { super("integration-test"); }
        public void initialize(Namespace ignored) {}
        public Mino makeMove(Game.GameView ignored) { return chosen; }
        public void onGameEnd(Game.GameView ignored) {}
    }

    private static void engineAgreement() {
        int clears = 0, moves = 0, topouts = 0;
        for (int seed = 0; seed < 8; seed++) {
            FixedAgent agent = new FixedAgent();
            Game game = new Game(2000 + seed, 304, agent);
            Random random = new Random(3000 + seed);
            for (int step = 0; step < 300 && !game.isOver(); step++) {
                boolean[][] before = TetrisQAgent.occupancy(game.getBoard());
                List<Mino> legal = TetrisQAgent.orderedMoves(game.getView());
                Mino chosen = legal.get(random.nextInt(legal.size()));
                if (seed >= 3) {
                    double best = Double.NEGATIVE_INFINITY;
                    for (Mino candidate : legal) {
                        TetrisQAgent.Afterstate after = TetrisQAgent.simulate(before,TetrisQAgent.actionCells(candidate));
                        TetrisQAgent.BoardFeatures f = TetrisQAgent.analyze(after.occupancy());
                        double value = 10 * after.linesCleared - .5 * f.aggregateHeight - .8 * f.holes
                            - .2 * f.bumpiness - (after.topOut ? 100 : 0);
                        if (value > best) { chosen = candidate; best = value; }
                    }
                }
                TetrisQAgent.Afterstate predicted = TetrisQAgent.simulate(before,TetrisQAgent.actionCells(chosen));
                check(Arrays.deepEquals(before,TetrisQAgent.occupancy(game.getBoard())),"Candidate board mutation");
                agent.chosen = chosen;
                game.update();
                check(Arrays.deepEquals(predicted.occupancy(),TetrisQAgent.occupancy(game.getBoard())),
                    "Engine mismatch seed=" + seed + " step=" + step);
                moves++;
                if (predicted.linesCleared > 0) clears++;
                if (predicted.topOut) topouts++;
            }
        }
        check(clears >= 20 && topouts >= 3, "Integration must exercise clears and top-outs");
        System.out.println("  engine matched " + moves + " moves, " + clears + " clears, " + topouts + " top-outs");
    }

    private static void trainingCli() throws Exception {
        Path directory = Files.createTempDirectory("tetris-training-test-");
        try {
            for (int run = 0; run < 2; run++) {
                HeadlessRunner.main(new String[] {"train", "--seed", "123", "--cycles", "2",
                    "--games-per-cycle", "2", "--eval-games", "1", "--eval-seed", "100123",
                    "--max-pieces", "40", "--replay-capacity", "16", "--sample-size", "16",
                    "--epochs", "2", "--batch-size", "8", "--model", directory.resolve("run" + run + ".model").toString(),
                    "--csv", directory.resolve("run" + run + ".csv").toString()});
            }
            check(Arrays.equals(Files.readAllBytes(directory.resolve("run0.model")),
                Files.readAllBytes(directory.resolve("run1.model"))), "Training weights must reproduce exactly");
            List<String> lines = Files.readAllLines(directory.resolve("run0.csv"));
            check(lines.size() == 3, "Two cycles must produce two metric rows");
            for (int row = 1; row < lines.size(); row++) {
                check(Integer.parseInt(lines.get(row).split(",")[2]) == 16, "Replay must respect its capacity");
            }
            check(Files.size(directory.resolve("run0.best.model")) > 0, "Best validation checkpoint missing");
        } finally {
            try (java.util.stream.Stream<Path> paths = Files.walk(directory)) {
                for (Path path : (Iterable<Path>) paths.sorted(Comparator.reverseOrder())::iterator) Files.delete(path);
            }
        }
    }
}
