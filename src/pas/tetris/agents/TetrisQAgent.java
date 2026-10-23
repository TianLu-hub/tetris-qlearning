package src.pas.tetris.agents;

import edu.bu.pas.tetris.agents.QAgent;
import edu.bu.pas.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.pas.tetris.game.Block;
import edu.bu.pas.tetris.game.Board;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import edu.bu.pas.tetris.linalg.Matrix;
import edu.bu.pas.tetris.nn.LossFunction;
import edu.bu.pas.tetris.nn.Model;
import edu.bu.pas.tetris.nn.Optimizer;
import edu.bu.pas.tetris.nn.Parameter;
import edu.bu.pas.tetris.nn.layers.Dense;
import edu.bu.pas.tetris.nn.layers.Tanh;
import edu.bu.pas.tetris.nn.models.Sequential;
import edu.bu.pas.tetris.training.data.Dataset;
import edu.bu.pas.tetris.utils.Pair;
import net.sourceforge.argparse4j.inf.Namespace;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

/**
 * Reconstructed neural Q-learning agent for the BU CS 440 Tetris 1.0.3 engine.
 * Features describe the board after a candidate placement. Training targets and
 * replay are supplied by the caller; this class does not add heuristic Q-values.
 */
public final class TetrisQAgent extends QAgent {
    public static final int FEATURE_COUNT = 18;
    private static final String[] FEATURE_NAMES = {
        "mean_height", "max_height", "holes", "bumpiness", "well_depth",
        "row_transitions", "column_transitions", "hole_depth", "rows_with_holes",
        "cleared_lines", "squared_cleared_lines", "landing_height", "eroded_cells",
        "filled_cells", "top_out", "perfect_clear", "hole_delta", "height_delta"
    };

    /** Immutable experiment settings. Epsilon decay is measured in training games. */
    public static final class Config {
        public final long seed;
        public final double epsilonStart, epsilonEnd, epsilonDecayGames, gradientClip;

        public Config(long seed, double epsilonStart, double epsilonEnd,
                      double epsilonDecayGames, double gradientClip) {
            if (!Double.isFinite(epsilonStart) || !Double.isFinite(epsilonEnd)
                    || epsilonEnd < 0 || epsilonStart > 1 || epsilonEnd > epsilonStart
                    || !Double.isFinite(epsilonDecayGames) || epsilonDecayGames <= 0
                    || !Double.isFinite(gradientClip) || gradientClip <= 0) {
                throw new IllegalArgumentException("Invalid exploration or gradient settings");
            }
            this.seed = seed;
            this.epsilonStart = epsilonStart;
            this.epsilonEnd = epsilonEnd;
            this.epsilonDecayGames = epsilonDecayGames;
            this.gradientClip = gradientClip;
        }

        public static Config defaults() {
            return new Config(20250331L, 0.40, 0.05, 3000.0, 10.0);
        }
    }

    private final Config configuration;
    private long effectiveSeed;
    private Random explorationRandom;

    public TetrisQAgent(String name) { this(name, Config.defaults()); }

    public TetrisQAgent(String name, Config configuration) {
        super(name);
        if (configuration == null) throw new IllegalArgumentException("Config is required");
        this.configuration = configuration;
        this.effectiveSeed = configuration.seed;
        this.explorationRandom = new Random(effectiveSeed);
    }

    public Config config() { return configuration; }

    public static String[] featureNames() { return FEATURE_NAMES.clone(); }

    @Override
    public void initialize(Namespace arguments) {
        if (arguments == null) throw new IllegalArgumentException("Arguments are required");
        Object seed = arguments.get("seed");
        if (seed instanceof Number && ((Number) seed).longValue() >= 0) {
            effectiveSeed = ((Number) seed).longValue();
        } else {
            effectiveSeed = configuration.seed;
        }
        explorationRandom = new Random(effectiveSeed);
        super.initialize(arguments);
    }

    /** 18 -> 32 -> 16 -> 1, tanh hidden layers, unbounded linear output. */
    @Override
    public Model initQFunction() {
        Random initializationRandom = new Random(effectiveSeed ^ 0x5DEECE66DL);
        Sequential network = new Sequential();
        network.add(dense(FEATURE_COUNT, 32, initializationRandom));
        network.add(new Tanh());
        network.add(dense(32, 16, initializationRandom));
        network.add(new Tanh());
        network.add(dense(16, 1, initializationRandom));
        return network;
    }

    private static Dense dense(int inputs, int outputs, Random random) {
        Dense layer = new Dense(inputs, outputs, random);
        Matrix weights = layer.getWeight().getValue();
        double scale = Math.sqrt(2.0 / (inputs + outputs));
        for (int row = 0; row < inputs; row++) {
            for (int column = 0; column < outputs; column++) {
                weights.set(row, column, weights.get(row, column) * scale);
            }
        }
        Matrix bias = layer.getBias().getValue();
        for (int column = 0; column < outputs; column++) bias.set(0, column, 0.0);
        return layer;
    }

    @Override
    public Matrix getQFunctionInput(GameView game, Mino action) {
        double[] features = extractFeatures(occupancy(game.getBoard()), actionCells(action));
        Matrix result = Matrix.zeros(1, FEATURE_COUNT);
        for (int i = 0; i < FEATURE_COUNT; i++) result.set(0, i, features[i]);
        return result;
    }

    /** Stable action order also makes ties and random exploration reproducible. */
    public static List<Mino> orderedMoves(GameView game) {
        List<Mino> moves = new ArrayList<Mino>(game.getFinalMinoPositions());
        moves.sort(Comparator.comparingInt((Mino m) -> m.getType().ordinal())
            .thenComparingInt(m -> m.getOrientation().ordinal())
            .thenComparingInt(m -> m.getPivotBlockCoordinate().getXCoordinate())
            .thenComparingInt(m -> m.getPivotBlockCoordinate().getYCoordinate()));
        return moves;
    }

    @Override
    public Pair<Mino, Double> getBestActionAndQValue(GameView game) {
        List<Mino> moves = orderedMoves(game);
        if (moves.isEmpty()) throw new IllegalArgumentException("No legal action in this state");
        if (getQFunction() == null) throw new IllegalStateException("Initialize the agent first");
        Matrix inputs = Matrix.zeros(moves.size(), FEATURE_COUNT);
        boolean[][] board = occupancy(game.getBoard());
        for (int row = 0; row < moves.size(); row++) {
            double[] features = extractFeatures(board, actionCells(moves.get(row)));
            for (int column = 0; column < FEATURE_COUNT; column++) {
                inputs.set(row, column, features[column]);
            }
        }
        try {
            Matrix values = getQFunction().forward(inputs);
            if (values.getShape().getNumRows() != moves.size()
                    || values.getShape().getNumCols() != 1) {
                throw new IllegalStateException("Q model must output one scalar per action");
            }
            requireFinite(values, "Q values");
            int best = 0;
            for (int i = 1; i < moves.size(); i++) {
                if (values.get(i, 0) > values.get(best, 0)) best = i;
            }
            return new Pair<Mino, Double>(moves.get(best), values.get(best, 0));
        } catch (Exception exception) {
            throw new IllegalStateException("Unable to evaluate candidate actions", exception);
        }
    }

    public double explorationRate(long trainingGames) {
        if (trainingGames < 0) throw new IllegalArgumentException("Game count cannot be negative");
        return configuration.epsilonEnd + (configuration.epsilonStart - configuration.epsilonEnd)
            * Math.exp(-trainingGames / configuration.epsilonDecayGames);
    }

    @Override
    public boolean shouldExplore(GameView game, GameCounter counter) {
        if (!counter.isTrainingGame()) return false;
        long trainingGames = counter.getCurrentCycleIdx() * counter.getNumTrainingGames()
            + counter.getCurrentGameIdx();
        return explorationRandom.nextDouble() < explorationRate(trainingGames);
    }

    @Override
    public Mino getExplorationMove(GameView game) {
        List<Mino> moves = orderedMoves(game);
        if (moves.isEmpty()) throw new IllegalArgumentException("Cannot explore without a legal move");
        return moves.get(explorationRandom.nextInt(moves.size()));
    }

    @Override
    public double getReward(GameView game) {
        // The engine leaves scoreThisTurn unchanged when top-out skips scoring.
        boolean lost = game.didAgentLose();
        return rewardFor(occupancy(game.getBoard()), lost ? 0 : game.getScoreThisTurn(), lost);
    }

    /** Shaped training reward, distinct from the engine's reported game score. */
    public static double rewardFor(boolean[][] board, int scoreThisTurn, boolean lost) {
        if (scoreThisTurn < 0) throw new IllegalArgumentException("Score cannot be negative");
        BoardFeatures features = analyze(board);
        return 2.0 * scoreThisTurn + 0.10 - 0.01 * features.aggregateHeight
            - 0.10 * features.holes - 0.015 * features.bumpiness - (lost ? 5.0 : 0.0);
    }

    @Override
    public void trainQFunction(Dataset dataset, LossFunction loss, Optimizer optimizer,
                               long epochs) {
        if (dataset == null || loss == null || optimizer == null || epochs < 0) {
            throw new IllegalArgumentException("Training inputs are missing or invalid");
        }
        if (getQFunction() == null) throw new IllegalStateException("Initialize the agent first");
        if (dataset.getBatchSize() <= 0) throw new IllegalArgumentException("Batch size must be positive");
        try {
            for (long epoch = 0; epoch < epochs; epoch++) {
                dataset.shuffle();
                Iterator<Pair<Matrix, Matrix>> batches = dataset.iterator();
                while (batches.hasNext()) {
                    Pair<Matrix, Matrix> batch = batches.next();
                    Matrix inputs = batch.getFirst();
                    Matrix targets = batch.getSecond();
                    if (inputs.getShape().getNumRows() == 0) continue;
                    if (inputs.getShape().getNumCols() != FEATURE_COUNT
                            || targets.getShape().getNumRows() != inputs.getShape().getNumRows()
                            || targets.getShape().getNumCols() != 1) {
                        throw new IllegalArgumentException("Expected N x 18 inputs and N x 1 targets");
                    }
                    requireFinite(inputs, "training inputs");
                    requireFinite(targets, "Bellman targets");
                    optimizer.reset();
                    Matrix prediction = getQFunction().forward(inputs);
                    requireFinite(prediction, "predictions");
                    Matrix outputGradient = loss.backwards(prediction, targets);
                    requireFinite(outputGradient, "loss gradient");
                    getQFunction().backwards(inputs, outputGradient);
                    for (Parameter parameter : getQFunction().getParameters()) {
                        requireFinite(parameter.getGradient(), "parameter gradient");
                        parameter.setGradient(Matrix.clip(parameter.getGradient(),
                            -configuration.gradientClip, configuration.gradientClip));
                    }
                    optimizer.step();
                    for (Parameter parameter : getQFunction().getParameters()) {
                        requireFinite(parameter.getValue(), "updated parameters");
                    }
                }
            }
        } catch (Exception exception) {
            throw new IllegalStateException("Q-function training failed", exception);
        }
    }

    private static void requireFinite(Matrix matrix, String label) {
        for (int row = 0; row < matrix.getShape().getNumRows(); row++) {
            for (int column = 0; column < matrix.getShape().getNumCols(); column++) {
                if (!Double.isFinite(matrix.get(row, column))) {
                    throw new IllegalStateException("Non-finite " + label + " at " + row + "," + column);
                }
            }
        }
    }

    public static boolean[][] occupancy(Board board) {
        boolean[][] result = new boolean[Board.NUM_ROWS][Board.NUM_COLS];
        for (int row = 0; row < Board.NUM_ROWS; row++) {
            for (int column = 0; column < Board.NUM_COLS; column++) {
                result[row][column] = board.isCoordinateOccupied(column, row);
            }
        }
        return result;
    }

    /** Convert framework coordinates to explicit (x, y) pairs. */
    public static int[][] actionCells(Mino action) {
        Block[] blocks = action.getBlocks();
        int[][] cells = new int[blocks.length][2];
        for (int i = 0; i < blocks.length; i++) {
            cells[i][0] = blocks[i].getCoordinate().getXCoordinate();
            cells[i][1] = blocks[i].getCoordinate().getYCoordinate();
        }
        return cells;
    }

    public static final class BoardFeatures {
        public final int aggregateHeight, maxHeight, holes, bumpiness, wells;
        public final int rowTransitions, columnTransitions, holeDepth, rowsWithHoles, filledCells;
        private BoardFeatures(int aggregateHeight, int maxHeight, int holes, int bumpiness,
                              int wells, int rowTransitions, int columnTransitions,
                              int holeDepth, int rowsWithHoles, int filledCells) {
            this.aggregateHeight = aggregateHeight; this.maxHeight = maxHeight;
            this.holes = holes; this.bumpiness = bumpiness; this.wells = wells;
            this.rowTransitions = rowTransitions; this.columnTransitions = columnTransitions;
            this.holeDepth = holeDepth; this.rowsWithHoles = rowsWithHoles;
            this.filledCells = filledCells;
        }
    }

    /** Pure board measurements: rows run top to bottom; columns run left to right. */
    public static BoardFeatures analyze(boolean[][] board) {
        validateBoard(board);
        int height = board.length, width = board[0].length;
        int[] heights = new int[width];
        boolean[] holeRows = new boolean[height];
        int aggregate = 0, maximum = 0, holes = 0, holeDepth = 0, filled = 0;
        int rowTransitions = 0, columnTransitions = 0, bumpiness = 0, wells = 0;
        for (int column = 0; column < width; column++) {
            int blocksAbove = 0;
            boolean previous = false; // Open sky; the floor below the board is solid.
            for (int row = 0; row < height; row++) {
                boolean occupied = board[row][column];
                if (occupied != previous) columnTransitions++;
                previous = occupied;
                if (occupied) {
                    if (blocksAbove == 0) heights[column] = height - row;
                    blocksAbove++;
                    filled++;
                } else if (blocksAbove > 0) {
                    holes++;
                    holeDepth += blocksAbove;
                    holeRows[row] = true;
                }
            }
            if (!previous) columnTransitions++;
            aggregate += heights[column];
            maximum = Math.max(maximum, heights[column]);
            if (column > 0) bumpiness += Math.abs(heights[column] - heights[column - 1]);
        }
        for (int row = 0; row < height; row++) {
            boolean previous = true; // Side walls are solid.
            for (int column = 0; column < width; column++) {
                if (board[row][column] != previous) rowTransitions++;
                previous = board[row][column];
            }
            if (!previous) rowTransitions++;
        }
        for (int column = 0; column < width; column++) {
            int depth = 0;
            for (int row = 0; row < height; row++) {
                boolean left = column == 0 || board[row][column - 1];
                boolean right = column == width - 1 || board[row][column + 1];
                if (!board[row][column] && left && right) {
                    depth++;
                    wells += depth;
                } else depth = 0;
            }
        }
        int rowsWithHoles = 0;
        for (boolean hasHole : holeRows) if (hasHole) rowsWithHoles++;
        return new BoardFeatures(aggregate, maximum, holes, bumpiness, wells,
            rowTransitions, columnTransitions, holeDepth, rowsWithHoles, filled);
    }

    public static final class Afterstate {
        private final boolean[][] cells;
        public final int linesCleared, erodedPieceCells;
        public final double landingHeight;
        public final boolean topOut;
        private Afterstate(boolean[][] cells, int linesCleared, int erodedPieceCells,
                           double landingHeight, boolean topOut) {
            this.cells = cells; this.linesCleared = linesCleared;
            this.erodedPieceCells = erodedPieceCells; this.landingHeight = landingHeight;
            this.topOut = topOut;
        }
        public boolean[][] occupancy() { return copy(cells); }
    }

    /**
     * Pure hypothetical placement. The engine checks its two hidden rows before
     * line clearing, so a top-out placement is not rescued by a full bottom row.
     * Primitive copies avoid the engine Board copy constructor's shared Blocks.
     */
    public static Afterstate simulate(boolean[][] board, int[][] actionCells) {
        validateBoard(board);
        if (actionCells == null || actionCells.length != 4) {
            throw new IllegalArgumentException("A tetromino must contain four cells");
        }
        int height = board.length, width = board[0].length;
        boolean[][] result = copy(board);
        double landingHeight = 0;
        for (int[] cell : actionCells) {
            if (cell == null || cell.length != 2) throw new IllegalArgumentException("Expected (x,y) cell");
            int x = cell[0], y = cell[1];
            if (x < 0 || x >= width || y < 0 || y >= height || result[y][x]) {
                throw new IllegalArgumentException("Placement overlaps or leaves the board");
            }
            result[y][x] = true;
            landingHeight += height - 1 - y;
        }
        landingHeight /= actionCells.length;
        boolean topOut = false;
        for (int column = 0; column < width; column++) {
            topOut |= result[0][column] || result[1][column];
        }
        if (topOut) return new Afterstate(result, 0, 0, landingHeight, true);
        boolean[] full = new boolean[height];
        int lines = 0;
        for (int row = 2; row < height; row++) {
            full[row] = true;
            for (int column = 0; column < width; column++) full[row] &= result[row][column];
            if (full[row]) lines++;
        }
        int eroded = 0;
        for (int[] cell : actionCells) if (full[cell[1]]) eroded++;
        int destination = height - 1;
        for (int row = height - 1; row >= 0; row--) {
            if (!full[row]) result[destination--] = result[row].clone();
        }
        while (destination >= 0) result[destination--] = new boolean[width];
        return new Afterstate(result, lines, eroded * lines, landingHeight, false);
    }

    /** All features are bounded by [-1,1] for legal tetromino placements. */
    public static double[] extractFeatures(boolean[][] board, int[][] actionCells) {
        BoardFeatures before = analyze(board);
        Afterstate after = simulate(board, actionCells);
        BoardFeatures features = analyze(after.cells);
        double height = board.length, width = board[0].length, area = height * width;
        return new double[] {
            features.aggregateHeight / area,
            features.maxHeight / height,
            features.holes / area,
            features.bumpiness / (height * (width - 1)),
            features.wells / (width * height * (height + 1) / 2.0),
            features.rowTransitions / (height * (width + 1)),
            features.columnTransitions / (width * (height + 1)),
            features.holeDepth / (area * height),
            features.rowsWithHoles / height,
            after.linesCleared / 4.0,
            after.linesCleared * after.linesCleared / 16.0,
            after.landingHeight / height,
            after.erodedPieceCells / 16.0,
            features.filledCells / area,
            after.topOut ? 1.0 : 0.0,
            features.filledCells == 0 ? 1.0 : 0.0,
            (features.holes - before.holes) / area,
            (features.aggregateHeight - before.aggregateHeight) / area
        };
    }

    private static boolean[][] copy(boolean[][] board) {
        boolean[][] result = new boolean[board.length][];
        for (int row = 0; row < board.length; row++) result[row] = board[row].clone();
        return result;
    }

    private static void validateBoard(boolean[][] board) {
        if (board == null || board.length < 3 || board[0] == null || board[0].length < 2) {
            throw new IllegalArgumentException("Board must be rectangular with at least 3 rows and 2 columns");
        }
        int width = board[0].length;
        for (boolean[] row : board) {
            if (row == null || row.length != width) throw new IllegalArgumentException("Board is ragged");
        }
    }
}
