import java.util.*;
import edu.bu.pas.tetris.agents.Agent;
import edu.bu.pas.tetris.game.Game;
import edu.bu.pas.tetris.game.Game.GameView;
import edu.bu.pas.tetris.game.minos.Mino;
import net.sourceforge.argparse4j.inf.Namespace;
import src.pas.tetris.agents.TetrisQAgent;

public final class SimulationProbe {
  static class FixedAgent extends Agent {
    Mino chosen;
    FixedAgent() { super("simulation-probe"); }
    public void initialize(Namespace ignored) {}
    public Mino makeMove(GameView ignored) { return chosen; }
    public void onGameEnd(GameView ignored) {}
  }
  public static void main(String[] args) {
    int moves=0, clears=0, topouts=0;
    for(int seed=0; seed<30; seed++) {
      FixedAgent agent = new FixedAgent();
      Game game = new Game(2000+seed, 1004, agent);
      Random random = new Random(3000+seed);
      for(int step=0; step<1000 && !game.isOver(); step++) {
        boolean[][] before = TetrisQAgent.occupancy(game.getBoard());
        List<Mino> legal = TetrisQAgent.orderedMoves(game.getView());
        Mino chosen = legal.get(random.nextInt(legal.size()));
        if(seed >= 10) {
          double best=Double.NEGATIVE_INFINITY;
          for(Mino candidate:legal) {
            TetrisQAgent.Afterstate a=TetrisQAgent.simulate(before,TetrisQAgent.actionCells(candidate));
            TetrisQAgent.BoardFeatures f=TetrisQAgent.analyze(a.occupancy());
            double value=10*a.linesCleared-.5*f.aggregateHeight-.8*f.holes-.2*f.bumpiness-(a.topOut?100:0);
            if(value>best) { chosen=candidate; best=value; }
          }
        }
        TetrisQAgent.Afterstate predicted=TetrisQAgent.simulate(before,TetrisQAgent.actionCells(chosen));
        if(!Arrays.deepEquals(before,TetrisQAgent.occupancy(game.getBoard()))) throw new AssertionError("Input mutation");
        agent.chosen=chosen;
        game.update();
        if(!Arrays.deepEquals(predicted.occupancy(),TetrisQAgent.occupancy(game.getBoard()))) {
          throw new AssertionError("Simulation mismatch seed="+seed+" step="+step+" clears="+predicted.linesCleared);
        }
        moves++;
        if(predicted.linesCleared>0) clears++;
        if(predicted.topOut) topouts++;
      }
    }
    System.out.println("Simulation matched engine for "+moves+" moves, "+clears+" clearing placements, "+topouts+" topouts across 30 seeded games");
  }
}
