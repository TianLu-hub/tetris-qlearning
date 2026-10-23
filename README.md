# Tetris Q-Learning

A Java reinforcement-learning agent that ranks legal Tetris placements with a neural Q-function. The project connects board feature engineering, replay-based Bellman updates, deterministic experiments and actual game-score evaluation.

**500 held-out games: 48.54 average score**, after 600 training games; untrained Q: 4.57, random: 0.13, hand-coded baseline: 59.62. All use the same held-out seed range. [Full results and limitations](results/REPORT.md).

**Reconstruction of a 2025 BU CS 440 project, implemented in October 2026.** The original source and weights were lost. This repository contains newly written code, not recovered historical work. See [provenance and attribution](NOTICE.md).

## How it works

```text
Legal placements → simulate each afterstate → 18 board/action features
                                                 ↓
                                    Dense 32 → tanh → Dense 16 → tanh → Dense 1
                                                 ↓
                                          greedy Q-action
                                                 ↓
                 game transition → replay buffer → frozen Bellman targets
                                                 ↓
                                      shuffled minibatch updates
```

- **State/action representation:** heights, holes, bumpiness, wells, transitions, covered-hole depth, clears, landing height, eroded cells, filled cells, top-out, perfect clear and changes in holes/height. Features are normalized; candidate simulation leaves the engine state unchanged.
- **Neural value function:** `18 → 32 → 16 → 1`, with tanh hidden activations, seeded initialization and an unbounded scalar output. No handwritten heuristic is added to the learned Q-values.
- **Exploration:** epsilon-greedy with exponential decay by training-game count; stable candidate ordering makes seeded runs reproducible.
- **Reward:** line-clear score plus survival reward, with penalties for aggregate height, holes, surface bumpiness and losing. This shaped reward is distinct from the game's score.
- **Training:** bounded replay, targets frozen during each fitting phase, minibatch shuffling, finite-value validation and gradient clipping.
- **Evaluation:** a frozen greedy policy on held-out piece sequences, compared with random and explicitly labeled hand-coded baselines. CSV output records individual game scores.

## Local setup

Requires JDK 8 or later, Bash and curl. The course engine is not publicly redistributed. Obtain your authorized `tetris-1.0.3.jar` and run:

```bash
bash scripts/setup.sh /path/to/tetris-1.0.3.jar
bash scripts/build.sh
bash scripts/test.sh
bash scripts/run.sh smoke
```

`setup.sh` copies the engine into the ignored `lib/` directory and downloads checksum-verified argparse4j 0.9.0 from Maven Central. Set `JAVA_HOME` to select a JDK. On Windows, use Git Bash and change the classpath separator in `scripts/common.sh` to `;`, or compile directly with `javac`.

## Train and evaluate

```bash
bash scripts/run.sh train --seed 42 --cycles 30 --games-per-cycle 20 \
  --eval-games 20 --eval-seed 1000042 --max-pieces 1000 \
  --model params/q.model --csv results/training.csv

bash scripts/run.sh evaluate --model models/params.model --policy all \
  --seed 42 --eval-seed 2000042 --games 500 --max-pieces 1000 --csv results/evaluation.csv
```

Evaluation scores depend on the engine's scoring rules, game cap, checkpoint and seed range. A finite piece cap is an experiment limit; reaching it is not the same as losing. Use fresh seeds for final evaluation after selecting a checkpoint on validation games. See the checked-in [run report](results/REPORT.md) for measured results and exact commands.

The custom headless runner keeps transitions within their own games and reports `Game.getTotalScore()`. It is preferred over the bundled trainer, whose logged trajectory utility is shaped reward and whose previous-transition bookkeeping crosses game boundaries in version 1.0.3.

### Course interface compatibility

The agent implements all six abstract methods required by the supplied JAR, including `trainQFunction`. The supplied version's namespace is `edu.bu.pas.tetris`, even though the assignment PDF shows `edu.bu.tetris` in examples. To use its original trainer:

```bash
java -cp 'lib/*:build/classes' edu.bu.pas.tetris.Main \
  -q src.pas.tetris.agents.TetrisQAgent -s --seed 42
```

Use `--help` for the engine's options. The original `tetris.srcs` path is preserved.

An evaluated checkpoint is included at `models/params.model`. To evaluate your own new training run, use its output model path. The optional chart script uses Python and matplotlib; neither is required for Java training.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/pas/tetris/agents/TetrisQAgent.java` | Features, afterstate simulation, neural model, exploration, reward and training |
| `src/pas/tetris/tools/HeadlessRunner.java` | Reproducible train/evaluate CLI and replay management |
| `tests/` | Board fixtures, model checks and real-engine integration tests |
| `scripts/` | Dependency setup, build, test and run commands |
| `results/` | Measured reconstruction results and experiment metadata |

## 中文说明

这是 BU CS 440 俄罗斯方块强化学习项目的重建版本，保留课程引擎的 Java 接口。智能体对每个合法落点生成 18 维特征，通过前馈神经网络估计动作价值，并使用经验回放、Bellman 目标和 epsilon-greedy 探索进行训练。

原始代码与模型已丢失，因此本仓库不声称恢复了 2025 年版本，也不把简历中的历史成绩当作本次结果。测试和实际游戏评估见 `results/REPORT.md`。课程 JAR、作业 PDF、个人简历和成绩单均未上传；运行需要自行提供有权使用的课程引擎。
