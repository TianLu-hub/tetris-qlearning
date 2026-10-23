"""Plot actual experiment CSVs. Optional dependency: matplotlib."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--training", type=Path, default=Path("results/training.csv"))
parser.add_argument("--evaluation", type=Path, default=Path("results/evaluation.csv"))
parser.add_argument("--untrained", type=Path, default=Path("results/untrained.csv"))
parser.add_argument("--output", type=Path, default=Path("results/learning-curve.png"))
args = parser.parse_args()
with args.training.open() as source:
    training = list(csv.DictReader(source))
scores = defaultdict(list)
for path in (args.untrained, args.evaluation):
    with path.open() as source:
        for row in csv.DictReader(source):
            scores[row["policy"]].append(float(row["score"]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
fig.suptitle("Tetris Q-Learning | reconstruction benchmark", fontsize=17, fontweight="bold")
left.plot([int(r["training_games"]) for r in training],
          [float(r["validation_mean_score"]) for r in training],
          color="#222222", linewidth=2)
left.set(title="Validation during training", xlabel="Training games", ylabel="Mean engine score")
left.grid(axis="y", color="#dddddd", linewidth=0.6)
names = ["q-untrained", "random", "q", "heuristic"]
labels = ["Untrained Q", "Random", "Trained Q", "Heuristic\n(hand-coded)"]
means = [sum(scores[n]) / len(scores[n]) for n in names]
bars = right.bar(labels, means, color=["#bdbdbd", "#dddddd", "#222222", "#999999"], width=0.65)
for bar, mean in zip(bars, means):
    right.annotate(f"{mean:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   xytext=(0, 5), textcoords="offset points", ha="center")
right.set(title=f"Held-out evaluation ({len(scores['q'])} games / policy)", ylabel="Mean engine score")
right.set_ylim(0, max(means) * 1.18)
right.grid(axis="y", color="#dddddd", linewidth=0.6)
right.set_axisbelow(True)
for ax in (left, right):
    ax.spines[["top", "right"]].set_visible(False)
fig.text(0.02, -0.06, "Validation seeds are reused across cycles. Final evaluation uses separate seeds. See REPORT.md for run limits and parameters.", fontsize=9)
args.output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output, dpi=180, bbox_inches="tight")
print(args.output)
