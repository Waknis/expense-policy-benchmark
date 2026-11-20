"""Plot training returns from the five published REINFORCE runs."""

import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]


def main():
    logs = sorted((ROOT / "results/interactive/training/logs").glob("seed-*.jsonl"))
    fig, ax = plt.subplots(figsize=(8.4, 4.65), dpi=160)
    colors = ("#0c5960", "#8c402c", "#437cb2", "#8e6828", "#80577d")
    for path, color in zip(logs, colors, strict=True):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        x, y = [], []
        for i in range(7, len(rows)):
            x.append(rows[i]["episodes"])
            y.append(sum(r["batch_mean_return"] for r in rows[i-7:i+1]) / 8)
        ax.plot(x, y, color=color, linewidth=1.35, label=f"Seed {rows[0]['seed']}")
    ax.axhline(0, color="#a2acae", linewidth=.7)
    ax.grid(axis="y", color="#e4e8e9", linewidth=.7)
    ax.set(xlim=(0, 25000), ylim=(-1.5, 2.08), xlabel="Training episodes per seed",
           ylabel="Mean episode return")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{val/1000:g}k" if val else "0"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Learning when to inspect, stop and decide", x=.11, y=.96, ha="left",
                 fontsize=16, fontweight="bold")
    fig.text(.11, .873, "Eight-update rolling mean during stochastic training · 64 episodes per update",
             fontsize=9, color="#43565a")
    fig.subplots_adjust(left=.11, right=.97, top=.82, bottom=.16)
    fig.savefig(ROOT / "reports/interactive-training.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
