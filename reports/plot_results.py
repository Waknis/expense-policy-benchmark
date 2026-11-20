"""Plot paired-case successes from the saved held-out report."""

import json
from pathlib import Path
import textwrap

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("llm_policy", "Policy only", "#91a0aa"),
    ("llm_evidence", "With records", "#216b73"),
    ("llm_finding", "With records + finding", "#a74b2a"),
]


def main():
    report = json.loads((ROOT / "results/heldout.json").read_text(encoding="utf-8"))
    families = sorted(report["families"])
    fig, ax = plt.subplots(figsize=(9.0, 5.7), dpi=150)
    max_pairs = max(report["families"][f]["llm_evidence"]["n_pairs"] for f in families)
    for offset, (key, label, color) in zip((-.23, 0, .23), METHODS):
        x = [report["families"][f][key]["pairs_both_correct"] for f in families]
        y = [i + offset for i in range(len(families))]
        ax.scatter(x, y, s=59, color=color, label=label, zorder=4)
    ax.set_yticks(range(len(families)), [textwrap.fill(f.replace("_", " ").capitalize(), 25) for f in families])
    ax.set_ylim(len(families) - .5, -.6)
    ax.set_xlim(-.35, max_pairs + .3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Pairs where both decisions were correct", labelpad=10)
    ax.grid(axis="x", color="#e3e8e8", linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Can the model change its decision when one fact changes?",
                 x=.03, ha="left", y=.97, fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=.26, right=.97, top=.81, bottom=.14)
    fig.savefig(ROOT / "reports/paired-decisions.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
