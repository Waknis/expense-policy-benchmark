"""Render the held-out decision counts from the public score report."""

import json
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    ("llm_policy", "Policy only"),
    ("llm_evidence", "Policy + records"),
    ("llm_finding", "Policy + records\n+ a written finding"),
]
ACTIONS = ["approve", "review", "reject"]


def main():
    report = json.loads((ROOT / "results/heldout.json").read_text(encoding="utf-8"))
    systems = report["systems"]
    n = systems["llm_policy"]["n_cases"]
    outcomes = ACTIONS + [k for k in ("invalid", "missing")
                          if any(systems[m]["action_counts"][k] for m, _ in METHODS)]
    matrices = [
        [[systems[method]["confusion"][ref][out] for out in outcomes] for ref in ACTIONS]
        for method, _ in METHODS
    ]
    ceiling = max(max(row) for matrix in matrices for row in matrix)
    cmap = LinearSegmentedColormap.from_list("counts", ["#eff5f4", "#467f87", "#084651"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8), dpi=160)
    for ax, (_, label), matrix in zip(axes, METHODS, matrices):
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=ceiling)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(range(len(outcomes)), [s.title() for s in outcomes], fontsize=9)
        ax.set_yticks(range(len(ACTIONS)), [s.title() for s in ACTIONS], fontsize=9)
        ax.tick_params(length=0, pad=5)
        ax.spines[:].set_visible(False)
        for i, row in enumerate(matrix):
            for j, count in enumerate(row):
                ax.text(j, i, str(count), ha="center", va="center", fontsize=14,
                        color="white" if count > ceiling * .46 else "#12383d")
    axes[0].set_ylabel("Reference action", fontsize=10)
    fig.suptitle(f"How Qwen classified {n} held-out expenses", x=.065, y=.98,
                 ha="left", fontsize=16, fontweight="bold")
    fig.supxlabel("Model decision", fontsize=11, y=.055)
    fig.subplots_adjust(left=.10, right=.98, top=.73, bottom=.20, wspace=.37)
    fig.savefig(ROOT / "reports/decisions.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
