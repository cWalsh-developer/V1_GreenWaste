from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results_assets"
OUT.mkdir(parents=True, exist_ok=True)


def specialist_comparison() -> None:
    labels = ["Original", "45 reviewed", "91 reviewed", "91 + 135 pseudo"]
    metrics = {
        "Precision": [0.043, 0.882, 0.799, 0.860],
        "Recall": [0.925, 0.846, 0.981, 0.887],
        "mAP50": [0.256, 0.927, 0.943, 0.918],
        "mAP50–95": [0.074, 0.455, 0.574, 0.623],
    }
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = ["#4472C4", "#70AD47", "#ED7D31", "#A5A5A5"]
    for index, ((name, values), color) in enumerate(zip(metrics.items(), colors)):
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8, rotation=90)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_xticks(x, labels)
    ax.set_title("Tables and desks specialist-model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=4, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "tables_desks_specialist_model_comparison.png", dpi=300)
    plt.close(fig)


def lca_ranges() -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.8))

    disposal_names = ["Recycling", "Incineration", "Landfill"]
    disposal_low = np.array([0.118, 0.118, 7.533])
    disposal_high = np.array([0.182, 0.182, 12.459])
    disposal_mid = (disposal_low + disposal_high) / 2
    disposal_err = (disposal_high - disposal_low) / 2
    left.errorbar(disposal_mid, disposal_names, xerr=disposal_err, fmt="o", capsize=6,
                  color="#4472C4", ecolor="#4472C4", markersize=7)
    left.set_xlabel("Estimated emissions (kg CO₂e)")
    left.set_title("Treatment routes")
    left.grid(axis="x", alpha=0.25)

    reuse_low, reuse_high = -590.885, -289.330
    reuse_mid = (reuse_low + reuse_high) / 2
    reuse_err = (reuse_high - reuse_low) / 2
    right.errorbar([reuse_mid], ["Conditional reuse"], xerr=[reuse_err], fmt="o", capsize=6,
                   color="#70AD47", ecolor="#70AD47", markersize=7)
    right.axvline(0, color="black", linewidth=0.8)
    right.set_xlabel("Potential avoided emissions (kg CO₂e)")
    right.set_title("Conditional avoided-production proxy")
    right.grid(axis="x", alpha=0.25)

    fig.suptitle("CO₂e scenario ranges for capture_20260620_164635")
    fig.tight_layout()
    fig.savefig(OUT / "capture_20260620_164635_lca_scenarios.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    specialist_comparison()
    lca_ranges()
