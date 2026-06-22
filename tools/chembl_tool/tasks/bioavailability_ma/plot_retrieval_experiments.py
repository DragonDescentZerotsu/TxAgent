"""Create publication-style figures for Bioavailability_Ma retrieval ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyBboxPatch


DEFAULT_OUTPUT_DIR = Path(
    "outputs/chembl_tool/tasks/bioavailability_ma/figures"
)

TINOS_FONT_FILES = (
    Path("/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-Bold.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-Italic.ttf"),
    Path("/usr/share/fonts/truetype/croscore/Tinos-BoldItalic.ttf"),
)

COLORS = {
    "ink": "#17212B",
    "muted": "#5D6975",
    "grid": "#D9E0E5",
    "panel": "#F7F9FA",
    "strict": "#7C8793",
    "memory": "#A77956",
    "identity": "#C39A16",
    "knn": "#695AA6",
    "tier1": "#247BA0",
    "all": "#145374",
    "starling_v1": "#E39A8F",
    "starling": "#C94736",
    "combined_v1": "#72B8A7",
    "combined": "#087A60",
    "overlap": "#695AA6",
}

EXPERIMENTS = [
    {
        "short": "No retrieval\nstrict",
        "name": "Properties only, identity forbidden",
        "accuracy": 0.6562,
        "macro_f1": 0.6000,
        "high_precision": 0.8354,
        "high_recall": 0.6804,
        "low_recall": 0.5806,
        "color": COLORS["strict"],
        "historical": True,
    },
    {
        "short": "No retrieval\n+ memory",
        "name": "Properties only, memory comparison",
        "accuracy": 0.6953,
        "macro_f1": 0.6323,
        "high_precision": 0.8452,
        "high_recall": 0.7320,
        "low_recall": 0.5806,
        "color": COLORS["memory"],
        "historical": True,
    },
    {
        "short": "No retrieval\n+ identity",
        "name": "Properties only, identity allowed",
        "accuracy": 0.7422,
        "macro_f1": 0.6598,
        "high_precision": 0.8404,
        "high_recall": 0.8144,
        "low_recall": 0.5161,
        "color": COLORS["identity"],
        "historical": True,
    },
    {
        "short": "Starling v1\nKNN-3",
        "name": "Starling v1 numeric KNN-3 majority",
        "accuracy": 0.78125,
        "macro_f1": 0.702028,
        "high_precision": 0.85567,
        "high_recall": 0.85567,
        "low_recall": 0.548387,
        "color": COLORS["knn"],
        "historical": False,
        "method": "KNN",
    },
    {
        "short": "ChEMBL\nTier 1",
        "name": "ChEMBL Tier 1 retrieval",
        "accuracy": 0.7031,
        "macro_f1": 0.6673,
        "high_precision": 0.9041,
        "high_recall": 0.6804,
        "low_recall": 0.7742,
        "color": COLORS["tier1"],
        "historical": False,
    },
    {
        "short": "ChEMBL\nall tiers",
        "name": "ChEMBL all-tier retrieval",
        "accuracy": 0.7188,
        "macro_f1": 0.6762,
        "high_precision": 0.8861,
        "high_recall": 0.7216,
        "low_recall": 0.7097,
        "color": COLORS["all"],
        "historical": False,
    },
    {
        "short": "Starling\nv1",
        "name": "Starling-only retrieval v1",
        "accuracy": 0.7344,
        "macro_f1": 0.6909,
        "high_precision": 0.8987,
        "high_recall": 0.7320,
        "low_recall": 0.7419,
        "color": COLORS["starling_v1"],
        "historical": False,
    },
    {
        "short": "Starling\nv2",
        "name": "Starling-only retrieval v2",
        "accuracy": 0.7656,
        "macro_f1": 0.7342,
        "high_precision": 0.9467,
        "high_recall": 0.7320,
        "low_recall": 0.8710,
        "color": COLORS["starling"],
        "historical": False,
    },
    {
        "short": "Combined\nv1",
        "name": "ChEMBL Tier 1 + Starling v1",
        "accuracy": 0.7656,
        "macro_f1": 0.7149,
        "high_precision": 0.8941,
        "high_recall": 0.7835,
        "low_recall": 0.7097,
        "color": COLORS["combined_v1"],
        "historical": False,
    },
    {
        "short": "Combined\nv2",
        "name": "ChEMBL Tier 1 + Starling v2",
        "accuracy": 0.7656,
        "macro_f1": 0.7193,
        "high_precision": 0.9036,
        "high_recall": 0.7732,
        "low_recall": 0.7419,
        "color": COLORS["combined"],
        "historical": False,
    },
]


def configure_matplotlib() -> None:
    missing_fonts = [path for path in TINOS_FONT_FILES if not path.is_file()]
    if missing_fonts:
        missing = ", ".join(str(path) for path in missing_fonts)
        raise FileNotFoundError(
            f"Required Tinos font files are missing: {missing}. "
            "Install the Tinos font family before generating the figures."
        )
    for font_path in TINOS_FONT_FILES:
        font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "Tinos",
            "font.serif": ["Tinos"],
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelcolor": COLORS["ink"],
            "text.color": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "axes.edgecolor": COLORS["grid"],
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def style_axis(ax: plt.Axes, *, y_grid: bool = True) -> None:
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.tick_params(length=0)
    if y_grid:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.04,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
    )


def save_figure(
    fig: plt.Figure, output_path: Path, *, preview_png: bool
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="svg", bbox_inches="tight", facecolor="white")
    if preview_png:
        fig.savefig(
            output_path.with_suffix(".png"),
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def plot_performance(output_dir: Path, *, preview_png: bool) -> Path:
    fig = plt.figure(figsize=(18.5, 11.2), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], width_ratios=[1.45, 1])
    ax_overall = fig.add_subplot(grid[0, :])
    ax_class = fig.add_subplot(grid[1, 0])
    ax_gain = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "Oral Bioavailability: Retrieval-Source Ablation",
        x=0.03,
        y=1.035,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=COLORS["ink"],
    )
    fig.text(
        0.03,
        0.997,
        "Non-LLM KNN baseline and DeepSeek-v4-pro retrieval experiments on the same 128-molecule test split",
        ha="left",
        fontsize=11,
        color=COLORS["muted"],
    )

    x = np.arange(len(EXPERIMENTS))
    width = 0.34
    accuracy = [row["accuracy"] for row in EXPERIMENTS]
    macro_f1 = [row["macro_f1"] for row in EXPERIMENTS]
    colors = [row["color"] for row in EXPERIMENTS]
    bars_acc = ax_overall.bar(
        x - width / 2,
        accuracy,
        width,
        color=colors,
        alpha=0.42,
        edgecolor=colors,
        linewidth=1.2,
        label="Accuracy",
    )
    bars_f1 = ax_overall.bar(
        x + width / 2,
        macro_f1,
        width,
        color=colors,
        edgecolor=colors,
        linewidth=1.2,
        label="Macro-F1",
    )
    ax_overall.axvspan(-0.55, 2.55, color=COLORS["panel"], zorder=-2)
    ax_overall.axvspan(2.75, 3.55, color="#F2F0F8", zorder=-2)
    ax_overall.axvline(2.75, color=COLORS["grid"], linewidth=1.2)
    ax_overall.axvline(3.75, color=COLORS["grid"], linewidth=1.2)
    ax_overall.text(
        1,
        0.818,
        "Historical no-retrieval references",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    ax_overall.text(
        3,
        0.818,
        "Non-LLM\nbaseline",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    ax_overall.text(
        6.5,
        0.818,
        "DeepSeek retrieval-source experiments",
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    for bars in (bars_acc, bars_f1):
        for bar in bars:
            ax_overall.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=COLORS["ink"],
            )
    ax_overall.set_ylim(0.54, 0.835)
    ax_overall.set_yticks(np.arange(0.55, 0.81, 0.05))
    ax_overall.set_ylabel("Score")
    ax_overall.set_xticks(x, [row["short"] for row in EXPERIMENTS])
    ax_overall.tick_params(axis="x", pad=9)
    ax_overall.legend(frameon=False, loc="upper left", ncol=2)
    ax_overall.set_title(
        "Overall performance", loc="left", pad=14
    )
    style_axis(ax_overall)
    add_panel_label(ax_overall, "A")

    retrieval = EXPERIMENTS[3:]
    y = np.arange(len(retrieval))
    metric_specs = [
        ("high_precision", "High-BA precision", "o"),
        ("high_recall", "High-BA recall", "s"),
        ("low_recall", "Low-BA recall", "^"),
    ]
    offsets = [-0.18, 0, 0.18]
    metric_colors = [COLORS["all"], COLORS["combined"], COLORS["starling"]]
    for (key, label, marker), offset, color in zip(
        metric_specs, offsets, metric_colors
    ):
        values = [row[key] for row in retrieval]
        ax_class.scatter(
            values,
            y + offset,
            s=62,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=label,
            zorder=3,
        )
        for value, y_pos in zip(values, y + offset):
            ax_class.text(
                value + 0.008,
                y_pos,
                f"{value:.3f}",
                va="center",
                fontsize=8,
                color=COLORS["muted"],
            )
    ax_class.set_xlim(0.52, 0.98)
    ax_class.set_xticks(np.arange(0.55, 1.00, 0.05))
    ax_class.set_yticks(y, [row["short"].replace("\n", " ") for row in retrieval])
    ax_class.invert_yaxis()
    ax_class.set_xlabel("Class-specific score")
    ax_class.set_title("Where retrieval changes the decision profile", loc="left")
    ax_class.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=8.5,
    )
    style_axis(ax_class, y_grid=False)
    ax_class.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax_class.set_axisbelow(True)
    add_panel_label(ax_class, "B")

    strict_f1 = EXPERIMENTS[0]["macro_f1"]
    gains = [(row["macro_f1"] - strict_f1) * 100 for row in EXPERIMENTS[1:]]
    gain_labels = [row["short"].replace("\n", " ") for row in EXPERIMENTS[1:]]
    gain_colors = [row["color"] for row in EXPERIMENTS[1:]]
    gain_y = np.arange(len(gains))
    gain_bars = ax_gain.barh(gain_y, gains, color=gain_colors, height=0.58)
    for bar, gain in zip(gain_bars, gains):
        ax_gain.text(
            gain + 0.25,
            bar.get_y() + bar.get_height() / 2,
            f"+{gain:.1f} pp",
            va="center",
            fontsize=9,
            fontweight="bold" if gain == max(gains) else "normal",
        )
    ax_gain.set_yticks(gain_y, gain_labels)
    ax_gain.invert_yaxis()
    ax_gain.set_xlim(0, 15)
    ax_gain.set_xlabel("Macro-F1 gain vs strict no-retrieval baseline")
    ax_gain.set_title("Incremental value over the strict baseline", loc="left")
    style_axis(ax_gain, y_grid=False)
    ax_gain.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax_gain.set_axisbelow(True)
    add_panel_label(ax_gain, "C")

    fig.text(
        0.03,
        -0.025,
        "Key result: Starling v1 KNN-3 has the highest accuracy (0.781) but is high-class biased; "
        "Starling v2 DeepSeek has the highest macro-F1 (0.734) and low-BA recall (0.871).",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["knn"],
    )
    fig.text(
        0.03,
        -0.055,
        "Caveat: the three no-retrieval runs are historical Intern-S1/TRIM "
        "references with an older prompt/tool workflow. KNN-3 uses clean numeric Starling v1, "
        "unweighted majority vote, and no minimum similarity; DeepSeek retrieval runs use "
        "top-k=3 and minimum Tanimoto=0.3. No repeated-run variance estimate.",
        fontsize=9,
        color=COLORS["muted"],
    )

    output = output_dir / "oral_bioavailability_retrieval_experiments.svg"
    save_figure(fig, output, preview_png=preview_png)
    return output


def metric_card(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    value: str,
    label: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.03",
        transform=ax.transAxes,
        linewidth=1,
        edgecolor=COLORS["grid"],
        facecolor=COLORS["panel"],
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + 0.04,
        xy[1] + height * 0.62,
        value,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        color=color,
        va="center",
    )
    ax.text(
        xy[0] + 0.04,
        xy[1] + height * 0.27,
        label,
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["muted"],
        va="center",
    )


def plot_overlap(output_dir: Path, *, preview_png: bool) -> Path:
    chembl_total = 21_321
    starling_total = 17_215
    shared = 8_365
    chembl_only = 12_956
    starling_only = 8_850
    union = 30_171
    starling_covered = shared / starling_total
    chembl_covered = shared / chembl_total
    jaccard = shared / union

    fig = plt.figure(figsize=(15, 7.8), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.15, 1])
    ax_venn = fig.add_subplot(grid[0, 0])
    ax_stats = fig.add_subplot(grid[0, 1])

    fig.suptitle(
        "Retrieval-Library Molecule Overlap",
        x=0.03,
        y=1.065,
        ha="left",
        fontsize=22,
        fontweight="bold",
    )
    fig.text(
        0.03,
        1.015,
        "Combined v2 index: ChEMBL Tier 1 and Starling numeric + qualitative evidence",
        ha="left",
        fontsize=11,
        color=COLORS["muted"],
    )

    ax_venn.set_xlim(0, 10)
    ax_venn.set_ylim(0, 7.5)
    ax_venn.set_aspect("equal")
    ax_venn.axis("off")
    ax_venn.set_title("Unique molecular connectivity", loc="left", pad=12)
    add_panel_label(ax_venn, "A")

    chembl_circle = Circle(
        (4.0, 3.7),
        2.65,
        facecolor=COLORS["tier1"],
        edgecolor=COLORS["tier1"],
        alpha=0.62,
        linewidth=2,
    )
    starling_circle = Circle(
        (6.25, 3.7),
        2.08,
        facecolor=COLORS["starling"],
        edgecolor=COLORS["starling"],
        alpha=0.62,
        linewidth=2,
    )
    ax_venn.add_patch(chembl_circle)
    ax_venn.add_patch(starling_circle)

    ax_venn.text(
        2.2,
        6.75,
        f"ChEMBL Tier 1\n{chembl_total:,} entities",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["tier1"],
    )
    ax_venn.text(
        7.55,
        6.25,
        f"Starling v2\n{starling_total:,} entities",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color=COLORS["starling"],
    )
    region_labels = [
        (2.75, chembl_only, "ChEMBL-only"),
        (5.12, shared, "shared"),
        (7.15, starling_only, "Starling-only"),
    ]
    for x_pos, value, label in region_labels:
        ax_venn.text(
            x_pos,
            3.85,
            f"{value:,}",
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color="white",
        )
        ax_venn.text(
            x_pos,
            3.33,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
        )
    ax_venn.text(
        5.12,
        2.55,
        "same InChIKey\nconnectivity layer",
        ha="center",
        fontsize=8.5,
        color="white",
    )

    ax_stats.axis("off")
    ax_stats.set_title("Coverage and source complementarity", loc="left", pad=12)
    add_panel_label(ax_stats, "B")
    metric_card(
        ax_stats,
        (0.00, 0.70),
        0.47,
        0.20,
        f"{starling_covered:.2%}",
        "Starling covered by ChEMBL",
        COLORS["tier1"],
    )
    metric_card(
        ax_stats,
        (0.51, 0.70),
        0.47,
        0.20,
        f"{chembl_covered:.2%}",
        "ChEMBL covered by Starling",
        COLORS["starling"],
    )
    metric_card(
        ax_stats,
        (0.00, 0.45),
        0.47,
        0.20,
        f"{jaccard:.2%}",
        "Jaccard similarity",
        COLORS["overlap"],
    )
    metric_card(
        ax_stats,
        (0.51, 0.45),
        0.47,
        0.20,
        f"{union:,}",
        "Unique molecules in union",
        COLORS["combined"],
    )

    ax_stats.text(
        0,
        0.355,
        "Composition of the union",
        transform=ax_stats.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    segments = [
        (chembl_only, COLORS["tier1"], "ChEMBL-only"),
        (shared, COLORS["overlap"], "Shared"),
        (starling_only, COLORS["starling"], "Starling-only"),
    ]
    left = 0.0
    for value, color, label in segments:
        fraction = value / union
        ax_stats.barh(
            [0.29],
            [fraction],
            left=[left],
            height=0.055,
            color=color,
            transform=ax_stats.transAxes,
            clip_on=False,
        )
        if fraction > 0.19:
            ax_stats.text(
                left + fraction / 2,
                0.29,
                f"{fraction:.1%}",
                transform=ax_stats.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
        left += fraction
    legend_y = 0.215
    legend_x = [0.0, 0.35, 0.64]
    for x_pos, (value, color, label) in zip(legend_x, segments):
        ax_stats.scatter(
            [x_pos],
            [legend_y],
            s=55,
            color=color,
            transform=ax_stats.transAxes,
            clip_on=False,
        )
        ax_stats.text(
            x_pos + 0.025,
            legend_y,
            f"{label}: {value:,}",
            transform=ax_stats.transAxes,
            va="center",
            fontsize=8.7,
            color=COLORS["muted"],
        )

    ax_stats.text(
        0,
        0.105,
        "Interpretation",
        transform=ax_stats.transAxes,
        fontsize=11,
        fontweight="bold",
    )
    ax_stats.text(
        0,
        0.025,
        "After connectivity-level merging, Starling v2 adds 8,850 entities absent "
        "from ChEMBL Tier 1, including qualitative-only molecules. ChEMBL contributes "
        "12,956 entities absent from Starling.",
        transform=ax_stats.transAxes,
        fontsize=9.5,
        color=COLORS["muted"],
        wrap=True,
        va="bottom",
    )

    fig.text(
        0.03,
        -0.035,
        "Identity definition: first block of the standard InChIKey "
        "(connectivity layer), matching retrieval exact-query exclusion. "
        "Counts are current combined-v2 merge entities. ChEMBL includes direct and "
        "context-dependent Tier 1; Starling includes numeric and qualitative/contextual evidence.",
        fontsize=9,
        color=COLORS["muted"],
    )
    fig.text(
        0.03,
        -0.066,
        "Source canonical-SMILES counts before connectivity merging are larger "
        "(ChEMBL 21,860; Starling 18,117). Shared/only counts use InChIKey "
        "connectivity with canonical-SMILES fallback.",
        fontsize=9,
        color=COLORS["muted"],
    )

    output = output_dir / "starling_chembl_tier1_molecule_overlap.svg"
    save_figure(fig, output, preview_png=preview_png)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--preview-png",
        action="store_true",
        help="Also emit PNG previews for visual inspection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib()
    outputs = [
        plot_performance(args.output_dir, preview_png=args.preview_png),
        plot_overlap(args.output_dir, preview_png=args.preview_png),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
