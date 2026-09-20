#!/usr/bin/env python3
"""Render a validated staged-comparison figure from TSV data and a JSON spec."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9")
MARKERS = ("o", "s", "^", "D", "P", "X")
STATUSES = {"complete", "provisional", "reference"}
REQUIRED_COLUMNS = {
    "panel", "kind", "series", "stage", "value", "n", "status", "lower", "upper", "source"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(items: list[dict], name: str) -> dict[str, dict]:
    result = {str(item["id"]): item for item in items}
    if len(result) != len(items) or not all(result):
        raise ValueError(f"{name} IDs must be non-empty and unique")
    return result


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Data TSV is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Data TSV has no rows")
    for row in rows:
        if row["kind"] not in {"curve", "baseline"}:
            raise ValueError(f"Invalid kind: {row['kind']!r}")
        if row["status"] not in STATUSES:
            raise ValueError(f"Invalid status: {row['status']!r}")
        row["value"] = float(row["value"])
        if not math.isfinite(row["value"]):
            raise ValueError("Values must be finite")
        row["n"] = None if not row["n"] else int(row["n"])
        if row["n"] is not None and row["n"] <= 0:
            raise ValueError("Panel denominators must be positive")
        if not row["source"]:
            raise ValueError("Every row needs a source")
        bounds = (row["lower"], row["upper"])
        if bool(bounds[0]) != bool(bounds[1]):
            raise ValueError("lower and upper must be supplied together")
        row["lower"], row["upper"] = (
            (None, None) if not bounds[0] else (float(bounds[0]), float(bounds[1]))
        )
        if row["lower"] is not None and not all(math.isfinite(value) for value in (row["lower"], row["upper"])):
            raise ValueError("Uncertainty bounds must be finite")
        if row["lower"] is not None and not row["lower"] <= row["value"] <= row["upper"]:
            raise ValueError("Uncertainty bounds must contain the value")
    return rows


def resolved_styles(items: list[dict], *, baseline: bool = False) -> dict[str, dict]:
    styles = {}
    for index, item in enumerate(items):
        styles[item["id"]] = {
            "label": item["label"],
            "tick_label": item.get("tick_label", item["label"]),
            "color": item.get("color", "#666666" if baseline else COLORS[index % len(COLORS)]),
            "marker": item.get("marker", MARKERS[index % len(MARKERS)]),
            "linestyle": item.get("linestyle", "none" if baseline else ("--" if index == 2 else "-")),
        }
    return styles


def validate(rows: list[dict], spec: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    panels, stages = spec.get("panels") or [], spec.get("stages") or []
    curves, baselines = spec.get("curves") or [], spec.get("baselines") or []
    panel_map = unique(panels, "panel")
    stage_map = unique(stages, "stage")
    curve_map = unique(curves, "curve")
    baseline_map = unique(baselines, "baseline")
    if not panels or not stages or not curves:
        raise ValueError("Spec requires at least one panel, stage, and curve")

    seen = set()
    for row in rows:
        if row["panel"] not in panel_map:
            raise ValueError(f"Unknown panel: {row['panel']}")
        allowed = curve_map if row["kind"] == "curve" else baseline_map
        if row["series"] not in allowed:
            raise ValueError(f"Unknown {row['kind']} series: {row['series']}")
        if row["kind"] == "curve" and row["stage"] not in stage_map:
            raise ValueError(f"Unknown stage: {row['stage']}")
        if row["kind"] == "baseline" and row["stage"]:
            raise ValueError("Baseline rows must have an empty stage")
        key = (row["panel"], row["kind"], row["series"], row["stage"])
        if key in seen:
            raise ValueError(f"Duplicate row: {key}")
        seen.add(key)

    expected = {
        (panel, "curve", curve, stage)
        for panel in panel_map for curve in curve_map for stage in stage_map
    } | {
        (panel, "baseline", baseline, "")
        for panel in panel_map for baseline in baseline_map
    }
    if seen != expected:
        raise ValueError(f"Missing rows: {sorted(expected - seen)}; extra rows: {sorted(seen - expected)}")
    for panel in panel_map:
        denominators = {row["n"] for row in rows if row["panel"] == panel}
        if len(denominators) != 1:
            raise ValueError(f"Panel {panel} has mixed denominators: {denominators}")
    return panels, stages, curves, {
        "curves": resolved_styles(curves),
        "baselines": resolved_styles(baselines, baseline=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/comparison-figure"))
    parser.add_argument("--output-stem", default="comparison")
    args = parser.parse_args()

    rows = load_rows(args.data)
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    panels, stages, curves, styles = validate(rows, spec)
    baselines = spec.get("baselines") or []
    lookup = {
        (row["panel"], row["kind"], row["series"], row["stage"]): row for row in rows
    }
    values = [
        bound
        for row in rows
        for bound in (
            row["value"] if row["lower"] is None else row["lower"],
            row["value"] if row["upper"] is None else row["upper"],
        )
    ]
    if spec.get("y_limits") is not None:
        limits = tuple(float(value) for value in spec["y_limits"])
        if len(limits) != 2 or not limits[0] <= min(values) <= max(values) <= limits[1]:
            raise ValueError("y_limits must contain every value and uncertainty bound")
    else:
        span = max(values) - min(values) or max(abs(min(values)), 1.0) * 0.1
        limits = (min(values) - 0.08 * span, max(values) + 0.12 * span)

    columns, grid_rows = min(3, len(panels)), math.ceil(len(panels) / min(3, len(panels)))
    handles_count = len(curves) + len(baselines) + int(any(row["status"] == "provisional" for row in rows))
    legend_rows = math.ceil(handles_count / min(4, max(1, handles_count)))
    figure_height = 4.0 * grid_rows + 2.8 + 0.38 * legend_rows
    fig, axes = plt.subplots(
        grid_rows, columns, figsize=(6.2 * columns, figure_height), sharey=True, squeeze=False
    )
    stage_x = list(range(len(stages)))
    baseline_x = [len(stages) + 0.75 + 0.65 * index for index in range(len(baselines))]

    for ax, panel in zip(axes.flat, panels):
        if baselines:
            ax.axvspan(len(stages) - 0.35, baseline_x[-1] + 0.4, color="#F4F4F4", zorder=0)
            ax.axvline(len(stages) - 0.35, color="#BBBBBB", linewidth=0.8)
        for curve in curves:
            style = styles["curves"][curve["id"]]
            selected = [lookup[(panel["id"], "curve", curve["id"], stage["id"])] for stage in stages]
            y = [row["value"] for row in selected]
            ax.plot(stage_x, y, color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], linewidth=2, markersize=6, zorder=3)
            for x, row in zip(stage_x, selected):
                if row["lower"] is not None:
                    ax.errorbar(x, row["value"], yerr=[[row["value"] - row["lower"]],
                                [row["upper"] - row["value"]]], color=style["color"],
                                capsize=3, linewidth=1.2, zorder=2)
                if row["status"] == "provisional":
                    ax.plot(x, row["value"], marker=style["marker"], linestyle="none",
                            markerfacecolor="white", markeredgecolor=style["color"],
                            markeredgewidth=2.2, markersize=7, zorder=4)
        for x, baseline in zip(baseline_x, baselines):
            style = styles["baselines"][baseline["id"]]
            row = lookup[(panel["id"], "baseline", baseline["id"], "")]
            face = "white" if row["status"] == "provisional" else style["color"]
            ax.plot(x, row["value"], marker=style["marker"], linestyle="none",
                    markerfacecolor=face, markeredgecolor=style["color"],
                    markeredgewidth=2.2 if row["status"] == "provisional" else 1.2,
                    markersize=7, zorder=3)
            if row["lower"] is not None:
                ax.errorbar(x, row["value"], yerr=[[row["value"] - row["lower"]],
                            [row["upper"] - row["value"]]], color=style["color"],
                            capsize=3, linewidth=1.2, zorder=2)

        denominator = next(row["n"] for row in rows if row["panel"] == panel["id"])
        suffix = "" if denominator is None else f"  ·  n = {denominator}"
        ax.set_title(f"{panel['label']}{suffix}", loc="left", fontsize=12, fontweight="bold", pad=22)
        ax.text(sum(stage_x) / len(stage_x), 1.025, spec.get("curve_section_label", "Experiments"),
                transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color="#666666")
        if baselines:
            ax.text(sum(baseline_x) / len(baseline_x), 1.025,
                    spec.get("baseline_section_label", "Baselines"),
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=9, color="#666666")
        ticks = stage_x + baseline_x
        labels = [stage["label"] for stage in stages] + [styles["baselines"][item["id"]]["tick_label"] for item in baselines]
        ax.set_xticks(ticks, labels, fontsize=8)
        ax.set_xlim(-0.3, (baseline_x[-1] if baselines else stage_x[-1]) + 0.4)
        ax.set_ylim(*limits)
        ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(spec.get("y_label", "Value"), fontsize=11)

    handles = [Line2D([0], [0], color=styles["curves"][item["id"]]["color"],
                      marker=styles["curves"][item["id"]]["marker"],
                      linestyle=styles["curves"][item["id"]]["linestyle"], label=item["label"])
               for item in curves]
    handles += [Line2D([0], [0], color=styles["baselines"][item["id"]]["color"],
                       marker=styles["baselines"][item["id"]]["marker"],
                       linestyle="none", label=item["label"])
                for item in baselines]
    if any(row["status"] == "provisional" for row in rows):
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white",
                              markeredgecolor="#444444", markeredgewidth=2.2, label="Provisional point"))

    fig.suptitle(spec["title"], x=0.055, y=0.975, ha="left", fontsize=19, fontweight="bold")
    if spec.get("subtitle"):
        fig.text(0.055, 0.925, spec["subtitle"], fontsize=10, color="#555555")
    note_y = 0.13 + 0.025 * legend_rows
    if spec.get("figure_note"):
        fig.text(0.055, note_y, spec["figure_note"], fontsize=9, color="#555555", linespacing=1.5)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.015),
               ncol=min(4, len(handles)), frameon=False, fontsize=9)
    bottom = min(0.38, 0.20 + 0.035 * legend_rows + (0.04 if spec.get("figure_note") else 0))
    fig.subplots_adjust(left=0.055, right=0.99, top=0.82, bottom=bottom,
                        hspace=0.55, wspace=0.14)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for extension in ("png", "pdf", "svg"):
        path = args.output_dir / f"{args.output_stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        outputs[extension] = str(path)
    plt.close(fig)
    data_output = args.output_dir / f"{args.output_stem}.tsv"
    shutil.copyfile(args.data, data_output)
    outputs["tsv"] = str(data_output)
    receipt = {
        "schema_version": "staged_comparison_figure.v1",
        "data": {"path": str(args.data), "sha256": sha256_file(args.data)},
        "spec": {"path": str(args.spec), "sha256": sha256_file(args.spec)},
        "row_count": len(rows),
        "styles": styles,
        "y_limits": limits,
        "outputs": outputs,
        "output_hashes": {name: sha256_file(Path(path)) for name, path in outputs.items()},
    }
    receipt_path = args.output_dir / f"{args.output_stem}_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**outputs, "receipt": str(receipt_path)}, indent=2))


if __name__ == "__main__":
    main()
