"""Shared visual primitives for paper-experiment SVG figures."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
from html import escape
from pathlib import Path
import shutil
import subprocess


FONT = "Inter, DejaVu Sans, Arial, sans-serif"
BG = "#F7F8FA"
CARD = "#FFFFFF"
INK = "#172033"
MUTED = "#5E6878"
GRID = "#D9DEE7"
BLIND = "#2F6FB0"
VISIBLE = "#D95F3D"
PARENT = "#7A8338"
NEUTRAL = "#8993A2"
PURPLE = "#6557A4"
GOLD = "#A66B12"
POSITIVE = "#177A58"
NEGATIVE = "#B54848"


def export_png(svg_path: Path, png_path: Path) -> None:
    """Export a canonical SVG through ImageMagick when PNG is requested."""
    converter = shutil.which("convert")
    if converter is None:
        raise RuntimeError("ImageMagick 'convert' is required for --png-output")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [converter, "-background", "white", str(svg_path), str(png_path)],
        check=True,
    )


def run_metric_plot_cli(
    *,
    description: str | None,
    render: Callable[[Path, Path], None],
    default_metrics: Path,
    default_output: Path,
) -> None:
    """Run the standard metrics-to-SVG CLI shared by paper figures."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--metrics", type=Path, default=default_metrics)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--png-output", type=Path)
    args = parser.parse_args()
    render(args.metrics, args.output)
    print(args.output)
    if args.png_output is not None:
        export_png(args.output, args.png_output)
        print(args.png_output)


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 16,
    weight: int = 400,
    fill: str = INK,
    anchor: str = "start",
    spacing: float | None = None,
) -> str:
    attrs = [
        f'x="{x:g}"',
        f'y="{y:g}"',
        f'font-family="{FONT}"',
        f'font-size="{size}"',
        f'font-weight="{weight}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
    ]
    if spacing is not None:
        attrs.append(f'letter-spacing="{spacing:g}"')
    return f"<text {' '.join(attrs)}>{escape(value)}</text>"


def multiline(
    x: float,
    y: float,
    lines: Iterable[str],
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = MUTED,
    line_height: int = 20,
) -> str:
    spans = [
        f'<tspan x="{x:g}" dy="{0 if index == 0 else line_height:g}">{escape(line)}</tspan>'
        for index, line in enumerate(lines)
    ]
    return (
        f'<text x="{x:g}" y="{y:g}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="start">'
        + "".join(spans)
        + "</text>"
    )


def rect(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: str,
    stroke: str = "none",
    rx: int = 8,
) -> str:
    return (
        f'<rect x="{x:g}" y="{y:g}" width="{width:g}" height="{height:g}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
    )


def append_vertical_grid(
    parts: list[str],
    *,
    plot_left: float,
    plot_right: float,
    plot_top: float,
    plot_bottom: float,
    ticks: Iterable[float],
    scale_max: float,
    label_y: float,
) -> None:
    """Append the shared vertical grid and numeric tick labels for bar charts."""
    if scale_max <= 0:
        raise ValueError("scale_max must be positive")
    for tick in ticks:
        tick_x = plot_left + tick / scale_max * (plot_right - plot_left)
        parts.append(
            f'<line x1="{tick_x:.1f}" y1="{plot_top}" x2="{tick_x:.1f}" '
            f'y2="{plot_bottom}" stroke="{GRID}" stroke-width="1"/>'
        )
        parts.append(
            svg_text(
                tick_x,
                label_y,
                f"{tick:.1f}",
                size=12,
                fill=MUTED,
                anchor="middle",
            )
        )
