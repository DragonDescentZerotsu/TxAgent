"""Shared visual primitives for paper-experiment SVG figures."""

from __future__ import annotations

from html import escape
from typing import Iterable


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
