"""
Ø100 mm ウェハ（オリフラ 32.5 mm）の SVG マップとクラスタ分布表示。

幾何は `wafer-contour-studio` の waferGeometry.ts と同じ定義
（https://github.com/ks250206/wafer-contour-studio）。
座標系: ウェハ中心が原点、x・y は mm（ファイル名パース値と一致）。
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import numpy as np
import polars as pl

from rasx_cluster_analyzer.palette import CLUSTER_PALETTE

WAFER_DIAMETER_MM: float = 100.0
ORIFLA_LENGTH_MM: float = 32.5
WAFER_MAP_PADDING: float = 22.0
DEFAULT_WAFER_SVG_SIZE: int = 300


def _legend_name(label: int) -> str:
    return "noise" if int(label) == -1 else f"cluster {int(label)}"


def _color_map(labels: np.ndarray) -> dict[int, str]:
    uniq = sorted({int(x) for x in np.unique(labels)})
    out: dict[int, str] = {}
    used = 0
    for lab in uniq:
        if lab == -1:
            out[lab] = "#BDBDBD"
        else:
            out[lab] = CLUSTER_PALETTE[used % len(CLUSTER_PALETTE)]
            used += 1
    return out


def get_wafer_flat_y(diameter: float, flat_length: float = ORIFLA_LENGTH_MM) -> float:
    """オリフラ弦の y（ウェハ座標）。負側がフラット側（TS の getWaferFlatY と同じ）。"""
    radius = diameter / 2.0
    half_flat = flat_length / 2.0
    return -math.sqrt(radius**2 - half_flat**2)


def is_inside_wafer(
    x: float,
    y: float,
    diameter: float = WAFER_DIAMETER_MM,
    flat_length: float = ORIFLA_LENGTH_MM,
) -> bool:
    """ウェハ有効領域内（円弧＋フラットで切った領域）か。"""
    radius = diameter / 2.0
    flat_y = get_wafer_flat_y(diameter, flat_length)
    return x * x + y * y <= radius * radius and y >= flat_y


def get_wafer_svg_path_d(
    center_x: float,
    center_y: float,
    px_per_unit_x: float,
    px_per_unit_y: float,
    diameter: float = WAFER_DIAMETER_MM,
    flat_length: float = ORIFLA_LENGTH_MM,
    segments: int = 180,
) -> str:
    """ウェハ外形の SVG path ``d``（閉路）。"""
    radius = diameter / 2.0
    half_flat = flat_length / 2.0
    flat_y = get_wafer_flat_y(diameter, flat_length)
    start_angle = math.atan2(flat_y, half_flat)
    end_angle = math.atan2(flat_y, -half_flat) + 2.0 * math.pi
    pts: list[str] = []
    for i in range(segments + 1):
        t = i / segments
        ang = start_angle + (end_angle - start_angle) * t
        x = center_x + math.cos(ang) * radius * px_per_unit_x
        y = center_y - math.sin(ang) * radius * px_per_unit_y
        pts.append(f"{x:.6g} {y:.6g}")
    return "M " + " L ".join(pts) + " Z"


def build_wafer_cluster_map_panel_html(
    labels: np.ndarray,
    meta: pl.DataFrame,
    *,
    size: int = DEFAULT_WAFER_SVG_SIZE,
    diameter_mm: float = WAFER_DIAMETER_MM,
    flat_length_mm: float = ORIFLA_LENGTH_MM,
) -> str:
    """
    埋め込み図の右に置く HTML 断片（見出し＋SVG＋凡例）。

    ``meta`` に ``x_mm``, ``y_mm``, ``path`` 列が必要。
    """
    names = [Path(p).name for p in meta["path"].to_list()]
    x_mm = np.asarray(meta["x_mm"].to_numpy(), dtype=np.float64)
    y_mm = np.asarray(meta["y_mm"].to_numpy(), dtype=np.float64)
    lab_int = labels.astype(np.int64, copy=False)
    cmap = _color_map(lab_int)

    padding = WAFER_MAP_PADDING
    display_d = float(size) - padding * 2.0
    px_per_unit = display_d / diameter_mm
    cx = float(size) / 2.0
    cy = float(size) / 2.0
    radius = diameter_mm / 2.0

    outline_d = get_wafer_svg_path_d(cx, cy, px_per_unit, px_per_unit, diameter_mm, flat_length_mm)

    grid_lines: list[float] = []
    v = -radius
    while v <= radius + 1e-9:
        grid_lines.append(round(v, 6))
        v += 10.0
    ruler_vals = [v for v in grid_lines if -radius < v < radius]

    plot_x1 = padding
    plot_x2 = float(size) - padding
    plot_y1 = padding
    plot_y2 = float(size) - padding
    axis_inset = 5.0

    grid_elems: list[str] = []
    for gv in grid_lines:
        x_line = cx + gv * px_per_unit
        grid_elems.append(
            f'<line x1="{x_line:.6g}" y1="{plot_y1:.6g}" x2="{x_line:.6g}" '
            f'y2="{plot_y2:.6g}" stroke="#CCCCCC" stroke-width="0.6"/>'
        )
    for gv in grid_lines:
        y_line = cy - gv * px_per_unit
        grid_elems.append(
            f'<line x1="{plot_x1:.6g}" y1="{y_line:.6g}" x2="{plot_x2:.6g}" '
            f'y2="{y_line:.6g}" stroke="#CCCCCC" stroke-width="0.6"/>'
        )

    label_elems: list[str] = []
    for gv in ruler_vals:
        x_px = cx + gv * px_per_unit
        label_elems.append(
            f'<text x="{x_px:.6g}" y="{size - padding + 14:.6g}" text-anchor="middle" '
            f'font-size="9" fill="#444">{gv:.0f}</text>'
        )
    for gv in ruler_vals:
        y_px = cy - gv * px_per_unit
        label_elems.append(
            f'<text x="{padding - 6:.6g}" y="{y_px + 3:.6g}" text-anchor="end" '
            f'font-size="9" fill="#444">{gv:.0f}</text>'
        )

    point_elems: list[str] = []
    for i in range(len(names)):
        xm = float(x_mm[i])
        ym = float(y_mm[i])
        lab = int(lab_int[i])
        px = cx + xm * px_per_unit
        py = cy - ym * px_per_unit
        fill = cmap[lab]
        inside = is_inside_wafer(xm, ym, diameter_mm, flat_length_mm)
        opacity = "1" if inside else "0.45"
        stroke = "#333333" if inside else "#E65100"
        title = (
            f"{names[i]}\n{_legend_name(lab)}\nx={xm:.3f} mm, y={ym:.3f} mm"
            + ("" if inside else "\n(outside wafer outline)")
        )
        point_elems.append(
            f'<circle cx="{px:.6g}" cy="{py:.6g}" r="4.2" fill="{escape(fill)}" '
            f'stroke="{stroke}" stroke-width="0.7" opacity="{opacity}">'
            f"<title>{escape(title)}</title></circle>"
        )

    uniq_labels = sorted({int(x) for x in np.unique(lab_int)})
    legend_items: list[str] = []
    for lab in uniq_labels:
        c = cmap[lab]
        nm = escape(_legend_name(lab))
        legend_items.append(
            f'<span class="rasx-wafer-legend-item">'
            f'<span class="rasx-wafer-swatch" style="background:{escape(c)}"></span>'
            f"{nm}</span>"
        )

    grid_svg = "\n    ".join(grid_elems)
    labels_svg = "\n    ".join(label_elems)
    points_svg = "\n    ".join(point_elems)

    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}" role="img" '
        f'aria-label="Wafer map: measurement points colored by cluster">'
    )
    # グリッド矩形内：上辺より少し下、左寄せ
    ax_y_y = plot_y1 + 12.0
    ax_y = (
        f'<text x="{plot_x1 + axis_inset:.6g}" y="{ax_y_y:.6g}" text-anchor="start" '
        f'font-size="10" fill="#333">↑ y (mm)</text>'
    )
    # グリッド矩形内：下辺より少し上、右寄せ（目盛テキストは +14 で枠外）
    ax_x_y = plot_y2 - 7.0
    ax_x = (
        f'<text x="{plot_x2 - axis_inset:.6g}" y="{ax_x_y:.6g}" text-anchor="end" '
        f'font-size="10" fill="#333">x (mm) →</text>'
    )
    svg = f"""{svg_open}
    <path d="{outline_d}" fill="#EEEEEE" stroke="#222222" stroke-width="1.2" />
    <g aria-hidden="true">
    {grid_svg}
    </g>
    {points_svg}
    {labels_svg}
    {ax_y}
    {ax_x}
  </svg>"""

    legend_html = (
        '<div class="rasx-wafer-legend" role="list">' + "".join(legend_items) + "</div>"
    )

    return f"""<section class="rasx-wafer-panel">
  <h2 class="rasx-wafer-title">Wafer map (clusters)</h2>
  <div class="rasx-wafer-svg-wrap">
    {svg}
  </div>
  {legend_html}
</section>"""
