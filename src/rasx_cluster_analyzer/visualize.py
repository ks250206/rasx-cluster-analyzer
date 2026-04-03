"""Plotly HTML figures for publication-style cluster maps."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
from plotly.subplots import make_subplots

from rasx_cluster_analyzer.config import AppConfig
from rasx_cluster_analyzer.palette import CLUSTER_PALETTE
from rasx_cluster_analyzer.wafer_svg import build_wafer_cluster_map_panel_html

_EMBED_WIDTH = 900
_EMBED_HEIGHT = 520
_XRD_WIDTH = 920
_XRD_BLOCK_GAP_PX = 24

# レポート HTML 末尾に付与（f-string と分離し { } をエスケープ不要にする）
_REPORT_SIDEBAR_SCRIPT = """
<script>
(function () {
  var aside = document.getElementById("rasx-meta-sidebar");
  var openBtn = document.getElementById("rasx-sidebar-open");
  var closeBtn = document.getElementById("rasx-sidebar-close");
  var backdrop = document.getElementById("rasx-sidebar-backdrop");
  var foldBtn = document.getElementById("rasx-sidebar-fold");
  var unfoldBtn = document.getElementById("rasx-sidebar-unfold");
  if (!aside || !openBtn || !backdrop || !foldBtn || !unfoldBtn) return;
  var mq = window.matchMedia("(max-width: 1100px)");
  function isNarrow() {
    return mq.matches;
  }
  function syncBackdrop() {
    if (isNarrow()) {
      backdrop.hidden = !aside.classList.contains("rasx-sidebar--open");
    } else {
      backdrop.hidden = true;
    }
  }
  function onViewportChange() {
    if (isNarrow()) {
      aside.classList.remove("rasx-meta--folded");
      aside.classList.remove("rasx-sidebar--open");
    } else {
      aside.classList.remove("rasx-sidebar--open");
    }
    syncBackdrop();
  }
  openBtn.addEventListener("click", function () {
    aside.classList.add("rasx-sidebar--open");
    syncBackdrop();
  });
  if (closeBtn) {
    closeBtn.addEventListener("click", function () {
      aside.classList.remove("rasx-sidebar--open");
      syncBackdrop();
    });
  }
  backdrop.addEventListener("click", function () {
    aside.classList.remove("rasx-sidebar--open");
    syncBackdrop();
  });
  foldBtn.addEventListener("click", function () {
    if (!isNarrow()) aside.classList.add("rasx-meta--folded");
  });
  unfoldBtn.addEventListener("click", function () {
    aside.classList.remove("rasx-meta--folded");
  });
  mq.addEventListener("change", onViewportChange);
  onViewportChange();
})();
</script>
"""


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


def _profile_panel_title(label: int) -> str:
    return "Noise — XRD profiles" if int(label) == -1 else f"Cluster {int(label)} — XRD profiles"


def _y_profile_for_plot(row: np.ndarray) -> np.ndarray:
    """
    プロファイル線を途切れさせない。

    0・負・非有限は極小正に載せ替える。線形軸では通常スケールではベースラインと見分けがつかない
    程度で、対数軸でも有効な正値として描画できる。
    """
    y = np.asarray(row, dtype=np.float64).copy()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    floor = np.finfo(np.float64).tiny
    return np.maximum(y, floor)


def _validate_profile_inputs(
    n_rows: int,
    twotheta_grid: np.ndarray,
    intensity_matrix: np.ndarray,
) -> None:
    if intensity_matrix.ndim != 2:
        msg = "intensity_matrix must be 2-dimensional"
        raise ValueError(msg)
    if intensity_matrix.shape[0] != n_rows:
        msg = "intensity_matrix rows must match meta length"
        raise ValueError(msg)
    if intensity_matrix.shape[1] != twotheta_grid.shape[0]:
        msg = "intensity_matrix columns must match twotheta_grid length"
        raise ValueError(msg)


def _secondary_embedding_axis_titles(secondary_title: str) -> tuple[str, str]:
    """Axis titles for the non-PCA embedding panel; inferred from subplot title."""
    key = secondary_title.casefold()
    if "pca space used for dbscan" in key:
        return ("PC1", "PC2")
    if "umap" in key:
        return ("UMAP dimension 1", "UMAP dimension 2")
    if "t-sne" in key or "tsne" in key:
        return ("tSNE1", "tSNE2")
    return ("Embedding dimension 1", "Embedding dimension 2")


def build_embedding_figure(
    pca_xy: np.ndarray,
    secondary_xy: np.ndarray,
    labels: np.ndarray,
    meta: pl.DataFrame,
    *,
    secondary_title: str = "t-SNE (after PCA)",
    secondary_x_title: str | None = None,
    secondary_y_title: str | None = None,
) -> go.Figure:
    """Two-panel scatter: PCA and a secondary embedding (e.g. t-SNE), 1:1 aspect, cluster legend."""
    names = [Path(p).name for p in meta["path"].to_list()]
    x_mm = meta["x_mm"].to_numpy()
    y_mm = meta["y_mm"].to_numpy()
    samples = meta["sample"].to_list()

    lab_int = labels.astype(np.int64, copy=False)
    unique_labels = sorted({int(x) for x in np.unique(lab_int)})
    cmap = _color_map(lab_int)

    hover_lines = [
        f"{nm}<br>sample: {sam}<br>x: {xm:.4f} mm, y: {ym:.4f} mm<br>cluster: "
        f"{'noise' if int(lab) == -1 else str(int(lab))}"
        for nm, sam, xm, ym, lab in zip(names, samples, x_mm, y_mm, lab_int, strict=True)
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("PCA: PC1 vs PC2", secondary_title),
        horizontal_spacing=0.16,
    )

    for col, xy in ((1, pca_xy), (2, secondary_xy)):
        for lab in unique_labels:
            mask = lab_int == lab
            if not np.any(mask):
                continue
            name = _legend_name(lab)
            fig.add_trace(
                go.Scatter(
                    x=xy[mask, 0],
                    y=xy[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=7,
                        color=cmap[int(lab)],
                        line=dict(width=0.4, color="#333333"),
                    ),
                    text=[hover_lines[i] for i in np.flatnonzero(mask)],
                    hoverinfo="text",
                    name=name,
                    legendgroup=name,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )

    axis_kw = dict(
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        zeroline=False,
    )
    title_standoff = 12
    sx, sy = _secondary_embedding_axis_titles(secondary_title)
    if secondary_x_title is not None:
        sx = secondary_x_title
    if secondary_y_title is not None:
        sy = secondary_y_title
    fig.update_xaxes(
        **axis_kw,
        title_text="PC1",
        title_standoff=title_standoff,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        **axis_kw,
        title_text="PC2",
        title_standoff=title_standoff,
        constrain="domain",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        **axis_kw,
        title_text=sx,
        title_standoff=title_standoff,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        **axis_kw,
        title_text=sy,
        title_standoff=title_standoff,
        constrain="domain",
        row=1,
        col=2,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        title=dict(text="Embedding (PCA / t-SNE)", x=0.5, xanchor="center"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=72, r=200, t=80, b=72),
        width=_EMBED_WIDTH,
        height=_EMBED_HEIGHT,
        legend=dict(
            title=dict(text="Clusters (scatter)"),
            traceorder="normal",
            itemsizing="constant",
            groupclick="togglegroup",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            font=dict(size=11),
        ),
    )
    return fig


def build_xrd_profiles_figure(
    title: str,
    meta: pl.DataFrame,
    twotheta_grid: np.ndarray,
    intensity_matrix: np.ndarray,
    exclude_ranges: tuple[tuple[float, float], ...] = (),
    xrd_min_panel_height_px: int = 240,
) -> go.Figure:
    """
    単一パネルの XRD プロファイル図。凡例はファイル名。
    ``intensity_matrix`` は meta と行対応し、正規化後・標準化前のグリッド値。
    """
    names = [Path(p).name for p in meta["path"].to_list()]
    _validate_profile_inputs(len(names), twotheta_grid, intensity_matrix)

    fig = make_subplots(rows=1, cols=1)

    theta_min = float(np.min(twotheta_grid))
    theta_max = float(np.max(twotheta_grid))
    for idx, nm in enumerate(names):
        line_color = CLUSTER_PALETTE[int(idx) % len(CLUSTER_PALETTE)]
        y_plot = _y_profile_for_plot(intensity_matrix[int(idx)])
        fig.add_trace(
            go.Scatter(
                x=twotheta_grid,
                y=y_plot,
                mode="lines",
                line=dict(width=1.0, color=line_color),
                opacity=0.88,
                name=nm,
                legendgroup=f"prof:{nm}",
                showlegend=True,
                hovertemplate=(
                    f"{nm}<br>2theta: %{{x:.3f}} deg<br>Intensity: %{{y:.5g}} a.u.<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    fig.update_xaxes(
        title_text="2theta / degree",
        title_standoff=28,
        range=[theta_min, theta_max],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        zeroline=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Intensity / a.u.",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
        zeroline=False,
        row=1,
        col=1,
    )
    for start, end in exclude_ranges:
        fig.add_vrect(
            x0=float(start),
            x1=float(end),
            fillcolor="#D9D9D9",
            opacity=0.35,
            line_width=0,
            layer="below",
            row=1,
            col=1,
        )

    fig.update_layout(
        template="simple_white",
        font=dict(family="Times New Roman, Times, serif", size=14, color="black"),
        title=dict(
            text=title,
            xref="paper",
            x=0.02,
            xanchor="left",
            yanchor="top",
            # 上マージン内に収め、軸枠（プロット領域）と重ねない
            pad=dict(t=2, b=12),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(
            l=60,
            r=int(max(200, min(300, 52 + 8 * len(names)))),
            t=82,
            b=60,
        ),
        width=_XRD_WIDTH,
        height=max(int(xrd_min_panel_height_px * 1.5) + 120, 480),
        legend=dict(
            title=dict(text="XRD patterns"),
            traceorder="normal",
            itemsizing="constant",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            x=1.02,
            xanchor="left",
            y=1.0,
            yanchor="top",
            font=dict(size=10),
        ),
    )
    return fig


def _xrd_figure_specs(
    labels: np.ndarray,
    meta: pl.DataFrame,
    intensity_matrix: np.ndarray,
) -> list[tuple[str, pl.DataFrame, np.ndarray]]:
    """Return title, metadata, and intensity slices for all-profiles and each cluster."""
    lab_int = labels.astype(np.int64, copy=False)
    specs: list[tuple[str, pl.DataFrame, np.ndarray]] = [("All profiles", meta, intensity_matrix)]
    uniq = sorted({int(x) for x in np.unique(lab_int)})
    for lab in uniq:
        idx = np.flatnonzero(lab_int == lab)
        specs.append((_profile_panel_title(lab), meta[idx.tolist()], intensity_matrix[idx]))
    return specs


def _xrd_controls_html() -> str:
    """Form for relayouting all XRD figures together."""
    return """
    <section class="rasx-xrd-controls">
      <h2>XRD Patterns</h2>
      <form id="rasx-xrd-axis-form">
        <label>X min <input type="number" step="any" name="x_min" /></label>
        <label>X max <input type="number" step="any" name="x_max" /></label>
        <label>Y min <input type="number" step="any" name="y_min" /></label>
        <label>Y max <input type="number" step="any" name="y_max" /></label>
        <label>Intensity floor
          <input type="number" step="any" min="0" name="y_floor" placeholder="1e-6" />
        </label>
        <label>Intensity offset
          <input type="number" step="any" name="y_offset" placeholder="0.001" />
        </label>
        <label>Y scale
          <select name="y_type">
            <option value="linear">linear</option>
            <option value="log">log</option>
          </select>
        </label>
        <button type="submit">Apply</button>
        <button type="button" id="rasx-xrd-axis-reset">Reset</button>
      </form>
    </section>
    """


def _xrd_controls_script() -> str:
    """JavaScript wiring for the XRD axis control form."""
    return """
    <script>
    (function () {
      const form = document.getElementById("rasx-xrd-axis-form");
      const reset = document.getElementById("rasx-xrd-axis-reset");
      const targets = () => Array.from(document.querySelectorAll(".rasx-xrd-plot"));
      function ensureOriginalY(plot) {
        if (!plot.__rasxOriginalY) {
          plot.__rasxOriginalY = plot.data.map((trace) => Array.from(trace.y || []));
        }
        return plot.__rasxOriginalY;
      }
      function parsePositiveNumber(value) {
        if (value === "" || value == null) return null;
        const num = Number(value);
        return Number.isFinite(num) && num > 0 ? num : null;
      }
      function parseNumber(value) {
        if (value === "" || value == null) return 0;
        const num = Number(value);
        return Number.isFinite(num) ? num : 0;
      }
      function applyIntensityTransform(plot, floorValue, offsetValue) {
        const originalY = ensureOriginalY(plot);
        originalY.forEach((row, index) => {
          const nextRow = row.map((value) => {
            const shifted = value + offsetValue;
            return floorValue == null || shifted >= floorValue ? shifted : floorValue;
          });
          Plotly.restyle(plot, {y: [nextRow]}, [index]);
        });
      }
      if (!form) return;

      form.addEventListener("submit", function (event) {
        event.preventDefault();
        const data = new FormData(form);
        const xMin = data.get("x_min");
        const xMax = data.get("x_max");
        const yMin = data.get("y_min");
        const yMax = data.get("y_max");
        const yFloor = parsePositiveNumber(data.get("y_floor"));
        const yOffset = parseNumber(data.get("y_offset"));
        const yType = data.get("y_type") || "linear";
        const update = {
          "yaxis.type": yType,
          "yaxis.exponentformat": yType === "log" ? "power" : null,
          "yaxis.showexponent": yType === "log" ? "all" : null,
        };
        if (xMin !== "" && xMax !== "") update["xaxis.range"] = [Number(xMin), Number(xMax)];
        if (yMin !== "" && yMax !== "") update["yaxis.range"] = [Number(yMin), Number(yMax)];
        targets().forEach((plot) => {
          applyIntensityTransform(plot, yFloor, yOffset);
          Plotly.relayout(plot, update);
        });
      });

      reset.addEventListener("click", function () {
        form.reset();
        const update = {
          "xaxis.autorange": true,
          "yaxis.autorange": true,
          "yaxis.type": "linear",
          "yaxis.range": null,
          "xaxis.range": null,
          "yaxis.exponentformat": null,
          "yaxis.showexponent": null,
        };
        targets().forEach((plot) => {
          applyIntensityTransform(plot, null, 0);
          Plotly.relayout(plot, update);
        });
      });
    }());
    </script>
    """


def _cluster_summary_lines(labels: np.ndarray) -> list[str]:
    lab_int = labels.astype(np.int64, copy=False)
    uniq = sorted({int(x) for x in np.unique(lab_int)})
    lines: list[str] = []
    for lab in uniq:
        c = int(np.sum(lab_int == lab))
        if lab == -1:
            lines.append(f"noise: {c}")
        else:
            lines.append(f"cluster {lab}: {c}")
    return lines


def _cluster_file_listing_html(meta: pl.DataFrame, labels: np.ndarray) -> str:
    """HTML fragment listing file names for each cluster."""
    names = [Path(p).name for p in meta["path"].to_list()]
    lab_int = labels.astype(np.int64, copy=False)
    uniq = sorted({int(x) for x in np.unique(lab_int)})

    parts = ['<section class="rasx-cluster-files"><h2>Files by cluster</h2>']
    for lab in uniq:
        title = "noise" if lab == -1 else f"cluster {lab}"
        files = [escape(names[i]) for i in np.flatnonzero(lab_int == lab)]
        items = "".join(f"<li>{name}</li>" for name in files)
        parts.append(f"<h3>{escape(title)}</h3><ul>{items}</ul>")
    parts.append("</section>")
    return "".join(parts)


def build_metadata_sidebar_html(
    cfg: AppConfig,
    rasx_dir: str | Path,
    output_path: str | Path,
    meta: pl.DataFrame,
    labels: np.ndarray,
) -> str:
    """解析メタデータ用 HTML 断片（右サイドバー内に埋め込む）。"""
    rd = Path(rasx_dir).resolve()
    outp = Path(output_path)
    n_files = meta.height
    gen_at = datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    summary = _cluster_summary_lines(labels)
    n_cluster = len({int(x) for x in np.unique(labels.astype(np.int64)) if int(x) >= 0})
    n_noise = int(np.sum(labels.astype(np.int64) == -1))

    def dt(title: str, value: str) -> str:
        return f"<dt>{escape(title)}</dt><dd>{escape(value)}</dd>"

    rows = [
        dt("Generated at", gen_at),
        dt("rasX directory", str(rd)),
        dt("Output HTML", str(outp.resolve())),
        dt("Spectra (files)", str(n_files)),
        dt("Clusters (excluding noise)", str(n_cluster)),
        dt("Noise points", str(n_noise)),
        dt("Counts by cluster", ", ".join(summary)),
        dt(
            "2theta grid (deg)",
            f"{cfg.grid.theta_min} – {cfg.grid.theta_max}, n={cfg.grid.n_points}",
        ),
        dt("Intensity normalization", cfg.preprocess.intensity_normalization),
        dt("PCA components", str(cfg.pca.n_components)),
        dt("PCA random_state", str(cfg.pca.random_state)),
        dt("t-SNE perplexity", str(cfg.tsne.perplexity)),
        dt("t-SNE max_iter", str(cfg.tsne.max_iter)),
        dt("t-SNE random_state", str(cfg.tsne.random_state)),
        dt("DBSCAN eps", str(cfg.dbscan.eps)),
        dt("DBSCAN min_samples", str(cfg.dbscan.min_samples)),
        dt("DBSCAN space", cfg.dbscan.clustering_space),
    ]
    if cfg.paths.output_html:
        rows.insert(3, dt("Config output_html", cfg.paths.output_html))

    meta_html = (
        '<section class="rasx-meta-card"><h2>Analysis metadata</h2><dl>'
        + "".join(rows)
        + "</dl></section>"
    )
    files_html = _cluster_file_listing_html(meta, labels)
    return meta_html + files_html


def _full_report_html(
    embedding_fragment: str,
    wafer_panel_html: str,
    xrd_controls_html: str,
    xrd_fragments_html: str,
    sidebar_inner: str,
) -> str:
    """ページ全体: 埋め込み＋ウェハ、XRD、sticky サイドバー。"""
    style = (
        """
    html, body {
      margin: 0;
      min-height: 100%;
      font-family: "Times New Roman", Times, serif;
      color: #111;
    }
    .wrap {
      display: flex;
      flex-direction: row;
      flex-wrap: nowrap;
      justify-content: flex-start;
      align-items: flex-start;
      gap: 0;
      width: 100%;
      max-width: 100%;
      box-sizing: border-box;
      min-height: 100vh;
    }
    main.rasx-report-main {
      flex: 1 1 0%;
      min-width: 0;
      max-width: 100%;
      padding: 20px 16px 36px 20px;
      box-sizing: border-box;
    }
    aside.rasx-meta-sidebar {
      width: min(380px, 36vw);
      flex-shrink: 0;
      box-sizing: border-box;
      position: sticky;
      top: 0;
      align-self: flex-start;
      height: 100vh;
      max-height: 100vh;
      overflow-y: auto;
      overflow-x: hidden;
      border-left: 1px solid #ccc;
      padding: 12px 16px 24px 16px;
      font-size: 13px;
      line-height: 1.45;
      background: #f6f6f6;
      transition: width 0.2s ease, min-width 0.2s ease, padding 0.2s ease;
      box-shadow: inset 1px 0 0 rgba(255,255,255,0.75);
    }
    .rasx-sidebar-toolbar {
      display: flex;
      flex-direction: row;
      justify-content: flex-end;
      gap: 6px;
      margin-bottom: 8px;
    }
    .rasx-sidebar-toolbar button {
      font: inherit;
      font-size: 12px;
      padding: 4px 8px;
      cursor: pointer;
      border: 1px solid #999;
      border-radius: 4px;
      background: #fff;
    }
    .rasx-sidebar-toolbar button:hover {
      background: #eee;
    }
    .rasx-sidebar-body {
      min-height: 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    aside.rasx-meta-sidebar.rasx-meta--folded {
      width: 44px !important;
      min-width: 44px;
      padding-left: 6px;
      padding-right: 6px;
      overflow: visible;
    }
    aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-body {
      display: none;
    }
    aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-toolbar {
      flex-direction: column;
      align-items: center;
      margin-bottom: 0;
    }
    aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-fold {
      display: none;
    }
    aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-unfold {
      display: block !important;
    }
    .rasx-sidebar-unfold {
      display: none;
      writing-mode: vertical-rl;
      text-orientation: mixed;
      margin-top: 12px;
      padding: 8px 4px;
      width: 100%;
      box-sizing: border-box;
    }
    .rasx-sidebar-open-btn {
      position: fixed;
      right: 12px;
      bottom: 24px;
      z-index: 60;
      font: inherit;
      font-size: 13px;
      padding: 10px 14px;
      border-radius: 8px;
      border: 1px solid #555;
      background: #fafafa;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      cursor: pointer;
    }
    .rasx-sidebar-open-btn:hover {
      background: #eee;
    }
    .rasx-sidebar-backdrop {
      position: fixed;
      inset: 0;
      z-index: 55;
      background: rgba(0,0,0,0.25);
    }
    @media (max-width: 1100px) {
      aside.rasx-meta-sidebar {
        position: fixed;
        top: 0;
        right: 0;
        width: min(380px, 94vw);
        height: 100vh;
        max-height: 100vh;
        z-index: 56;
        transform: translateX(100%);
        transition: transform 0.22s ease;
        border-left: 1px solid #aaa;
        box-shadow: -4px 0 16px rgba(0,0,0,0.12);
      }
      aside.rasx-meta-sidebar.rasx-sidebar--open {
        transform: translateX(0);
      }
      aside.rasx-meta-sidebar.rasx-meta--folded {
        width: min(380px, 94vw) !important;
        min-width: unset;
      }
      aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-body {
        display: block;
      }
      aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-unfold {
        display: none !important;
      }
      aside.rasx-meta-sidebar.rasx-meta--folded .rasx-sidebar-fold {
        display: inline-block;
      }
      .rasx-sidebar-open-btn {
        display: block;
      }
      .rasx-sidebar-fold {
        display: none;
      }
    }
    @media (min-width: 1101px) {
      .rasx-sidebar-open-btn {
        display: none;
      }
      .rasx-sidebar-close {
        display: none;
      }
    }
    aside.rasx-meta-sidebar h2 {
      font-size: 15px;
      margin: 0 0 10px 0;
      font-weight: bold;
      letter-spacing: 0.01em;
    }
    aside.rasx-meta-sidebar h3 {
      font-size: 13px;
      margin: 12px 0 6px 0;
      font-weight: bold;
    }
    aside.rasx-meta-sidebar dt { font-weight: bold; margin-top: 10px; color: #333; }
    aside.rasx-meta-sidebar dd {
      margin: 4px 0 0 0;
      padding: 0;
      word-break: break-word;
    }
    aside.rasx-meta-sidebar ul {
      margin: 0 0 10px 18px;
      padding: 0;
    }
    aside.rasx-meta-sidebar li {
      margin: 0 0 3px 0;
      word-break: break-word;
    }
    .rasx-meta-card,
    .rasx-cluster-files {
      background: #fcfcfc;
      border: 1px solid #d9d9d9;
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
      transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
    }
    .rasx-meta-card:hover,
    .rasx-cluster-files:hover {
      box-shadow: 0 8px 18px rgba(0,0,0,0.10);
      border-color: #c5c5c5;
      transform: translateY(-1px);
    }
    .rasx-meta-card dl {
      margin: 0;
    }
    .rasx-meta-card dt {
      margin-top: 12px;
      padding-top: 10px;
      border-top: 1px solid #ececec;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .rasx-meta-card dt:first-of-type {
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
    }
    .rasx-meta-card dd {
      color: #222;
    }
    .rasx-cluster-files h3 {
      padding-top: 8px;
      border-top: 1px solid #ececec;
    }
    .rasx-cluster-files h3:first-of-type {
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
    }
    .rasx-cluster-files ul {
      margin-bottom: 0;
    }
    .rasx-embedding-row {
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      justify-content: flex-start;
      align-items: flex-start;
      gap: 18px 20px;
      margin-bottom: 8px;
      width: fit-content;
      max-width: 100%;
    }
    .rasx-embedding-plots {
      flex: 0 1 auto;
      min-width: 0;
      max-width: 100%;
    }
    .rasx-wafer-panel {
      flex: 0 1 auto;
      min-width: 180px;
      max-width: min(340px, 100%);
      margin-top: 56px;
      box-sizing: border-box;
      padding: 8px 10px 12px 10px;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fafafa;
      display: flex;
      flex-direction: column;
      align-items: stretch;
    }
    .rasx-wafer-title {
      font-size: 16px;
      margin: 0 0 8px 0;
      font-weight: bold;
      text-align: left;
    }
    .rasx-wafer-svg-wrap {
      display: flex;
      justify-content: center;
    }
    .rasx-wafer-svg-wrap svg {
      max-width: 100%;
      height: auto;
    }
    .rasx-wafer-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 6px 10px;
      justify-content: flex-start;
      margin-top: 10px;
      font-size: 11px;
    }
    .rasx-wafer-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .rasx-wafer-swatch {
      display: inline-block;
      width: 11px;
      height: 11px;
      border-radius: 50%;
      border: 1px solid #333;
      flex-shrink: 0;
    }
    .page-title {
      text-align: left;
      font-size: 24px;
      margin: 0 0 16px 0;
      font-weight: bold;
    }
    .plot-gap {
      height: 40px;
      margin: 8px 0 20px 0;
      border-bottom: 1px solid #e0e0e0;
    }
    .rasx-xrd-controls {
      margin: 0 0 18px 0;
    }
    .rasx-xrd-controls h2 {
      margin: 0 0 10px 0;
      font-size: 22px;
      font-weight: bold;
      text-align: left;
    }
    #rasx-xrd-axis-form {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: end;
      font-size: 13px;
    }
    #rasx-xrd-axis-form label {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    #rasx-xrd-axis-form input,
    #rasx-xrd-axis-form select,
    #rasx-xrd-axis-form button {
      font: inherit;
      padding: 4px 8px;
      box-sizing: border-box;
    }
    .rasx-xrd-figures {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: """
        + str(_XRD_BLOCK_GAP_PX)
        + """px;
      align-items: start;
    }
    .rasx-xrd-panel {
      min-width: 0;
    }
    .rasx-xrd-panel--all {
      grid-column: 1 / -1;
    }
    .rasx-xrd-panel .plotly-graph-div {
      width: 100% !important;
    }
    @media (max-width: 1100px) {
      .rasx-xrd-figures {
        grid-template-columns: 1fr;
      }
      .rasx-xrd-panel--all {
        grid-column: auto;
      }
    }
    """
    )
    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>XRD cluster map</title>
  <style>{style}</style>
</head>
<body>
  <div class="wrap">
    <main class="rasx-report-main">
      <h1 class="page-title">XRD pattern clusters (DBSCAN)</h1>
      <div class="rasx-embedding-row">
        <div class="rasx-embedding-plots">{embedding_fragment}</div>
        {wafer_panel_html}
      </div>
      <div class="plot-gap" aria-hidden="true"></div>
      {xrd_controls_html}
      <section class="rasx-xrd-figures">
        {xrd_fragments_html}
      </section>
    </main>
    <aside id="rasx-meta-sidebar" class="rasx-meta-sidebar" aria-label="Analysis metadata">
      <div class="rasx-sidebar-toolbar">
        <button type="button" class="rasx-sidebar-close" id="rasx-sidebar-close"
          aria-label="Close panel">×</button>
        <button type="button" class="rasx-sidebar-fold" id="rasx-sidebar-fold"
          aria-label="Fold panel">◀</button>
        <button type="button" class="rasx-sidebar-unfold" id="rasx-sidebar-unfold"
          aria-label="Expand panel">▶ Meta</button>
      </div>
      <div class="rasx-sidebar-body">
        {sidebar_inner}
      </div>
    </aside>
  </div>
  <div class="rasx-sidebar-backdrop" id="rasx-sidebar-backdrop" hidden></div>
  <button type="button" class="rasx-sidebar-open-btn"
    id="rasx-sidebar-open">Analysis metadata</button>
{_REPORT_SIDEBAR_SCRIPT}
</body>
</html>
"""


def write_cluster_map_html(
    path: str | Path,
    pca_xy: np.ndarray,
    secondary_xy: np.ndarray,
    labels: np.ndarray,
    meta: pl.DataFrame,
    twotheta_grid: np.ndarray,
    intensity_matrix: np.ndarray,
    *,
    config: AppConfig,
    rasx_dir: str | Path,
    secondary_title: str = "t-SNE (after PCA)",
) -> Path:
    """埋め込み図・XRD 図・右メタデータサイドバーをまとめた HTML を書き出す。"""
    out = Path(path)
    fig_emb = build_embedding_figure(
        pca_xy,
        secondary_xy,
        labels,
        meta,
        secondary_title=secondary_title,
    )
    emb_html = pio.to_html(
        fig_emb,
        include_plotlyjs="cdn",
        full_html=False,
        div_id="rasx-embedding-plot",
        config={"responsive": True},
    )
    xrd_fragments: list[str] = []
    for i, (title, meta_part, intensity_part) in enumerate(
        _xrd_figure_specs(labels, meta, intensity_matrix)
    ):
        fig_xrd = build_xrd_profiles_figure(
            title,
            meta_part,
            twotheta_grid,
            intensity_part,
            exclude_ranges=config.grid.exclude_ranges,
            xrd_min_panel_height_px=config.visualize.xrd_min_panel_height_px,
        )
        div_id = f"rasx-xrd-plot-{i}"
        fragment = pio.to_html(
            fig_xrd,
            include_plotlyjs=False,
            full_html=False,
            div_id=div_id,
            config={"responsive": True},
        )
        fragment = fragment.replace(
            'class="plotly-graph-div"',
            'class="plotly-graph-div rasx-xrd-plot"',
            1,
        )
        original_y_json = json.dumps(
            [_y_profile_for_plot(row).tolist() for row in intensity_part],
            separators=(",", ":"),
        )
        fragment += (
            f'<script>'
            f'document.getElementById("{div_id}").__rasxOriginalY = {original_y_json};'
            f"</script>"
        )
        panel_class = "rasx-xrd-panel rasx-xrd-panel--all" if i == 0 else "rasx-xrd-panel"
        xrd_fragments.append(f'<section class="{panel_class}">{fragment}</section>')

    sidebar_html = build_metadata_sidebar_html(config, rasx_dir, out, meta, labels)
    wafer_html = build_wafer_cluster_map_panel_html(labels, meta)
    doc = _full_report_html(
        emb_html,
        wafer_html,
        _xrd_controls_html() + _xrd_controls_script(),
        "".join(xrd_fragments),
        sidebar_html,
    )
    out.write_text(doc, encoding="utf-8")
    return out
