"""Shared bubble-chart visualisation helpers for dataset comparison notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Bubble area = UNIT_SCALES[unit] * log10(value + 1)
# Area scales with log so the full dynamic range of each unit fits in the bubble size range.
# MAX_S caps very large values; MIN_S sets the smallest visible dot.
UNIT_SCALES = {
    "h":     1000,
    "#img":  900,
    "#cond": 500,
    "#":     500,
}
MIN_S = 60
MAX_S = 6000


def value_to_size(value, unit):
    """Return (scatter_s, fontsize); area ∝ log10(value), clamped to [MIN_S, MAX_S]."""
    s = min(MAX_S, max(MIN_S, UNIT_SCALES[unit] * np.log10(value + 1)))
    if s < 300:
        fs = 5.5
    elif s < 800:
        fs = 7.0
    elif s < 2000:
        fs = 8.0
    else:
        fs = 9.0
    return s, fs


def fmt(value, unit):
    if unit in ("#img", "#cond", "#"):
        return f"{value / 1000:.1f}k" if value >= 1000 else str(int(value))
    if value >= 10:
        return f"{value:.0f}h"
    if value >= 1:
        return f"{value:.1f}h"
    return f"{value * 60:.0f}m"


def make_bubble_chart(column_groups, pivot, datasets_list, title, out_path,
                      sort_by=None):
    """Draw and save a bubble chart.

    Parameters
    ----------
    column_groups : list of (group_name, color, [(label, dotpath, unit), ...])
    pivot         : DataFrame indexed by dataset name, columns are dotpaths
    datasets_list : list of dataset names to include
    title         : figure title string
    out_path      : Path to save the PNG
    sort_by       : dotpath to sort rows by (descending); defaults to first fMRI
                    per-subject column found, or alphabetical if none.
    """
    all_cols = []
    group_spans = []
    for group_name, color, fields in column_groups:
        start = len(all_cols)
        for label, path, unit in fields:
            all_cols.append((label, path, unit, color))
        group_spans.append((group_name, color, start, len(all_cols) - 1))
    n_cols = len(all_cols)

    neuro_hour_paths = [
        p for gname, _, fields in column_groups
        for _, p, u in fields
        if gname == "Brain recordings" and u == "h"
    ]

    def _neuro_sum(ds):
        total = 0.0
        for p in (neuro_hour_paths if sort_by is None else [sort_by]):
            if ds in pivot.index and p in pivot.columns:
                v = pivot.loc[ds, p]
                if pd.notna(v):
                    total += float(v)
        return total

    if sort_by is None or sort_by in pivot.columns:
        datasets_sorted = sorted(datasets_list, key=_neuro_sum, reverse=True)
    else:
        datasets_sorted = sorted(datasets_list)
    n_ds = len(datasets_sorted)

    COL_W, ROW_H = 0.78, 0.80
    LABEL_W = 3.2
    HEADER_H = 1.6

    fig, ax = plt.subplots(figsize=(LABEL_W + n_cols * COL_W, HEADER_H + n_ds * ROW_H))

    for gname, color, c0, c1 in group_spans:
        ax.axvspan(c0 - 0.5, c1 + 0.5, color=color, alpha=0.07, zorder=0)

    for gname, color, c0, c1 in group_spans:
        ax.text((c0 + c1) / 2, n_ds + 0.55, gname,
                ha="center", va="center", fontsize=11, fontweight="bold", color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([c[0] for c in all_cols], rotation=45, ha="right", fontsize=9)
    for tick, (_, _, _, color) in zip(ax.get_xticklabels(), all_cols):
        tick.set_color(color)

    col_maxima = {}
    for col_j, (label, path, unit, color) in enumerate(all_cols):
        if path not in pivot.columns:
            continue
        col_vals = pivot.loc[pivot.index.isin(datasets_sorted), path].dropna()
        col_vals = col_vals[col_vals > 0]
        if not col_vals.empty:
            col_maxima[col_j] = col_vals.idxmax()

    for row_i, ds_name in enumerate(datasets_sorted):
        y = n_ds - 1 - row_i
        ax.text(-0.52, y, ds_name, ha="right", va="center", fontsize=9)
        for col_j, (label, path, unit, color) in enumerate(all_cols):
            if path not in pivot.columns or ds_name not in pivot.index:
                continue
            value = pivot.loc[ds_name, path]
            if pd.isna(value) or value == 0:
                continue
            s, fs = value_to_size(value, unit)
            is_col_max = col_maxima.get(col_j) == ds_name
            edge_color = "black" if is_col_max else "none"
            edge_width = 2.5 if is_col_max else 0
            ax.scatter(col_j, y, s=s, color=color, alpha=0.75,
                       edgecolors=edge_color, linewidths=edge_width, zorder=3)
            if s <= MIN_S:
                ax.text(col_j + 0.12, y, fmt(value, unit),
                        ha="left", va="center", fontsize=fs, fontweight="bold",
                        color="black", zorder=4)
            else:
                ax.text(col_j, y, fmt(value, unit),
                        ha="center", va="center", fontsize=fs, fontweight="bold",
                        color="white", zorder=4)

    ax.set_yticks([])
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.55, n_ds + 1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=1)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


def make_neuroimaging_depthvsbreadth(pivot_per_subject, pivot_total, datasets_list, out_path,
                      highlight="CNeuroMod", column_groups_per_subject=None,
                      column_groups_total=None, extra_points=None):
    """Scatter plot: neuroimaging hours per subject (x) vs number of subjects (y).

    Neuroimaging hours sum all modalities in the "Neuroimaging" column group.
    Iso-hours lines show constant total neuroimaging hours:
    n_subjects = H / hours_per_subject.

    Parameters
    ----------
    pivot_per_subject        : DataFrame indexed by dataset, columns are dotpaths
    pivot_total              : DataFrame indexed by dataset, columns are dotpaths
    datasets_list            : list of dataset names
    out_path                 : Path to save the PNG
    highlight                : dataset name to highlight (drawn larger, distinct color)
    column_groups_per_subject: column_groups list used to derive per-subject paths/label
    column_groups_total      : column_groups list used to derive total paths
    extra_points             : list of dicts with keys label, x, n_subjects, color
    """
    def _neuro_fields(column_groups):
        if column_groups is None:
            return []
        for gname, _color, fields in column_groups:
            if gname == "Brain recordings":
                return fields
        return []

    neuro_ps = _neuro_fields(column_groups_per_subject)
    neuro_tot = _neuro_fields(column_groups_total)

    PER_SUBJECT_PATHS = [path for _, path, _ in neuro_ps] or [
        "neuroimaging.fmri.per_subject_h",
        "neuroimaging.eeg.per_subject_h",
        "neuroimaging.meg.per_subject_h",
        "neuroimaging.ieeg.per_subject_h",
    ]
    TOTAL_PATHS = [path for _, path, _ in neuro_tot] or [
        "neuroimaging.fmri.total_h",
        "neuroimaging.eeg.total_h",
        "neuroimaging.meg.total_h",
        "neuroimaging.ieeg.total_h",
    ]
    modality_labels = " + ".join(label for label, _, _ in neuro_ps) if neuro_ps \
        else "fMRI + EEG + MEG + iEEG"
    xlabel = f"Brain recording hours per subject ({modality_labels})"

    def _sum_paths(pivot, ds, paths):
        total = 0.0
        for p in paths:
            if ds in pivot.index and p in pivot.columns:
                v = pivot.loc[ds, p]
                if pd.notna(v):
                    total += float(v)
        return total

    points = []
    for ds in datasets_list:
        x = _sum_paths(pivot_per_subject, ds, PER_SUBJECT_PATHS)
        y_total = _sum_paths(pivot_total, ds, TOTAL_PATHS)
        if x > 0 and y_total > 0:
            points.append((ds, x, float(y_total / x)))

    if not points:
        print("No neuroimaging data found — skipping scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    y_vals = [n_sub for _, _, n_sub in points]
    x_vals = [x for _, x, _ in points]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(x_vals) * 0.3, max(x_vals) * 3)
    ax.set_ylim(min(y_vals) * 0.3, max(y_vals) * 3)

    # Iso-hours hyperbolas: n_subjects = H / hours_per_subject
    x_lo, x_hi = ax.get_xlim()
    x_pad = np.geomspace(x_lo, x_hi, 300)

    # Gray gradient bands — darker where total neuroimaging hours are lower
    iso_levels = [50, 200, 1000, 5000, 10000]
    band_alphas = [0.18, 0.13, 0.08, 0.05, 0.02]
    ax.fill_between(x_pad, 1e-3, iso_levels[0] / x_pad,
                    color="black", alpha=band_alphas[0], zorder=0)
    for i in range(len(iso_levels) - 1):
        ax.fill_between(x_pad, iso_levels[i] / x_pad, iso_levels[i + 1] / x_pad,
                        color="black", alpha=band_alphas[i + 1], zorder=0)

    for H in [50, 200, 1000, 5000, 10000]:
        label = f"{H}h" if H < 1000 else f"{H // 1000}kh"
        y_iso = H / x_pad
        ax.plot(x_pad, y_iso, color="grey", linewidth=0.7,
                linestyle="--", alpha=0.4, zorder=1)
        ax.text(x_pad[-1], y_iso[-1], f"  {label}",
                va="center", fontsize=7, color="grey", alpha=0.7)

    for ds, x, n_sub in points:
        is_highlight = ds == highlight
        color = "#e63946" if is_highlight else "#4472C4"
        marker_size = 120 if is_highlight else 60
        zorder = 4 if is_highlight else 3
        ax.scatter(x, n_sub, s=marker_size, color=color, zorder=zorder,
                   edgecolors="white", linewidths=0.8)
        va = "bottom" if not is_highlight else "top"
        offset = (0, 6) if not is_highlight else (0, -6)
        ax.annotate(ds, (x, n_sub), xytext=offset, textcoords="offset points",
                    ha="center", va=va, fontsize=8,
                    fontweight="bold" if is_highlight else "normal",
                    color=color, zorder=5)

    for ep in (extra_points or []):
        ep_x, ep_y = ep["x"], ep["n_subjects"]
        ep_color = ep.get("color", "#4472C4")
        ax.scatter(ep_x, ep_y, s=60, color=ep_color, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(ep["label"], (ep_x, ep_y), xytext=(0, 6),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=8, color=ep_color, zorder=5)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Number of subjects", fontsize=10)
    ax.set_title("Brain recordings depth vs. breadth", fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


def make_neuroimaging_depthvsbreadth_plotly(
    pivot_per_subject, pivot_total, datasets_list, out_path,
    highlight="CNeuroMod", column_groups_per_subject=None,
    column_groups_total=None, extra_points=None,
):
    """Interactive Plotly version of make_neuroimaging_depthvsbreadth.

    Saves a self-contained HTML file to out_path.
    """
    def _neuro_fields(column_groups):
        if column_groups is None:
            return []
        for gname, _color, fields in column_groups:
            if gname == "Brain recordings":
                return fields
        return []

    neuro_ps = _neuro_fields(column_groups_per_subject)
    neuro_tot = _neuro_fields(column_groups_total)

    PER_SUBJECT_PATHS = [path for _, path, _ in neuro_ps] or [
        "neuroimaging.fmri.per_subject_h",
        "neuroimaging.eeg.per_subject_h",
        "neuroimaging.meg.per_subject_h",
        "neuroimaging.ieeg.per_subject_h",
    ]
    TOTAL_PATHS = [path for _, path, _ in neuro_tot] or [
        "neuroimaging.fmri.total_h",
        "neuroimaging.eeg.total_h",
        "neuroimaging.meg.total_h",
        "neuroimaging.ieeg.total_h",
    ]
    modality_labels = " + ".join(label for label, _, _ in neuro_ps) if neuro_ps \
        else "fMRI + EEG + MEG + iEEG"
    xlabel = f"Brain recording hours per subject ({modality_labels})"

    def _sum_paths(pivot, ds, paths):
        total = 0.0
        for p in paths:
            if ds in pivot.index and p in pivot.columns:
                v = pivot.loc[ds, p]
                if pd.notna(v):
                    total += float(v)
        return total

    points = []
    for ds in datasets_list:
        x = _sum_paths(pivot_per_subject, ds, PER_SUBJECT_PATHS)
        y_total = _sum_paths(pivot_total, ds, TOTAL_PATHS)
        if x > 0 and y_total > 0:
            points.append((ds, x, float(y_total / x)))

    if not points:
        print("No neuroimaging data found — skipping interactive scatter plot.")
        return

    x_vals = [x for _, x, _ in points]
    y_vals = [n for _, _, n in points]
    for ep in (extra_points or []):
        x_vals.append(ep["x"])
        y_vals.append(ep["n_subjects"])

    x_lo = min(x_vals) * 0.3
    x_hi = max(x_vals) * 3
    y_lo = min(y_vals) * 0.3
    y_hi = max(y_vals) * 3

    x_pad = np.geomspace(x_lo, x_hi, 300)

    iso_levels = [50, 200, 1000, 5000, 10000]
    band_alphas = [0.18, 0.13, 0.08, 0.05, 0.02]

    fig = go.Figure()

    def _rgba(alpha):
        return f"rgba(0,0,0,{alpha})"

    # Bottom band: below the first iso-level
    y_top = np.clip(iso_levels[0] / x_pad, y_lo, y_hi)
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_pad, x_pad[::-1]]),
        y=np.concatenate([y_top, np.full(len(x_pad), y_lo)]),
        fill="toself",
        fillcolor=_rgba(band_alphas[0]),
        line=dict(width=0),
        hoverinfo="skip",
        showlegend=False,
    ))

    # Bands between successive iso-levels
    for i in range(len(iso_levels) - 1):
        y_bot = np.clip(iso_levels[i] / x_pad, y_lo, y_hi)
        y_top = np.clip(iso_levels[i + 1] / x_pad, y_lo, y_hi)
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_pad, x_pad[::-1]]),
            y=np.concatenate([y_top, y_bot[::-1]]),
            fill="toself",
            fillcolor=_rgba(band_alphas[i + 1]),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Iso-hour lines + labels
    for H in iso_levels:
        label = f"{H}h" if H < 1000 else f"{H // 1000}kh"
        y_iso = H / x_pad
        mask = (y_iso >= y_lo) & (y_iso <= y_hi)
        if mask.any():
            fig.add_trace(go.Scatter(
                x=x_pad[mask], y=y_iso[mask],
                mode="lines",
                line=dict(color="grey", width=0.8, dash="dash"),
                opacity=0.4,
                hoverinfo="skip",
                showlegend=False,
            ))
            # Label at the right end of the visible curve
            xi = x_pad[mask][-1]
            yi = y_iso[mask][-1]
            fig.add_annotation(
                x=np.log10(xi), y=np.log10(yi),
                xref="x", yref="y",
                text=f"  {label}",
                showarrow=False,
                font=dict(size=9, color="grey"),
                xanchor="left",
                yanchor="middle",
            )

    # Dataset points
    for ds, x, n_sub in points:
        is_highlight = ds == highlight
        color = "#e63946" if is_highlight else "#4472C4"
        size = 14 if is_highlight else 10
        total_h = n_sub * x
        fig.add_trace(go.Scatter(
            x=[x], y=[n_sub],
            mode="markers+text",
            marker=dict(
                size=size,
                color=color,
                line=dict(color="white", width=1),
            ),
            text=[ds],
            textposition="top center" if not is_highlight else "bottom center",
            textfont=dict(
                size=10,
                color=color,
                family="Arial Black" if is_highlight else "Arial",
            ),
            hovertemplate=(
                f"<b>{ds}</b><br>"
                f"Per-subject hours: %{{x:.1f}} h<br>"
                f"Subjects: %{{y:.0f}}<br>"
                f"Total: {total_h:.0f} h<extra></extra>"
            ),
            showlegend=False,
        ))

    # Extra points
    for ep in (extra_points or []):
        ep_x, ep_y = ep["x"], ep["n_subjects"]
        ep_color = ep.get("color", "#4472C4")
        ep_total = ep_x * ep_y
        fig.add_trace(go.Scatter(
            x=[ep_x], y=[ep_y],
            mode="markers+text",
            marker=dict(size=10, color=ep_color, line=dict(color="white", width=1)),
            text=[ep["label"]],
            textposition="top center",
            textfont=dict(size=10, color=ep_color),
            hovertemplate=(
                f"<b>{ep['label']}</b><br>"
                f"Per-subject hours: %{{x:.1f}} h<br>"
                f"Subjects: %{{y:.0f}}<br>"
                f"Total: {ep_total:.0f} h<extra></extra>"
            ),
            showlegend=False,
        ))

    def _pow10_ticks(lo, hi):
        import math
        exps = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
        vals = [10**e for e in exps if lo * 0.5 <= 10**e <= hi * 2]
        texts = [f"{v:g}" for v in vals]
        return vals, texts

    x_tickvals, x_ticktext = _pow10_ticks(x_lo, x_hi)
    y_tickvals, y_ticktext = _pow10_ticks(y_lo, y_hi)

    fig.update_layout(
        title=dict(text="Brain recordings depth vs. breadth", font=dict(size=14, family="Arial")),
        xaxis=dict(
            title=xlabel,
            type="log",
            showgrid=False,
            ticks="outside",
            range=[np.log10(x_lo), np.log10(x_hi)],
            tickvals=x_tickvals,
            ticktext=x_ticktext,
        ),
        yaxis=dict(
            title="Number of subjects",
            type="log",
            showgrid=False,
            ticks="outside",
            range=[np.log10(y_lo), np.log10(y_hi)],
            tickvals=y_tickvals,
            ticktext=y_ticktext,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=650,
        height=550,
        margin=dict(l=60, r=60, t=60, b=80),
    )

    out_path = str(out_path)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"Saved {out_path.split('/')[-1]}")


def make_legend(out_path):
    """Save a standalone bubble-size legend figure."""
    LEGEND_SETS = [
        ("Hours (h)",      "h",      [1, 10, 50, 200, 1000], "#4472C4"),
        ("Images (#)",     "#img",   [1000, 10000, 50000, 100000], "#538135"),
        ("Conditions (#)", "#cond",  [10, 100, 500],               "#C55A11"),
    ]

    n_sets = len(LEGEND_SETS)
    max_n = max(len(vals) for _, _, vals, _ in LEGEND_SETS)
    COL_SP = 2.0
    ROW_SP = 2.2

    fig_w = 2.2 + max_n * COL_SP * 0.75
    fig_h = n_sets * ROW_SP * 0.65 + 0.7

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-1.1, (max_n - 0.5) * COL_SP)
    ax.set_ylim(-0.8, n_sets * ROW_SP - 0.2)
    ax.set_axis_off()
    ax.set_title("Bubble size legend", fontsize=12, fontweight="bold", pad=8)

    for set_i, (set_name, unit, ref_values, color) in enumerate(LEGEND_SETS):
        y = (n_sets - 1 - set_i) * ROW_SP + ROW_SP / 2
        ax.text(-0.7, y, set_name, ha="right", va="center",
                fontsize=10, fontweight="bold", color=color)
        for col_j, val in enumerate(ref_values):
            x = col_j * COL_SP
            s, fs = value_to_size(val, unit)
            ax.scatter(x, y, s=s, color=color, alpha=0.75, linewidths=0, zorder=3)
            ax.text(x, y, fmt(val, unit),
                    ha="center", va="center", fontsize=fs, fontweight="bold",
                    color="white", zorder=4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")
