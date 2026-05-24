"""Shared bubble-chart visualisation helpers for dataset comparison notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
                      highlight="CNeuroMod", highlights=None,
                      column_groups_per_subject=None,
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

    all_highlights = {highlight: "#e63946"}
    if highlights:
        all_highlights.update(highlights)

    for ds, x, n_sub in points:
        is_highlight = ds in all_highlights
        color = all_highlights.get(ds, "#4472C4")
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


DATASET_COLORS = {
    "CNeuroMod": "#e63946",
    "IBC":       "#f4a261",
    "NSD":       "#2a9d8f",
}

# (label, dotpath, divisor)
# Images are divided by 100 so their scale (~0–100) matches hours (~0–100 h).
RADAR_TASK_FIELDS = [
    ("Images\n(×100)",  "tasks.images.per_subject_unique",           100),
    ("Video\n(h)",      "tasks.video.per_subject_unique",              1),
    ("Audio\n(h)",      "tasks.audio.per_subject_unique",              1),
    ("Speech\n(h)",     "tasks.speech_listening.per_subject_unique",   1),
    ("Text\n(h)",       "tasks.text_reading.per_subject_unique",       1),
    ("Rest\n(h)",       "tasks.resting_state.per_subject_h",           1),
    ("Controlled\n(h)", "tasks.controlled.per_subject_h",              1),
    ("Games\n(h)",      "tasks.game.per_subject_h",                    1),
]


def make_task_composition_radar(pivot_per_subject, dataset, out_path,
                                 task_fields=None, color=None, figsize=(4.5, 4.5),
                                 r_max=None):
    """Draw and save a polar bar (Nightingale rose) chart of per-subject task composition.

    Each task type is a separate wedge bar; radius = absolute value (hours or images÷100).
    Pass the same r_max to all datasets in a set so figures are panel-comparable.
    """
    if task_fields is None:
        task_fields = RADAR_TASK_FIELDS
    if color is None:
        color = DATASET_COLORS.get(dataset, "#4472C4")

    labels = [label for label, _, _  in task_fields]
    paths  = [path  for _, path, _   in task_fields]
    divs   = [div   for _, _, div    in task_fields]
    N = len(labels)

    values = []
    for path, div in zip(paths, divs):
        if dataset in pivot_per_subject.index and path in pivot_per_subject.columns:
            v = pivot_per_subject.loc[dataset, path]
            values.append(float(v) / div if pd.notna(v) and v > 0 else 0.0)
        else:
            values.append(0.0)

    if r_max is None:
        r_max = max(values) if any(v > 0 for v in values) else 1.0

    # Log-scale transform: map values to polar radius so 0.1 → 0, r_max → r_plot_max.
    # Values ≤ 0 produce zero-height bars (invisible).
    LOG_FLOOR = 0.1
    log_min = np.log10(LOG_FLOOR)
    log_max = np.log10(max(r_max, LOG_FLOOR))
    r_plot_max = log_max - log_min

    def _to_r(v):
        return np.log10(max(v, LOG_FLOOR)) - log_min if v > 0 else 0.0

    bar_heights = [_to_r(v) for v in values]

    # Radial tick marks at decade boundaries within range
    tick_vals = [t for t in [0.1, 1, 10, 100] if t <= max(r_max, LOG_FLOOR) * 1.01]
    tick_pos  = [_to_r(t) for t in tick_vals]
    tick_labels = [f"{t:g}" for t in tick_vals]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    bar_width = 2 * np.pi / N * 0.8

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.bar(angles, bar_heights, width=bar_width, bottom=0,
           color=color, alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)

    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, r_plot_max)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=7, color="grey")

    ax.set_title(dataset, fontsize=13, fontweight="bold", color=color, pad=14)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


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
