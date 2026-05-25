"""Shared bubble-chart visualisation helpers for dataset comparison notebooks."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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

RANK_GRAY = "#aaaaaa"


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


def compute_dataset_ranks(pivot_per_subject, datasets_list, column_groups_per_subject=None):
    """Rank datasets by total brain-recording hours per subject.

    Returns dict {dataset: rank} where rank 10 = deepest, rank 1 = 10th deepest,
    rank 0 = not in top 10.
    """
    ps_paths = []
    if column_groups_per_subject is not None:
        for gname, _, fields in column_groups_per_subject:
            if gname in ("Brain recordings", "Brain"):
                ps_paths = [path for _, path, unit in fields if unit == "h"]
                break
    if not ps_paths:
        ps_paths = [
            "neuroimaging.fmri.per_subject_h",
            "neuroimaging.eeg.per_subject_h",
            "neuroimaging.meg.per_subject_h",
            "neuroimaging.ieeg.per_subject_h",
        ]

    def _total(ds):
        total = 0.0
        for p in ps_paths:
            if ds in pivot_per_subject.index and p in pivot_per_subject.columns:
                v = pivot_per_subject.loc[ds, p]
                if pd.notna(v):
                    total += float(v)
        return total

    scores = {ds: _total(ds) for ds in datasets_list}
    top10 = sorted(
        [ds for ds, s in scores.items() if s > 0],
        key=scores.__getitem__,
        reverse=True,
    )[:10]

    ranks = {ds: 0 for ds in datasets_list}
    for i, ds in enumerate(reversed(top10)):  # rank 1 = worst of top10, 10 = best
        ranks[ds] = i + 1

    return ranks


def turbo_color(rank, n_ranks=10):
    """Return a turbo colormap color for rank (0 → RANK_GRAY sentinel, 1 → darkest, n_ranks → brightest)."""
    if rank == 0:
        return RANK_GRAY
    return cm.get_cmap("plasma")(rank / (n_ranks + 1))


def make_bubble_chart(column_groups, pivot, datasets_list, title, out_path,
                      sort_by=None, row_colors=None, transpose=False):
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
    row_colors    : optional dict {dataset_name: color} — overrides column-group
                    color for bubble fills (backgrounds and axis labels unchanged).
    transpose     : if True, datasets on x-axis and modalities on y-axis (landscape).
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

    col_maxima = {}
    for col_j, (label, path, unit, color) in enumerate(all_cols):
        if path not in pivot.columns:
            continue
        col_vals = pivot.loc[pivot.index.isin(datasets_sorted), path].dropna()
        col_vals = col_vals[col_vals > 0]
        if not col_vals.empty:
            col_maxima[col_j] = col_vals.idxmax()

    if not transpose:
        COL_W, ROW_H = 0.90, 0.65
        LABEL_W = 3.2
        HEADER_H = 1.6

        fig, ax = plt.subplots(figsize=(LABEL_W + n_cols * COL_W, HEADER_H + n_ds * ROW_H))

        for gname, color, c0, c1 in group_spans:
            ax.axvspan(c0 - 0.5, c1 + 0.5, color=color, alpha=0.07, zorder=0)

        for gname, color, c0, c1 in group_spans:
            ax.text((c0 + c1) / 2, n_ds + 0.55, gname,
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([c[0] for c in all_cols], rotation=45, ha="right", fontsize=11)
        for tick, (_, _, _, color) in zip(ax.get_xticklabels(), all_cols):
            tick.set_color(color)

        for row_i, ds_name in enumerate(datasets_sorted):
            y = n_ds - 1 - row_i
            ax.text(-0.52, y, ds_name, ha="right", va="center", fontsize=11)
            for col_j, (label, path, unit, color) in enumerate(all_cols):
                if path not in pivot.columns or ds_name not in pivot.index:
                    continue
                value = pivot.loc[ds_name, path]
                if pd.isna(value) or value == 0:
                    continue
                s, fs = value_to_size(value, unit)
                bubble_color = row_colors.get(ds_name, color) if row_colors else color
                is_col_max = col_maxima.get(col_j) == ds_name
                edge_color = "black" if is_col_max else "none"
                edge_width = 2.5 if is_col_max else 0
                ax.scatter(col_j, y, s=s, color=bubble_color, alpha=0.75,
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

    else:
        # Landscape: datasets on x-axis, modalities on y-axis.
        COL_W, ROW_H = 0.90, 0.65
        MOD_LABEL_W = 1.5
        HEADER_H = 1.6

        fig, ax = plt.subplots(figsize=(MOD_LABEL_W + n_ds * COL_W, HEADER_H + n_cols * ROW_H))

        for gname, color, c0, c1 in group_spans:
            ax.axhspan(c0 - 0.5, c1 + 0.5, color=color, alpha=0.07, zorder=0)

        for gname, color, c0, c1 in group_spans:
            ax.text(n_ds + 0.55, (c0 + c1) / 2, gname,
                    ha="center", va="center", fontsize=11, fontweight="bold", color=color,
                    rotation=90)

        ax.set_xticks(range(n_ds))
        ax.set_xticklabels(datasets_sorted, rotation=45, ha="right", fontsize=11)
        if row_colors:
            for tick, ds in zip(ax.get_xticklabels(), datasets_sorted):
                tick.set_color(row_colors.get(ds, "black"))

        ax.set_yticks(range(n_cols))
        ax.set_yticklabels([c[0] for c in all_cols], fontsize=11)
        for tick, (_, _, _, color) in zip(ax.get_yticklabels(), all_cols):
            tick.set_color(color)

        for row_i, ds_name in enumerate(datasets_sorted):
            for col_j, (label, path, unit, color) in enumerate(all_cols):
                if path not in pivot.columns or ds_name not in pivot.index:
                    continue
                value = pivot.loc[ds_name, path]
                if pd.isna(value) or value == 0:
                    continue
                s, fs = value_to_size(value, unit)
                bubble_color = row_colors.get(ds_name, color) if row_colors else color
                is_col_max = col_maxima.get(col_j) == ds_name
                edge_color = "black" if is_col_max else "none"
                edge_width = 2.5 if is_col_max else 0
                ax.scatter(row_i, col_j, s=s, color=bubble_color, alpha=0.75,
                           edgecolors=edge_color, linewidths=edge_width, zorder=3)
                if s <= MIN_S:
                    ax.text(row_i + 0.12, col_j, fmt(value, unit),
                            ha="left", va="center", fontsize=fs, fontweight="bold",
                            color="black", zorder=4)
                else:
                    ax.text(row_i, col_j, fmt(value, unit),
                            ha="center", va="center", fontsize=fs, fontweight="bold",
                            color="white", zorder=4)

        ax.set_xlim(-0.5, n_ds + 1.0)
        ax.set_ylim(-0.55, n_cols - 0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=1)

    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


def make_neuroimaging_depthvsbreadth(pivot_per_subject, pivot_total, datasets_list, out_path,
                      highlight="CNeuroMod", highlights=None,
                      column_groups_per_subject=None,
                      column_groups_total=None, extra_points=None,
                      dataset_colors=None):
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
    highlight                : dataset name to highlight (used only when dataset_colors is None)
    highlights               : dict {ds: color} for additional highlights (dataset_colors=None only)
    column_groups_per_subject: column_groups list used to derive per-subject paths/label
    column_groups_total      : column_groups list used to derive total paths
    extra_points             : list of dicts with keys label, x, n_subjects, color
    dataset_colors           : optional dict {ds: color} — when provided, overrides highlight/
                               highlights; top-ranked datasets (non-gray color) get larger markers.
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
    iso_levels = [100, 1000, 10000]
    band_alphas = [0.18, 0.10, 0.04]
    ax.fill_between(x_pad, 1e-3, iso_levels[0] / x_pad,
                    color="black", alpha=band_alphas[0], zorder=0)
    for i in range(len(iso_levels) - 1):
        ax.fill_between(x_pad, iso_levels[i] / x_pad, iso_levels[i + 1] / x_pad,
                        color="black", alpha=band_alphas[i + 1], zorder=0)

    for H in [100, 1000, 10000]:
        label = f"{H}h" if H < 1000 else f"{H // 1000}kh"
        y_iso = H / x_pad
        ax.plot(x_pad, y_iso, color="grey", linewidth=0.7,
                linestyle="--", alpha=0.4, zorder=1)
        ax.text(x_pad[-1], y_iso[-1], f"  {label}",
                va="center", fontsize=7, color="grey", alpha=0.7)

    _plasma_zero = cm.get_cmap("plasma")(0.0)
    # Per-dataset label offsets (xytext_dx, xytext_dy, ha, va) to avoid overlaps.
    _LABEL_OFFSETS = {
        "MSC":          (  0,   6, "center", "bottom"),
        "NL-fMRI":      (  0,  -6, "center", "top"),
        "BOLD5000":     (  0,  -6, "center", "top"),
        "IBC":          (  0,   6, "center", "bottom"),
        "NSD":          (  0,  -6, "center", "top"),
        "MyConnectome": (-25,  -6, "center", "top"),
        "Dr Who":       ( 15,  -6, "center", "top"),
    }

    if dataset_colors is not None:
        for ds, x, n_sub in points:
            color = dataset_colors.get(ds, RANK_GRAY)
            is_top = (color != RANK_GRAY)
            if not is_top:
                color = _plasma_zero
            marker_size = 90 if is_top else 50
            zorder = 4 if is_top else 3
            ax.scatter(x, n_sub, s=marker_size, color=color, zorder=zorder,
                       edgecolors="white", linewidths=0.8)
            if ds in _LABEL_OFFSETS:
                dx, dy, ha, va = _LABEL_OFFSETS[ds]
            else:
                va = "bottom" if not is_top else "top"
                dx, dy, ha = 0, (6 if not is_top else -6), "center"
            ax.annotate(ds, (x, n_sub), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=8,
                        fontweight="bold" if is_top else "normal",
                        color=color, zorder=5)
    else:
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
            if ds in _LABEL_OFFSETS:
                dx, dy, ha, va = _LABEL_OFFSETS[ds]
            else:
                va = "bottom" if not is_highlight else "top"
                dx, dy, ha = 0, (6 if not is_highlight else -6), "center"
            ax.annotate(ds, (x, n_sub), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va=va, fontsize=8,
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
    ax.set_title("")
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

# (full_label, abbrev, dotpath, divisor)
# Images are divided by 100 so their scale (~0–100) matches hours (~0–100 h).
# full_label is shown on the legend (empty) radar; abbrev is shown on each dataset radar.
RADAR_TASK_FIELDS = [
    ("Images\n(×100)", "I",  "tasks.images.per_subject_unique",            100),
    ("Video\n(h)",     "V",  "tasks.video.per_subject_unique",               1),
    ("Audio\n(h)",     "A",  "tasks.audio.per_subject_unique",               1),
    ("Speech\n(h)",    "Sp", "tasks.speech_listening.per_subject_unique",    1),
    ("Text\n(h)",      "T",  "tasks.text_reading.per_subject_unique",        1),
    ("Rest\n(h)",      "R",  "tasks.resting_state.per_subject_h",            1),
    ("Controlled\n(h)","C",  "tasks.controlled.per_subject_h",               1),
    ("Games\n(h)",     "G",  "tasks.game.per_subject_h",                     1),
    ("Contrasts\n(#)", "#",  "tasks.contrasts.per_subject",                  1),
]


def _draw_radar_on_ax(ax, pivot_per_subject, dataset, task_fields, color, r_max=None):
    """Draw a radar (Nightingale rose) chart onto an existing polar axes."""
    labels = [abbrev for _, abbrev, _, _ in task_fields]
    paths  = [path   for _, _, path, _ in task_fields]
    divs   = [div    for _, _, _, div in task_fields]
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

    LOG_FLOOR = 0.1
    log_min = np.log10(LOG_FLOOR)
    log_max = np.log10(max(r_max, LOG_FLOOR))
    r_plot_max = log_max - log_min

    def _to_r(v):
        return np.log10(max(v, LOG_FLOOR)) - log_min if v > 0 else 0.0

    bar_heights = [_to_r(v) for v in values]
    tick_vals = [t for t in [0.1, 1, 10, 100] if t <= max(r_max, LOG_FLOOR) * 1.01]
    tick_pos  = [_to_r(t) for t in tick_vals]
    tick_labels = [f"{t:g}" for t in tick_vals]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    bar_width = 2 * np.pi / N * 0.8

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(angles, bar_heights, width=bar_width, bottom=0,
           color=color, alpha=0.75, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=7)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, r_plot_max)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=6, color="grey")
    ax.set_title(dataset, fontsize=10, fontweight="bold", color=color, pad=10)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", linestyle=":", linewidth=0.5, alpha=0.5)


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

    paths = [path for _, _, path, _ in task_fields]
    divs  = [div  for _, _, _, div in task_fields]

    values = []
    for path, div in zip(paths, divs):
        if dataset in pivot_per_subject.index and path in pivot_per_subject.columns:
            v = pivot_per_subject.loc[dataset, path]
            values.append(float(v) / div if pd.notna(v) and v > 0 else 0.0)
        else:
            values.append(0.0)

    if r_max is None:
        r_max = max(values) if any(v > 0 for v in values) else 1.0

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})
    _draw_radar_on_ax(ax, pivot_per_subject, dataset, task_fields, color, r_max=r_max)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


def _draw_empty_radar_on_ax(ax, task_fields, r_max):
    """Draw an empty radar chart showing only the grid, tick values, and category labels."""
    labels = [label for label, _, _, _ in task_fields]
    N = len(labels)

    LOG_FLOOR = 0.1
    log_min = np.log10(LOG_FLOOR)
    log_max = np.log10(max(r_max, LOG_FLOOR))
    r_plot_max = log_max - log_min

    def _to_r(v):
        return np.log10(max(v, LOG_FLOOR)) - log_min if v > 0 else 0.0

    tick_vals = [t for t in [0.1, 1, 10, 100] if t <= max(r_max, LOG_FLOOR) * 1.01]
    tick_pos  = [_to_r(t) for t in tick_vals]
    tick_labels = [f"{t:g}" for t in tick_vals]

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_rlabel_position(0)
    ax.set_ylim(0, r_plot_max)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_labels, fontsize=13, color="black", fontweight="bold")
    ax.set_title("Scale", fontsize=11, fontweight="bold", color="black", pad=10)
    ax.spines["polar"].set_visible(False)
    ax.grid(color="grey", linestyle=":", linewidth=0.5, alpha=0.5)


def make_radar_grid(pivot_per_subject, datasets_ranked, out_path,
                    task_fields=None, r_max=None):
    """Draw a 2×5 grid of radar charts for the top-10 datasets, with an empty scale radar at left.

    Parameters
    ----------
    pivot_per_subject : DataFrame indexed by dataset name, columns are dotpaths
    datasets_ranked   : list of (dataset_name, color) ordered rank-10 first (best → worst)
    out_path          : Path to save the PNG
    task_fields       : list of (label, dotpath, divisor); defaults to RADAR_TASK_FIELDS
    r_max             : shared radial max across all panels; computed if None
    """
    if task_fields is None:
        task_fields = RADAR_TASK_FIELDS

    if r_max is None:
        all_vals = []
        for ds, _ in datasets_ranked:
            for _, _, path, div in task_fields:
                if ds in pivot_per_subject.index and path in pivot_per_subject.columns:
                    v = pivot_per_subject.loc[ds, path]
                    if pd.notna(v) and v > 0:
                        all_vals.append(float(v) / div)
        r_max = max(all_vals) if all_vals else 100.0

    n_data_cols = 5
    n_rows = 2
    legend_width_ratio = 1.4

    fig = plt.figure(figsize=(n_data_cols * 3.2 + legend_width_ratio * 3.2, n_rows * 3.4))
    gs = fig.add_gridspec(
        n_rows, n_data_cols + 1,
        width_ratios=[legend_width_ratio] + [1] * n_data_cols,
        hspace=0.55, wspace=0.45,
    )

    # Empty scale radar spanning both rows (replaces the text legend)
    ax_scale = fig.add_subplot(gs[:, 0], polar=True)
    _draw_empty_radar_on_ax(ax_scale, task_fields, r_max)

    # Radar panels: rank-10 (best) at top-left, rank-1 at bottom-right
    for i, (ds, color) in enumerate(datasets_ranked):
        row = i // n_data_cols
        col = i % n_data_cols + 1
        ax = fig.add_subplot(gs[row, col], polar=True)
        _draw_radar_on_ax(ax, pivot_per_subject, ds, task_fields, color, r_max=r_max)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved {out_path.name}")


def make_legend_per_subject(out_path):
    """Single-line compact bubble-size legend for per-subject figures. No title."""
    LEGEND_SETS = [
        ("h",     "h",     [1, 10, 50, 200],         "#4472C4"),
        ("#img",  "#img",  [1000, 10000],             "#538135"),
        ("#cond", "#cond", [10, 100, 500],            "#C55A11"),
    ]

    COL_SP = 2.0   # horizontal spacing between bubbles within a group
    LABEL_W = 2.2  # space reserved for the unit label before each group
    COL_W = 9.0    # equal column width allocated to each group

    # Center each group's content within its equal-width column
    xs_per_group = []
    for i, (_, unit, vals, _) in enumerate(LEGEND_SETS):
        col_center = (i + 0.5) * COL_W
        content_w = LABEL_W + (len(vals) - 1) * COL_SP
        label_x = col_center - content_w / 2
        bubble_xs = [label_x + LABEL_W + j * COL_SP for j in range(len(vals))]
        xs_per_group.append((label_x, bubble_xs))

    total_w = len(LEGEND_SETS) * COL_W
    fig_w = max(total_w * 0.5, 6)
    fig_h = 1.4

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, total_w + 0.5)
    ax.set_ylim(-0.8, 0.8)
    ax.set_axis_off()

    y = 0.0
    for (label_x, bubble_xs), (set_name, unit, ref_values, color) in zip(xs_per_group, LEGEND_SETS):
        ax.text(label_x, y, set_name, ha="left", va="center",
                fontsize=9, fontweight="bold", color=color)
        for bx, val in zip(bubble_xs, ref_values):
            s, fs = value_to_size(val, unit)
            ax.scatter(bx, y, s=s, color=color, alpha=0.75, linewidths=0, zorder=3)
            ax.text(bx, y, fmt(val, unit),
                    ha="center", va="center", fontsize=fs, fontweight="bold",
                    color="white", zorder=4)

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
