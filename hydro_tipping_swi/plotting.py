from __future__ import annotations

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import seaborn as sns




def load_plot_bundle_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    mr = {
        "top_keep_idx": d["top_keep_idx"],
        "bot_keep_idx": d["bot_keep_idx"],
        "top_tris_all": d["top_tris_all"],
        "bot_tris_all": d["bot_tris_all"],
        "boundary": d["boundary"],
        "sal_by_time": {
            str(d["date_early"]): d["sal_early"],
            str(d["date_late"]): d["sal_late"],
        },
    }
    date_early = str(d["date_early"])
    date_late  = str(d["date_late"])
    return mr, date_early, date_late


def plot_model_sal_and_change_tidy(
    model_result,
    date_early, date_late,
    sal_cmap="viridis", delta_cmap="coolwarm",
    sal_vmin=None, sal_vmax=None,
    delta_vlim=None, change_mode="absolute", robust_pct=98,
    panel_labels=("a", "b", "c", "d")
):
    s_e = model_result["sal_by_time"].get(date_early)
    s_l = model_result["sal_by_time"].get(date_late)
    if s_e is None or s_l is None:
        raise ValueError("Missing salinity for one/both dates.")

    top_idx = model_result["top_keep_idx"];  bot_idx = model_result["bot_keep_idx"]
    top_tris = model_result["top_tris_all"][top_idx]
    bot_tris = model_result["bot_tris_all"][bot_idx]

    top_e, bot_e = s_e[top_idx], s_e[bot_idx]
    if sal_vmin is None or sal_vmax is None:
        both_early = np.concatenate([top_e, bot_e])
        sal_vmin = np.nanmin(both_early) if sal_vmin is None else sal_vmin
        sal_vmax = np.nanmax(both_early) if sal_vmax is None else sal_vmax

    eps = 1e-12
    if change_mode == "percent":
        top_delta = 100*(s_l[top_idx]-top_e)/np.maximum(np.abs(top_e), eps)
        bot_delta = 100*(s_l[bot_idx]-bot_e)/np.maximum(np.abs(bot_e), eps)
        delta_units = "% change"
    else:
        top_delta = s_l[top_idx]-top_e
        bot_delta = s_l[bot_idx]-bot_e
        delta_units = "Change in salinity [PSU]"

    if delta_vlim is None:
        m = np.nanpercentile(np.abs(np.concatenate([top_delta, bot_delta])), robust_pct)
        dvmin, dvmax = -m, m
    elif isinstance(delta_vlim, (int, float)):
        dvmin, dvmax = -abs(delta_vlim), abs(delta_vlim)
    else:
        dvmin, dvmax = delta_vlim
    delta_norm = TwoSlopeNorm(vmin=dvmin, vcenter=0.0, vmax=dvmax)

    fig, axes = plt.subplots(2, 2, figsize=(7, 7),
                             constrained_layout=True, sharex=True, sharey=True)
    (ax_top_e, ax_bot_e), (ax_top_d, ax_bot_d) = axes

    pc_te = PolyCollection(top_tris, array=top_e, cmap=sal_cmap, edgecolor='none', rasterized=True)
    pc_te.set_clim(sal_vmin, sal_vmax); ax_top_e.add_collection(pc_te)

    pc_be = PolyCollection(bot_tris, array=bot_e, cmap=sal_cmap, edgecolor='none', rasterized=True)
    pc_be.set_clim(sal_vmin, sal_vmax); ax_bot_e.add_collection(pc_be)

    pc_td = PolyCollection(top_tris, array=top_delta, cmap=delta_cmap,
                           edgecolor='none', rasterized=True, norm=delta_norm)
    ax_top_d.add_collection(pc_td)

    pc_bd = PolyCollection(bot_tris, array=bot_delta, cmap=delta_cmap,
                           edgecolor='none', rasterized=True, norm=delta_norm)
    ax_bot_d.add_collection(pc_bd)

    b = model_result["boundary"]
    for ax in axes.ravel():
        ax.plot(np.r_[b[:,0], b[0,0]], np.r_[b[:,1], b[0,1]], 'k-', lw=0.8, alpha=0.9)
        ax.set_aspect("equal"); ax.autoscale()
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.tick_params(direction='out', length=3, width=0.8)

    ax_top_e.set_title("Top")
    ax_bot_e.set_title("Bottom")
    ax_top_d.set_title("")   # no titles on row 2
    ax_bot_d.set_title("")

    for ax in axes[0, :]: ax.tick_params(labelbottom=False)
    for ax in axes[:, 1]: ax.tick_params(labelleft=False)
    for ax in axes[1, :]: ax.set_xlabel("Distance X [m]")
    for ax in axes[:, 0]: ax.set_ylabel("Distance Y [m]")


    label_y = 1.04  # same vertical level as titles
    label_x = -0.02 # just outside the left edge of each axes
    for ax, lab in zip(axes.ravel(), panel_labels):
        ax.text(label_x, label_y, lab, transform=ax.transAxes,
                ha="right", va="baseline", fontweight="bold",
                fontsize=11, clip_on=False)

    # colorbars
    sm_sal = plt.cm.ScalarMappable(cmap=sal_cmap); sm_sal.set_clim(sal_vmin, sal_vmax)
    cb1 = fig.colorbar(sm_sal, ax=axes[0, :], location="right", fraction=0.046, pad=0.04)
    cb1.set_label("Salinity [PSU]")

    sm_delta = plt.cm.ScalarMappable(cmap=delta_cmap, norm=delta_norm)
    cb2 = fig.colorbar(sm_delta, ax=axes[1, :], location="right", fraction=0.046, pad=0.04)
    cb2.set_label(delta_units)

    used_limits = {"salinity": (sal_vmin, sal_vmax), "delta": (dvmin, dvmax), "delta_units": delta_units}
    return fig, axes, used_limits






def _apply_publication_style():
    """Apply a clean Nature/Science-like matplotlib style."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "axes.linewidth": 1.0,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
    })


def _beautify_ax(ax):
    """Remove top/right spines and add light major-grid styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", top=False, right=False)
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.grid(False, which="minor")


def _set_sci_y(ax, offset_text_x=-0.06, offset_text_size=11):
    """Format y-axis as ×10^n instead of 1e10."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))  # always scientific notation
    ax.yaxis.set_major_formatter(fmt)
    off = ax.yaxis.get_offset_text()
    off.set_size(offset_text_size)
    off.set_x(offset_text_x)


def _scale_sizes_linear(values, min_size=25, max_size=220):
    """Linearly scale a numeric array to marker sizes."""
    values = np.asarray(values, dtype=float)
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 0:
        return min_size + (values - vmin) / (vmax - vmin) * (max_size - min_size)

    return np.full_like(values, (min_size + max_size) / 2.0, dtype=float)


def plot_extreme_salinization_two_panel(
    df,
    dir_out=None,
    *,
    out_name="Fig_ESM_ab.png",
    save=True,
    show=True,
    dpi=450,
    figsize=(7.0, 6.6),
    k_tipping=1.58e-12,
    panel_a_offset="0m0",
    xcol="log permeability [m^2]",
    ycol="Max Global Sal [mol]",
    ccol="GWR_precp_rate [-]",
    scol="Permeability [m^2]",
    offset_col="SLR offset",
    min_size=25,
    max_size=220,
    cmap_panel_a="Spectral_r",
    cbar_label="Groundwater recharge ratio [-]",
    y_label="Extreme salinization mass [mol]",
    x_label=r"log permeability [m$^2$]",
    slr_map=None,
    slr_order=None,
    slr_markers=None,
    slr_colors=None,
):
    """
    Make a 2-panel publication-style figure.

    Panel a:
        One SLR scenario, colored by recharge ratio and sized by permeability.
    Panel b:
        All SLR scenarios with marker/color categories.
    Top row:
        Horizontal colorbar in its own row.

    Returns
    -------
    fig, (ax1, ax2), out_png
        Figure, axes tuple, and saved path (or None if not saved).
    """
    _apply_publication_style()

    if slr_map is None:
        slr_map = {
            "0m0": "Historical",
            "0m5": "+ 0.5 m",
            "1m0": "+ 1.0 m",
            "1m5": "+ 1.5 m",
        }

    if slr_order is None:
        slr_order = ["0m0", "0m5", "1m0", "1m5"]

    if slr_markers is None:
        slr_markers = {"0m0": "o", "0m5": "^", "1m0": "s", "1m5": "D"}

    if slr_colors is None:
        cmap = plt.get_cmap("viridis_r")
        slr_colors = {
            off: cmap(v)
            for off, v in zip(slr_order, np.linspace(0.15, 0.90, len(slr_order)))
        }

    logk_tipping = np.log10(k_tipping)

    df_a = df.loc[df[offset_col].astype(str) == str(panel_a_offset)].copy()
    df_a = df_a.dropna(subset=[xcol, ycol, ccol, scol])

    if df_a.empty:
        raise ValueError(
            f"No valid rows found for panel_a_offset='{panel_a_offset}' "
            f"after dropping NaNs from [{xcol}, {ycol}, {ccol}, {scol}]."
        )

    s_scaled = _scale_sizes_linear(
        df_a[scol].to_numpy(dtype=float),
        min_size=min_size,
        max_size=max_size,
    )
    cvals = df_a[ccol].to_numpy(dtype=float)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.07, 1.0, 1.0],
        hspace=0.60,
    )

    cax_h = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax1)

    sc = ax1.scatter(
        df_a[xcol].to_numpy(dtype=float),
        df_a[ycol].to_numpy(dtype=float),
        c=cvals,
        s=s_scaled,
        cmap=cmap_panel_a,
        alpha=0.92,
        linewidth=0.6,
        edgecolor="white",
        zorder=3,
    )

    ax1.axvline(
        logk_tipping,
        color="#5595b5",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )

    ax1.annotate(
        r"$K_{\mathrm{tipping}}$",
        xy=(logk_tipping, 0.6),
        xycoords=("data", "axes fraction"),
        xytext=(-6, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=11,
        color="#5595b5",
        clip_on=False,
    )

    _set_sci_y(ax1)
    _beautify_ax(ax1)
    ax1.tick_params(labelbottom=False)
    ax1.text(
        -0.09, 1.3, "a",
        transform=ax1.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    cbar = fig.colorbar(sc, cax=cax_h, orientation="horizontal")
    cbar.set_label(cbar_label, labelpad=6)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.outline.set_linewidth(0.9)
    cbar.ax.tick_params(labelsize=11)

    cax_h.set_yticks([])
    for sp in cax_h.spines.values():
        sp.set_visible(False)


    for off in slr_order:
        dfi = df.loc[df[offset_col].astype(str) == str(off), [xcol, ycol]].dropna()
        if dfi.empty:
            continue

        ax2.scatter(
            dfi[xcol].to_numpy(dtype=float),
            dfi[ycol].to_numpy(dtype=float),
            s=20,
            alpha=0.70,
            marker=slr_markers.get(off, "o"),
            color=slr_colors.get(off, "0.3"),
            linewidth=0.0,
            label=slr_map.get(off, str(off)),
            zorder=3,
        )

    ax2.axvline(
        logk_tipping,
        color="#5595b5",
        linestyle="--",
        linewidth=1.6,
        alpha=0.9,
        zorder=2,
    )

    _set_sci_y(ax2)
    _beautify_ax(ax2)
    ax2.text(
        -0.09, 1.3, "b",
        transform=ax2.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )

    leg = ax2.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.98),
        borderaxespad=0.0,
        columnspacing=1.2,
        handletextpad=0.6,
    )
    for t in leg.get_texts():
        t.set_fontsize(11)

    ax2.xaxis.set_major_locator(MaxNLocator(nbins=6))


    fig.supylabel(y_label, x=0.02, fontsize=12)
    fig.supxlabel(x_label, y=0.03, fontsize=12)

    out_png = None
    if save:
        if dir_out is None:
            raise ValueError("dir_out must be provided when save=True.")
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)
        out_png = dir_out / out_name
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax1, ax2), out_png






def _apply_nature_style_3panel(font_scale=1.15, figure_dpi=400):
    """Apply a clean Nature-style plotting theme."""
    sns.set_context("paper", font_scale=font_scale)
    sns.set_style("ticks")
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "axes.linewidth": 0.8,
        "figure.dpi": figure_dpi,
    })


def _ensure_permeability_range(
    df,
    perm_col="Permeability [m^2]",
    out_col="Permeability Range",
    k_low_thr=1.58e-12,
    k_high_thr=1e-11,
):
    """Add permeability classes if they are not already present."""
    df = df.copy()
    if out_col not in df.columns:
        df[out_col] = pd.cut(
            df[perm_col],
            bins=[-float("inf"), k_low_thr, k_high_thr, float("inf")],
            labels=["Low", "Moderate", "High"],
            right=False,
        )
    return df


def _ensure_slr_scenario(
    df,
    offset_col="SLR offset",
    out_col="SLR scenario",
    slr_map=None,
    slr_order=None,
):
    """Map raw SLR offsets to display labels."""
    df = df.copy()

    if slr_map is None:
        slr_map = {
            "0m0": "Historical",
            "0m5": "+ 0.5 m",
            "1m0": "+ 1.0 m",
            "1m5": "+ 1.5 m",
            0.0: "Historical",
            0.5: "+ 0.5 m",
            1.0: "+ 1.0 m",
            1.5: "+ 1.5 m",
        }

    if slr_order is None:
        slr_order = ["Historical", "+ 0.5 m", "+ 1.0 m", "+ 1.5 m"]

    df[out_col] = df[offset_col].map(slr_map).fillna(df[offset_col].astype(str))
    df[out_col] = pd.Categorical(df[out_col], categories=slr_order, ordered=True)
    return df


def monotone_boundary(
    ax,
    df_in,
    level,
    *,
    xcol="log permeability [m^2]",
    ycol="GWR_precp_rate [-]",
    fcol="f_vert",
    ybins=14,
    min_bin=8,
    color="0.45",
    lw=1.5,
    alpha=0.9,
    linestyle="--",
):
    """
    Draw a smooth approximate boundary curve for a given f_vert level.

    The method bins by y, estimates x-crossings of the chosen level,
    then fits a smooth curve x(y).
    """
    d = df_in[[xcol, ycol, fcol]].dropna().copy()
    if d.empty:
        return None

    d["_yb"] = pd.qcut(d[ycol], q=ybins, duplicates="drop")
    anchors = []

    for _, g in d.groupby("_yb", observed=False):
        if len(g) < min_bin:
            continue

        g = g.sort_values(xcol)
        f_s = g[fcol].rolling(7, center=True, min_periods=3).median()

        idx = np.where((f_s.values[:-1] - level) * (f_s.values[1:] - level) <= 0)[0]
        if len(idx) == 0:
            continue

        j = idx[len(idx) // 2]
        x0, x1 = g[xcol].values[j], g[xcol].values[j + 1]
        f0, f1 = f_s.values[j], f_s.values[j + 1]

        if np.isfinite(f0) and np.isfinite(f1) and (f1 - f0) != 0:
            x_cross = x0 + (level - f0) * (x1 - x0) / (f1 - f0)
            anchors.append((g[ycol].median(), x_cross))

    if len(anchors) < 5:
        return None

    anchors = np.array(sorted(anchors, key=lambda t: t[0]))
    yA, xA = anchors[:, 0], anchors[:, 1]
    yy = np.linspace(yA.min(), yA.max(), 250)

    try:
        from scipy.interpolate import UnivariateSpline
        spl = UnivariateSpline(yA, xA, s=0.8 * len(yA))
        xx = spl(yy)
    except Exception:
        coeff = np.polyfit(yA, xA, deg=3)
        xx = np.polyval(coeff, yy)

    line, = ax.plot(
        xx, yy,
        color=color,
        lw=lw,
        alpha=alpha,
        solid_capstyle="round",
        zorder=6,
        linestyle=linestyle,
    )
    return line


def plot_fvert_three_panel_figure(
    df,
    dir_out=None,
    *,
    out_name="fig_fvert_panels_abc.png",
    save=True,
    show=True,
    dpi=400,
    figsize=(7.6, 7.2),
    region_col=None,
    f_lateral_max=0.4,
    f_vertical_min=0.6,
    k_low_thr=1.58e-12,
    k_high_thr=1e-11,
    perm_col="Permeability [m^2]",
    perm_range_col="Permeability Range",
    fvert_col="f_vert",
    offset_col="SLR offset",
    slr_scenario_col="SLR scenario",
    xcol_scatter="log permeability [m^2]",
    ycol_scatter="GWR_precp_rate [-]",
    panel_a_palette=None,
    hist_palette="viridis_r",
    scatter_cmap="Spectral_r",
    scatter_size=55,
    add_boundaries=True,
):
    """
    Create a Nature-style 3-panel figure:
      a) violin + box + stripplot of f_vert by permeability class
      b) histogram of f_vert by SLR scenario
      c) scatter of recharge ratio vs permeability, colored by f_vert

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple
        (ax_a, ax_b, ax_c)
    out_path : Path or None
    stats_df : pandas.DataFrame or None
    """
    _apply_nature_style_3panel(font_scale=1.15, figure_dpi=dpi)

    if panel_a_palette is None:
        panel_a_palette = ["#e74c3c", "#f1c40f", "#2ecc71"][::-1]

    df_plot = df.copy()
    df_plot = _ensure_permeability_range(
        df_plot,
        perm_col=perm_col,
        out_col=perm_range_col,
        k_low_thr=k_low_thr,
        k_high_thr=k_high_thr,
    )
    df_plot = _ensure_slr_scenario(
        df_plot,
        offset_col=offset_col,
        out_col=slr_scenario_col,
    )

    # Per-panel filtering
    df_a = df_plot.dropna(subset=[perm_range_col, fvert_col]).copy()
    df_b = df_plot.dropna(subset=[fvert_col, offset_col]).copy()
    df_c = df_plot.dropna(subset=[xcol_scatter, ycol_scatter, fvert_col]).copy()
    if region_col is not None and region_col in df_plot.columns:
        df_c = df_c.dropna(subset=[region_col]).copy()

    preferred_order = ["Low", "Moderate", "High"]
    if set(preferred_order).issubset(set(df_a[perm_range_col].astype(str).unique())):
        order_a = preferred_order
    else:
        order_a = list(df_a[perm_range_col].astype(str).unique())

    # Figure layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.0, 1.25],
        width_ratios=[1.25, 0.95],
        wspace=0.38,
        hspace=0.52,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    sns.violinplot(
        data=df_a,
        x=perm_range_col,
        y=fvert_col,
        order=order_a,
        palette=panel_a_palette,
        inner=None,
        cut=0,
        linewidth=0,
        ax=ax_a,
    )

    sns.boxplot(
        data=df_a,
        x=perm_range_col,
        y=fvert_col,
        order=order_a,
        width=0.12,
        showcaps=True,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.1},
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"linewidth": 1.0},
        fliersize=0,
        ax=ax_a,
    )

    sns.stripplot(
        data=df_a,
        x=perm_range_col,
        y=fvert_col,
        order=order_a,
        color=".25",
        size=1.5,
        alpha=0.35,
        jitter=True,
        ax=ax_a,
    )

    ax_a.set_xlabel("Permeability classification", fontsize=10, labelpad=8)
    ax_a.set_ylabel(r"Vertical salt fraction ($f_{\mathrm{vert}}$)", fontsize=10)
    ax_a.set_ylim(-0.05, 1.05)

    for ythr in [f_lateral_max, f_vertical_min]:
        ax_a.axhline(ythr, color="black", linestyle=":", linewidth=1, alpha=0.35)

    ytrans = ax_a.get_yaxis_transform()
    ax_a.text(
        0.21, 0.30, "Lateral \ndominant",
        transform=ytrans, fontsize=8, alpha=0.75,
        ha="left", va="center", clip_on=False,
        multialignment="center",
    )
    ax_a.text(
        0.21, 0.50, "Mixed \n pathway",
        transform=ytrans, fontsize=8, alpha=0.75,
        ha="left", va="center", clip_on=False,
        multialignment="center",
    )
    ax_a.text(
        0.21, 0.70, "Vertical \ndominant",
        transform=ytrans, fontsize=8, alpha=0.75,
        ha="left", va="center", clip_on=False,
        multialignment="center",
    )

    if order_a == preferred_order:
        ax_a.set_xticklabels([
            "Low\n(<1.58×10$^{-12}$ m$^2$)",
            "Moderate\n(1.58×10$^{-12}$–10$^{-11}$ m$^2$)",
            "High\n(≥10$^{-11}$ m$^2$)",
        ], fontsize=6)
    else:
        ax_a.tick_params(axis="x", labelsize=9)

    ax_a.text(
        -0.12, 1.05, r"$\mathbf{a}$",
        transform=ax_a.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="right",
    )
    sns.despine(ax=ax_a)

    sns.histplot(
        data=df_b,
        x=fvert_col,
        hue=slr_scenario_col,
        element="step",
        stat="count",
        common_norm=False,
        palette=hist_palette,
        alpha=0.12,
        linewidth=1.4,
        bins=25,
        ax=ax_b,
    )

    ax_b.set_xlabel(r"Vertical salt fraction ($f_{\mathrm{vert}}$)", fontsize=10)
    ax_b.set_ylabel("Count [-]", fontsize=10)
    ax_b.text(
        -0.12, 1.05, r"$\mathbf{b}$",
        transform=ax_b.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="right",
    )

    leg = ax_b.get_legend()
    if leg is not None:
        leg.set_title("SLR scenarios")
        try:
            leg.set_alignment("center")
        except Exception:
            pass
        try:
            leg._legend_box.align = "center"
        except Exception:
            pass
        for t in leg.get_texts():
            t.set_fontsize(7.5)
        leg.get_title().set_fontsize(8)
        leg.set_frame_on(False)

    sns.despine(ax=ax_b)

    sns.scatterplot(
        data=df_c,
        x=xcol_scatter,
        y=ycol_scatter,
        hue=fvert_col,
        palette=scatter_cmap,
        s=scatter_size,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        legend=False,
        ax=ax_c,
    )

    ax_c.set_xlabel(r"log permeability [$\mathrm{m^2}$]", fontsize=10)
    ax_c.set_ylabel(r"Groundwater recharge ratio [$-$]", fontsize=10)
    ax_c.text(
        -0.03, 1.04, r"$\mathbf{c}$",
        transform=ax_c.transAxes,
        fontsize=12, fontweight="bold",
        va="bottom", ha="right",
    )

    norm = plt.Normalize(df_c[fvert_col].min(), df_c[fvert_col].max())
    sm = plt.cm.ScalarMappable(cmap=scatter_cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax_c, shrink=0.85, aspect=25, pad=0.02)
    cbar.set_label(r"Vertical salt fraction ($f_{\mathrm{vert}}$)", fontsize=9)
    cbar.outline.set_linewidth(0.5)

    if add_boundaries:
        monotone_boundary(
            ax_c,
            df_c,
            level=f_lateral_max,
            xcol=xcol_scatter,
            ycol=ycol_scatter,
            fcol=fvert_col,
            lw=1.5,
            color="#2ecc71",
            alpha=0.95,
        )
        monotone_boundary(
            ax_c,
            df_c,
            level=f_vertical_min,
            xcol=xcol_scatter,
            ycol=ycol_scatter,
            fcol=fvert_col,
            lw=1.5,
            color="#e74c3c",
            alpha=0.95,
        )

    sns.despine(ax=ax_c)

    plt.tight_layout()

    out_path = None
    if save:
        if dir_out is None:
            raise ValueError("dir_out must be provided when save=True.")
        dir_out = Path(dir_out)
        dir_out.mkdir(parents=True, exist_ok=True)
        out_path = dir_out / out_name
        fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
        print(f"Saved figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, (ax_a, ax_b, ax_c), out_path







def get_coastal_region(
    state,
    regions_mapping=None,
):
    """
    Map a U.S. state abbreviation to a plotting region.
    """
    if regions_mapping is None:
        regions_mapping = {
            "Pacific": ["WA", "OR", "CA"],
            "Louisiana": ["LA"],
            "Others in Gulf": ["MS", "AL", "TX"],
            "Florida": ["FL"],
        }

    if state in regions_mapping["Pacific"]:
        return "Pacific"
    if state in regions_mapping["Louisiana"]:
        return "Louisiana"
    if state in regions_mapping["Others in Gulf"]:
        return "Others in Gulf"
    if state in regions_mapping["Florida"]:
        return "Florida"
    return "Atlantic"


def parse_recharge_tag(tag):
    if pd.isna(tag):
        return (None, None, None)

    parts = str(tag).split("_")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    return (None, None, None)


def prepare_lollipop_change_data(
    swi_csv,
    *,
    regions_mapping=None,
    cols=None,
    historical_window="Historical",
    historical_scenario="0_0ft",
    future_window="FarFuture",
    future_scenario="5_0ft",
):

    if regions_mapping is None:
        regions_mapping = {
            "Pacific": ["WA", "OR", "CA"],
            "Louisiana": ["LA"],
            "Others in Gulf": ["MS", "AL", "TX"],
            "Florida": ["FL"],
        }

    if cols is None:
        cols = ["Transport", "Forcing", "Lateral", "Mixed", "Vertical"]

    df = pd.read_csv(swi_csv).copy()

    df[["Model", "SSP", "Window"]] = df["recharge_tag"].apply(
        lambda x: pd.Series(parse_recharge_tag(x))
    )

    df_hist = df[
        (df["Window"] == historical_window) &
        (df["scenario"] == historical_scenario)
    ].copy()

    df_far = df[
        (df["Window"] == future_window) &
        (df["scenario"] == future_scenario)
    ].copy()

    df_filtered = pd.concat([df_hist, df_far], ignore_index=True)

    df_filtered["Transport"] = df_filtered["SWI not accelerated_area_km2"]
    df_filtered["Lateral"] = df_filtered["Lateral-dominated_area_km2"]
    df_filtered["Mixed"] = df_filtered["Mixed_area_km2"]
    df_filtered["Vertical"] = df_filtered["Vertical-dominated_area_km2"]
    df_filtered["Forcing"] = (
        df_filtered["Lateral"] +
        df_filtered["Mixed"] +
        df_filtered["Vertical"]
    )

    df_filtered["Region"] = df_filtered["state"].apply(
        lambda s: get_coastal_region(s, regions_mapping=regions_mapping)
    )

    df_conus = df_filtered.copy()
    df_conus["Region"] = "Continental US"

    df_combined = pd.concat([df_filtered, df_conus], ignore_index=True)

    model_sums = (
        df_combined
        .groupby(["Region", "Window", "SSP", "Model"])[cols]
        .sum()
        .reset_index()
    )

    hist_df = (
        model_sums[model_sums["Window"] == historical_window]
        .groupby(["Region", "Model"])[cols]
        .mean()
        .reset_index()
    )

    future_df = model_sums[model_sums["Window"] == future_window].copy()

    change_df = future_df.merge(
        hist_df,
        on=["Region", "Model"],
        suffixes=("", "_hist"),
    )

    for c in cols:
        change_df[f"{c}_change"] = change_df[c] - change_df[f"{c}_hist"]

    stats_change = (
        change_df
        .groupby(["Region", "SSP"])[[f"{c}_change" for c in cols]]
        .agg(["mean", "std"])
        .reset_index()
    )

    stats_change.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in stats_change.columns
    ]

    plot_data = stats_change.copy()

    return {
        "raw_df": df,
        "filtered_df": df_filtered,
        "combined_df": df_combined,
        "model_sums": model_sums,
        "hist_df": hist_df,
        "future_df": future_df,
        "change_df": change_df,
        "stats_change": stats_change,
        "plot_data": plot_data,
    }


def plot_inundation_change_lollipop(
    swi_csv,
    output_dir,
    *,
    plot_csv_name="Dominant_Pathway_Change_Lollipop_plotdata.csv",
    figure_name="Inundation_Change_Lollipop.png",
    save_figure=True,
    show=True,
    dpi=400,
    figsize=(7.2, 5.2),
    regions_mapping=None,
    region_colors=None,
    plot_region_colors=None,
    regions_order=None,
    cat_labels=None,
    cat_order=None,
    scenario_styles=None,
):
    """
    Create a 2x3 regional lollipop plot of pathway-area change.

    Returns
    -------
    dict
        Contains:
            - fig
            - axes
            - plot_data
            - stats_change
            - plot_csv
            - figure_path
            - prep
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if regions_mapping is None:
        regions_mapping = {
            "Pacific": ["WA", "OR", "CA"],
            "Louisiana": ["LA"],
            "Others in Gulf": ["MS", "AL", "TX"],
            "Florida": ["FL"],
        }

    if region_colors is None:
        region_colors = {
            "Gulf": "#E69F00",
            "Atlantic_S": "#56B4E9",
            "Atlantic_N": "#009E73",
            "Pacific_S": "#F0E442",
            "Pacific_N": "#0072B2",
            "Continental US": "#D55E00",
        }

    if plot_region_colors is None:
        plot_region_colors = {
            "Continental US": region_colors["Continental US"],
            "Pacific": region_colors["Pacific_N"],
            "Atlantic": region_colors["Atlantic_N"],
            "Louisiana": region_colors["Atlantic_S"],
            "Florida": region_colors["Pacific_S"],
            "Others in Gulf": region_colors["Gulf"],
        }

    if regions_order is None:
        regions_order = [
            "Continental US",
            "Pacific",
            "Atlantic",
            "Others in Gulf",
            "Louisiana",
            "Florida",
        ]

    if cat_labels is None:
        cat_labels = {
            "Transport": "Transport-limited",
            "Forcing": "Forcing-sensitive",
            "Lateral": "Lateral-dominated",
            "Mixed": "Mixed pathway",
            "Vertical": "Vertical-dominated",
        }

    if cat_order is None:
        cat_order = ["Transport", "Forcing", "Lateral", "Mixed", "Vertical"]

    if scenario_styles is None:
        scenario_styles = {
            "ssp126": {"color": "#1f78b4", "label": "SSP1-2.6", "marker": "o"},
            "ssp245": {"color": "#fdbf6f", "label": "SSP2-4.5", "marker": "s"},
            "ssp585": {"color": "#e31a1c", "label": "SSP5-8.5", "marker": "D"},
        }

    prep = prepare_lollipop_change_data(
        swi_csv,
        regions_mapping=regions_mapping,
        cols=cat_order,
    )
    stats_change = prep["stats_change"]
    plot_data = prep["plot_data"]

    plot_csv = None


    plt.rcParams.update({
        "font.size": 8,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
    })

    fig, axes = plt.subplots(
        2, 3,
        figsize=figsize,
        sharey="row",
        sharex=True,
        dpi=dpi,
    )

    axes_flat = axes.flatten()
    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for i, region in enumerate(regions_order):
        ax = axes_flat[i]
        reg_stats = stats_change[stats_change["Region"] == region]

        y_coords = np.arange(len(cat_order))
        offsets = [-0.18, 0.0, 0.18]

        ax.axvline(0, color="black", lw=0.7, alpha=0.9)

        for s_idx, ssp in enumerate(["ssp126", "ssp245", "ssp585"]):
            ssp_rows = reg_stats[reg_stats["SSP"] == ssp]
            if ssp_rows.empty:
                continue

            row = ssp_rows.iloc[0]
            style = scenario_styles[ssp]

            for c_idx, cat in enumerate(cat_order):
                y_pos = y_coords[c_idx] + offsets[s_idx]
                mean_val = row.get(f"{cat}_change_mean", np.nan)
                std_val = row.get(f"{cat}_change_std", np.nan)

                if not np.isfinite(mean_val):
                    continue

                ax.hlines(
                    y_pos, 0, mean_val,
                    color=style["color"],
                    alpha=0.2,
                    lw=0.8,
                )
                ax.errorbar(
                    mean_val,
                    y_pos,
                    xerr=std_val if np.isfinite(std_val) else None,
                    fmt=style["marker"],
                    color=style["color"],
                    markersize=3.5,
                    capsize=1.5,
                    elinewidth=0.6,
                    label=style["label"] if i == 0 and c_idx == 0 else "",
                )

        ax.set_title(
            f"{region}",
            loc="center",
            fontweight="bold",
            pad=10,
            bbox=dict(
                facecolor=plot_region_colors[region],
                alpha=0.3,
                edgecolor="none",
                boxstyle="round,pad=0.3",
            ),
        )
        ax.set_title(
            f"{panel_labels[i]}",
            loc="left",
            fontweight="bold",
            pad=10,
        )

        if i % 3 == 0:
            ax.set_yticks(y_coords)
            ax.set_yticklabels([cat_labels[c] for c in cat_order])

        ax.invert_yaxis()
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Change in area [km$^2$]")

    fig.legend(
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    plt.subplots_adjust(
        bottom=0.15,
        hspace=0.45,
        wspace=0.15,
        left=0.18,
        right=0.95,
        top=0.88,
    )

    figure_path = None
    if save_figure:
        figure_path = output_dir / figure_name
        plt.savefig(figure_path, dpi=dpi)
        print(f"Figure saved to: {figure_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)