# src/plotting_utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep


def plot_efficiencies_single_region(
    efficiencies, pt_bins, region, outputfile,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    markers=None, mfc=None, colors=None, labs=None,
    extra_text=None
):
    assert markers is not None
    assert mfc is not None
    assert colors is not None
    assert labs is not None

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(12, 13), dpi=300)

    for l, (key, (pt_centers, eff, errors)) in enumerate(efficiencies.items()):
        ax.errorbar(
            pt_centers, eff, yerr=errors,
            fmt=markers[l],
            color=colors[l],
            label=labs[l],
            linestyle='',
            mfc=mfc[l],
            markersize=14,
            capsize=8,
            linewidth=3
        )

    # Reference lines
    ax.axhline(0.9, color='black', linestyle='--', linewidth=1)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)

    xmax = float(np.max(pt_bins))
    ax.set_xlim(0.0, xmax)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))

    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))

    ax.set_xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=24, loc="right")
    ax.set_ylabel("Efficiency", fontsize=24)
    ax.set_ylim(0.0, 1.15)

    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.35)

    hep.cms.label(
        ax=ax,
        llabel=llabel,
        rlabel=rlabel,
        loc=0
    )

    ax.text(
        0.03, 0.95, region,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=24,
        style="italic"
    )

    if extra_text:
        ax.text(
            0.03, 0.88, extra_text,
            transform=ax.transAxes,
            fontsize=18
        )

    ax.legend()
    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)




def plot_ridgeline(
    values_by_ptbin,
    pt_bins,
    xlabel,
    outputfile,
    llabel="Phase-2 Sim. Prelim.",
    rlabel="PU 200 (14 TeV)",
    title=None,
    xlim=None,
    title_anchor=(0.39, 0.96),
    title_ha="center",
    xlim_hist=None,           # histogram range used internally
    nbins=250,                
    smooth_k=11,              # moving average window (smoothing)
    smooth_pad_mode="edge",   # "edge" or "reflect"
    normalize_mode="peak",    # "peak" or "area" (pdf-like within xlim_hist)
    ridge_height=0.9,         
    show_q1q3=False,          # optionally draw Q1/Q3
):
    """
    Ridgeline plot of per-pt-bin distributions.
    """

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(6.5, 11), dpi=300)

    n = len(values_by_ptbin)
    offsets = np.arange(n)

    nonempty = [np.asarray(v, dtype=float) for v in values_by_ptbin if len(v) > 0]
    if len(nonempty) == 0:
        ax.text(0.2, 0.5, "No entries", transform=ax.transAxes, fontsize=16)
        ax.set_xlabel(xlabel, fontsize=18, loc="right")
        ax.set_ylabel(r"$p_T^{GEN}$ bin [GeV]", fontsize=18, labelpad=20)

        if title:
            ax.text(
                title_anchor[0], title_anchor[1], title,
                transform=ax.transAxes,
                fontsize=13,
                ha=title_ha,
                va="top",
            )

        hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
        plt.tight_layout()
        plt.savefig(outputfile, dpi=300)
        plt.close(fig)
        return

    all_vals = np.concatenate(nonempty)
    all_vals = all_vals[np.isfinite(all_vals)]

    # ---- choose plot window
    if xlim is None:
        lo, hi = np.percentile(all_vals, [1, 99])
        span = max(hi - lo, 1e-6)
        xlim = (lo - 0.2 * span, hi + 0.2 * span)

    # ---- choose histogram window (can be wider to capture out-of-range tails)
    if xlim_hist is None:
        # default: a bit wider than xlim so smoothing near edges doesn't "die"
        span = max(xlim[1] - xlim[0], 1e-9)
        xlim_hist = (xlim[0] - 0.25 * span, xlim[1] + 0.25 * span)

    span_plot = max(xlim[1] - xlim[0], 1e-9)
    x_text = xlim[1] - 0.02 * span_plot

    # Histogram edges
    edges = np.linspace(xlim_hist[0], xlim_hist[1], nbins)
    x = 0.5 * (edges[:-1] + edges[1:])  # bin centers
    dx = float(edges[1] - edges[0])

    def smooth_hist(y, k=smooth_k, pad_mode=smooth_pad_mode):
        """
        Moving-average smoothing with boundary-aware padding.
        Avoids the 'tails go to zero at edges' artifact from zero-padding.
        """
        y = np.asarray(y, dtype=float)
        if k is None or k < 2 or y.size < k:
            return y
        ker = np.ones(k, dtype=float) / float(k)
        p = k // 2
        # pad_mode: "edge" (repeat edge) or "reflect" (mirror)
        ypad = np.pad(y, (p, p), mode=pad_mode)
        return np.convolve(ypad, ker, mode="valid")

    cmap = plt.get_cmap("viridis")
    denom = max(n - 1, 1)

    for i, vals in enumerate(values_by_ptbin):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        col = cmap(i / denom)

        # Bin label outside left axis (y-axis transform)
        ax.text(
            -0.02,
            offsets[i] + 0.05,
            f"({pt_bins[i]:.0f}, {pt_bins[i+1]:.0f}]",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=11.5,
            clip_on=False,
        )

        if vals.size == 0:
            continue

        hist, _ = np.histogram(vals, bins=edges, density=True)

        y = smooth_hist(hist)

        # Optional renormalization:
        # - "peak": scale each ridge to same max height (shape-only, not probabilistic area)
        # - "area": make ridge integrate to 1 over histogram range (PDF-like)
        if normalize_mode == "area":
            area = float(np.sum(y) * dx)
            if area > 0:
                y = y / area
            y_max = float(np.max(y))
            if y_max > 0:
                y = (y / y_max) * ridge_height
        elif normalize_mode == "peak":
            y_max = float(np.max(y))
            if y_max > 0:
                y = (y / y_max) * ridge_height
        else:
            raise ValueError(f"Unknown normalize_mode: {normalize_mode}")

        # If you want to draw only the visible window but still use the wider histogram for shape:
        mask = (x >= xlim[0]) & (x <= xlim[1])
        x_plot = x[mask]
        y_plot = y[mask]

        ax.fill_between(
            x_plot,
            offsets[i],
            offsets[i] + y_plot,
            alpha=0.95,
            linewidth=0,
            color=col,
        )

        # ---- Quantiles (computed on full vals, independent of plotting)
        q1, q2, q3 = np.percentile(vals, [25, 50, 75])

        # Vertical line segments (only drawn if within plot xlim)
        def _vline(xpos, color, lw):
            if xlim[0] <= xpos <= xlim[1]:
                ax.vlines(
                    xpos,
                    offsets[i],
                    offsets[i] + ridge_height,
                    colors=color,
                    linestyles="-",
                    linewidth=lw,
                    zorder=5,
                )

        _vline(q2, "red", 1.2)
        if show_q1q3:
            _vline(q1, "red", 0.9)
            _vline(q3, "red", 0.9)

        # Median annotation (always printed; if median outside window, annotate as out-of-range)
        if q2 < xlim[0]:
            q2_txt = f"Q2 < {xlim[0]:.2f}"
        elif q2 > xlim[1]:
            q2_txt = f"Q2 > {xlim[1]:.2f}"
        else:
            q2_txt = f"Q2 = {q2:.3f}"

        ax.text(
            x_text,
            offsets[i] + 0.03,
            q2_txt,
            ha="right",
            va="bottom",
            fontsize=10,
            color="red",
            zorder=6,
            clip_on=True,
        )

    # Reference line at 0
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1.2)

    ax.set_yticks([])
    ax.set_xlim(*xlim)

    ax.set_xlabel(xlabel, fontsize=18, loc="right")
    ax.set_ylabel(r"$p_T^{GEN}$ bin [GeV]", fontsize=18, labelpad=75)

    if title:
        ax.text(
            title_anchor[0], title_anchor[1], title,
            transform=ax.transAxes,
            fontsize=13,
            ha=title_ha,
            va="top",
        )

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)

    fig.subplots_adjust(top=0.90)
    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)




def plot_agreement_fraction_hist(
    fractions,
    outputfile,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    title=None
):
    """
    Histogram of per-event agreement fraction:
      f = | GEN matched by alt AND GEN matched by PF AK4 | / | GEN matched by PF AK4 |
    """

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=300)

    fractions = np.asarray(fractions, dtype=float)

    if fractions.size == 0:
        ax.text(0.2, 0.5, "No entries", transform=ax.transAxes, fontsize=14)
        hep.cms.label(
            ax=ax,
            llabel=llabel,
            rlabel=rlabel,
            loc=0,
            fontsize=14.0, 
        )
        if title:
            ax.text(
                0.05, 0.92, title,
                transform=ax.transAxes,
                ha="left", va="top",
                fontsize=12.5,
                fontweight="bold"
            )
        ax.set_xlabel(r"$\frac{n(\mathrm{Alt. and GEN matched }\cap\mathrm{ AK4PF and GEN matched})}{n(\mathrm{AK4PF and GEN matched})}$", fontsize=16)
        ax.set_ylabel("Events", fontsize=16)
        plt.tight_layout()
        plt.savefig(outputfile, dpi=300)
        plt.close(fig)
        return

    bins = np.linspace(0.0, 1.0, 21)  
    ax.hist(fractions, bins=bins, histtype="step", linewidth=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(
        r"$\frac{n(\mathrm{GEN}\times\mathrm{Alt} \cap \mathrm{GEN}\times\mathrm{AK4PF})}"
        r"{n(\mathrm{GEN}\times\mathrm{AK4PF})}$",
        fontsize=16
    )
    ax.set_ylabel("Events", fontsize=16)
    ax.grid(alpha=0.25)

    hep.cms.label(
        ax=ax,
        llabel=llabel,
        rlabel=rlabel,
        loc=0,
        fontsize=14.0,
    )

    if title:
        ax.text(
            0.05, 0.92, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12.5,
            fontweight="bold"
        )

    ax.tick_params(axis="both", which="major", labelsize=11)

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_agreement_fraction_multi(
    fractions_by_algo,
    outputfile,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    title=None,
    bins=None,
):
    """
    Option A agreement plot: for a fixed INPUT, overlay agreement histograms
    for multiple algorithms.

    Parameters
    ----------
    fractions_by_algo : dict[str, array-like]
        Mapping: algo_name -> per-event agreement fractions
        (already defined w.r.t. PF·AntiKt in the processing step).
    """
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=300)

    if bins is None:
        bins = np.linspace(0.0, 1.0, 21)

    # Consistent ordering (AntiKt first if present, then others alphabetically)
    algos = list(fractions_by_algo.keys())
    algos_sorted = []
    if "AntiKt" in algos:
        algos_sorted.append("AntiKt")
    algos_sorted += sorted([a for a in algos if a != "AntiKt"])

    any_entries = False
    for algo in algos_sorted:
        fr = np.asarray(fractions_by_algo[algo], dtype=float)
        fr = fr[np.isfinite(fr)]
        if fr.size == 0:
            continue
        any_entries = True

        ax.hist(
            fr,
            bins=bins,
            histtype="step",
            linewidth=2.2,
            label=algo
        )

    ax.set_xlim(0.0, 1.0)

    ax.set_xlabel(
        r"$\left|\mathrm{GEN}\times\mathrm{Alt}\ \cap\ \mathrm{GEN}\times\mathrm{AK4PF}\right|"
        r"\,/\,\left|\mathrm{GEN}\times\mathrm{AK4PF}\right|$",
        fontsize=13.5
    )
    ax.set_ylabel("Events", fontsize=14)
    ax.grid(alpha=0.22)

    hep.cms.label(
        ax=ax,
        llabel=llabel,
        rlabel=rlabel,
        loc=0,
        fontsize=13.0,
    )

    if title:
        ax.text(
            0.05, 0.92, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=12.0,
            fontweight="bold"
        )

    if any_entries:
        ax.legend(frameon=False, fontsize=11, loc="center left")

    ax.tick_params(axis="both", which="major", labelsize=11)

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_violin_response(
    resp_by_ptbin,
    pt_bins,
    outputfile,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    title=None,
    ylim=(0.0, 3.5)
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    data_all = [np.asarray(v, dtype=float) for v in resp_by_ptbin]
    nonempty_idx = [i for i, v in enumerate(data_all) if v.size > 0]

    if len(nonempty_idx) == 0:
        ax.text(0.2, 0.5, "No entries", transform=ax.transAxes, fontsize=16)
        ax.set_ylabel(r"$p_T^{RECO}/p_T^{GEN}$", fontsize=16)
        ax.set_xlabel(r"GEN jet $p_T$ bin [GeV]", fontsize=16)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)

        hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, fontsize=14, loc=0)

        # Title block inside axes (top-right)
        if title:
            ax.text(
                0.98, 0.92, title,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=14
            )

        plt.tight_layout()
        plt.savefig(outputfile, dpi=300)
        plt.close(fig)
        return

    data = [data_all[i] for i in nonempty_idx]
    positions = np.array(nonempty_idx, dtype=float) + 1.0

    vp = ax.violinplot(
        data,
        positions=positions,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # Reference line at response=1
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1)

    labels = [f"{pt_bins[i]:.0f}-{pt_bins[i+1]:.0f}" for i in range(len(pt_bins) - 1)]
    ax.set_xticks(np.arange(len(labels)) + 1)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    ax.set_ylabel(r"$p_T^{RECO}/p_T^{GEN}$", fontsize=16)
    ax.set_xlabel(r"GEN jet $p_T$ bin [GeV]", fontsize=16)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, fontsize=18, loc=0)

    if title:
        ax.text(
            0.96, 0.92, title,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=14
        )

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_purity_or_fake_vs_ptreco(
    pt_centers,
    values,
    errors,
    outputfile,
    ylabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=300)

    ax.errorbar(pt_centers, values, yerr=errors, fmt="o", linestyle="", capsize=6)
    ax.set_xlabel(r"$p_T^{RECO}$ [GeV]", fontsize=14, loc="right")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)

    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_multiplicity_hist(
    njet_values,
    outputfile,
    xlabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    max_n=25
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)

    njet_values = np.asarray(njet_values, dtype=int)
    njet_values = njet_values[np.isfinite(njet_values)]

    ax.hist(njet_values, histtype="step", linewidth=2.0)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Events", fontsize=13)
    ax.grid(alpha=0.22)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_ht_hist(
    ht_values,
    outputfile,
    xlabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)

    ht_values = np.asarray(ht_values, dtype=float)
    ht_values = ht_values[np.isfinite(ht_values)]

    ax.hist(ht_values, bins=40, histtype="step", linewidth=2.0)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Events", fontsize=13)
    ax.grid(alpha=0.22)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.3, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_quantiles_vs_ptgen(
    pt_centers,
    q16, q50, q84,
    outputfile,
    ylabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    ylim=None
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=300)

    pt_centers = np.asarray(pt_centers, dtype=float)
    q16 = np.asarray(q16, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q84 = np.asarray(q84, dtype=float)

    ok = np.isfinite(pt_centers) & np.isfinite(q16) & np.isfinite(q50) & np.isfinite(q84)
    pt_centers, q16, q50, q84 = pt_centers[ok], q16[ok], q50[ok], q84[ok]

    ax.fill_between(pt_centers, q16, q84, alpha=0.25, step=None)
    ax.plot(pt_centers, q50, marker="o", linestyle="-", linewidth=1.6)

    ax.set_xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=14, loc="right")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(alpha=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_turnon_curves(
    pt_centers,
    curves,  # dict: threshold -> (eff, err)
    outputfile,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.8, 5.8), dpi=300)

    for thr, (eff, err) in curves.items():
        ax.errorbar(pt_centers, eff, yerr=err, fmt="o", linestyle="", capsize=5, label=fr"$p_T^{{RECO}}>{int(thr)}$")

    ax.set_xlabel(r"$p_T^{GEN}$ [GeV]", fontsize=14, loc="right")
    ax.set_ylabel("Turn-on efficiency", fontsize=14)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=11)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_neighbor_count_hists(
    counts,
    outputfile,
    xlabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    max_k=8
):
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.0, 5.2), dpi=300)

    counts = np.asarray(counts, dtype=int)
    counts = counts[np.isfinite(counts)]
    counts = np.clip(counts, 0, max_k)

    bins = np.arange(0, max_k + 2) - 0.5
    ax.hist(counts, bins=bins, histtype="step", linewidth=2.0)

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("nJets", fontsize=13)
    ax.grid(alpha=0.22)
    ax.set_xlim(-0.5, max_k + 0.5)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)

def plot_f1_vs_threshold(
    thresholds,
    f1,
    precision,
    recall,
    outputfile,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    title=None,
):
    """
    Plot F1 (and optionally precision/recall) vs threshold.

    Intended usage:
      - thresholds: array of pT thresholds [GeV]
      - f1, precision, recall: arrays same length
    """
    thresholds = np.asarray(thresholds, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=300)

    ax.plot(thresholds, f1, marker="o", linestyle="-", linewidth=2.0, label="F1")
    ax.plot(thresholds, precision, marker="o", linestyle="--", linewidth=1.6, label="Precision")
    ax.plot(thresholds, recall, marker="o", linestyle=":", linewidth=1.8, label="Recall")

    ax.set_xlabel(r"Threshold $T$ [GeV]  (use $p_T^{GEN} >= T$ and $p_T^{RECO} >= T$)", fontsize=12.5, loc="right")
    ax.set_ylabel("Score", fontsize=13)
    ax.set_ylim(0.0, 1.25)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)

    if title:
        ax.text(
            0.05, 0.92, title,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11.5,
            fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)


def plot_akcompat_overlay_hist_subplots(
    values_by_ptbin_by_algo,
    pt_bins,
    outputfile,
    xlabel,
    title,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    bins=np.linspace(0, 1, 25),
    xlim=(0, 1),
    ncols=None,  # accept for backward compatibility
):
    """
    Vertical stack of overlay histograms (one row per pT bin), with:
      - low pT at bottom
      - tight subplot spacing
      - pT bin labels to the far left (no overlap)
      - figure-level legend and title in a dedicated top band (no overlap)
      - larger usable axes area
      - smaller tick labels
      - per-panel y autoscale: ylim = 1.25 * max bin count in that slice
    """

    hep.style.use("CMS")

    pt_bins = np.asarray(pt_bins, dtype=float)
    bins = np.asarray(bins, dtype=float)

    nrows = len(values_by_ptbin_by_algo)
    if nrows == 0:
        fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=300)
        ax.text(0.2, 0.5, "No entries", transform=ax.transAxes, fontsize=14)
        hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Jets")
        plt.tight_layout()
        plt.savefig(outputfile, dpi=300)
        plt.close(fig)
        return

    all_algos = set()
    for d in values_by_ptbin_by_algo:
        all_algos |= set(d.keys())
    all_algos = sorted(all_algos)

    height_per_row = 0.82
    fig_w = 9.0
    fig_h = max(3.2, height_per_row * nrows + 2.0)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        sharex=True,
        figsize=(fig_w, fig_h),
        dpi=300
    )
    if nrows == 1:
        axes = [axes]

    legend_handles, legend_labels = None, None

    n = nrows  # number of pt bins
    for i, ax in enumerate(axes):
        # top=highest pT, bottom=lowest pT
        ibin = n - 1 - i
        d = values_by_ptbin_by_algo[ibin]

        ymax = 0.0
        any_drawn = False

        for algo in all_algos:
            vals = d.get(algo, None)
            if vals is None:
                continue
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            counts, _ = np.histogram(vals, bins=bins)
            ymax = max(ymax, float(np.max(counts)) if counts.size else 0.0)

            ax.hist(vals, bins=bins, histtype="step", linewidth=1.7, label=algo)
            any_drawn = True

        ax.set_xlim(*xlim)
        ax.grid(alpha=0.16)

        if ymax > 0:
            ax.set_ylim(0.0, 1.25 * ymax)
        else:
            ax.set_ylim(0.0, 1.0)

        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)

        ax.set_ylabel("Jets", fontsize=10.5, labelpad=8)

        lo = pt_bins[ibin]
        hi = pt_bins[ibin + 1]
        ax.text(
            -0.24, 0.50,
            f"{lo:.0f}–{hi:.0f} GeV",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            clip_on=False,
        )

        if legend_handles is None and any_drawn:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    axes[-1].set_xlabel(xlabel, fontsize=11.0, loc="right")

    hep.cms.label(
        ax=axes[0],
        llabel=llabel,
        rlabel=rlabel,
        loc=0,
        fontsize=9.,
    )

    # --- Legend: 1 or 2 columns; avoid overlapping title text
    if legend_handles is not None and len(legend_handles) > 0:
        # if >6 entries, use 2 columns, otherwise 1
        ncol = 2 if len(legend_handles) > 6 else 1
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(0.73, 0.77),  # to the right of the plots
            frameon=False,
            fontsize=10,
            ncol=1,                      
            handlelength=2.2,
            columnspacing=1.0,
        )

    if title:
        fig.text(
            0.74, 0.855,  
            title,
            ha="left",
            va="top",
            fontsize=11.0,
            fontweight="bold"
        )

    fig.subplots_adjust(
        left=0.34,   
        right=0.7,   
        bottom=0.07,
        top=0.90,     
        hspace=0.18 
    )

    plt.savefig(outputfile, dpi=300)
    plt.close(fig)




def plot_eff_or_fake_vs_pt(
    pt_centers,
    values,
    errors,
    outputfile,
    ylabel,
    title=None,
    llabel="Phase-2 Simulation Preliminary",
    rlabel="PU 200 (14 TeV)",
    ylim=(0.0, 1.15),
):

    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=300)

    ax.errorbar(pt_centers, values, yerr=errors, fmt="o", linestyle="", capsize=6)

    ax.set_xlabel(r"$p_T$ [GeV]", fontsize=14, loc="right")
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)

    hep.cms.label(ax=ax, llabel=llabel, rlabel=rlabel, loc=0, fontsize=12)
    if title:
        ax.text(0.05, 0.92, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(outputfile, dpi=300)
    plt.close(fig)