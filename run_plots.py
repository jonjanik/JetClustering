# run_plots.py
import os
import argparse
import importlib.util
import numpy as np

from src.utils import ensure_dir
from src.plotting_utils import (
    plot_efficiencies_single_region,
    plot_ridgeline,
    plot_agreement_fraction_hist,
    plot_violin_response,

    plot_purity_or_fake_vs_ptreco,
    plot_multiplicity_hist,
    plot_ht_hist,
    plot_quantiles_vs_ptgen,
    plot_turnon_curves,

    plot_f1_vs_threshold,
    plot_akcompat_overlay_hist_subplots,
    plot_eff_or_fake_vs_pt,
)

def load_cfg_from_path(cfg_path: str):
    cfg_path = os.path.abspath(cfg_path)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    spec = importlib.util.spec_from_file_location("user_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load config: {cfg_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def config_tag_from_path(cfg_path: str) -> str:
    base = os.path.basename(cfg_path)
    if base.endswith(".py"):
        base = base[:-3]
    return base

def parse_args():
    ap = argparse.ArgumentParser(description="Make plots from cached processing outputs.")
    ap.add_argument("--config", "-c", default="config.py",
                    help="Path to config file, e.g. configs/example_config.py (default: config.py)")
    return ap.parse_args()

def sanitize(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("(", "")
         .replace(")", "")
         .replace(",", "")
         .replace("=", "")
         .replace("|", "")
         .replace("≥", "ge")
         .replace("<", "lt")
         .replace(".", "p")
    )

def resolve_eff_style(cfg, input_name, algo_name, curve_key, fallback_index):
    if curve_key in getattr(cfg, "EFF_STYLE_KEY_MAP", {}):
        st = cfg.EFF_STYLE_KEY_MAP[curve_key]
        return st.get("marker", "o"), st.get("color", "black"), st.get("mfc", "none"), st.get("label", curve_key)

    if (input_name, algo_name) in getattr(cfg, "EFF_STYLE_MAP", {}):
        st = cfg.EFF_STYLE_MAP[(input_name, algo_name)]
        return st.get("marker", "o"), st.get("color", "black"), st.get("mfc", "none"), st.get("label", curve_key)

    defaults = getattr(cfg, "EFF_STYLE_DEFAULTS", [{"marker": "o", "color": "black", "mfc": "none"}])
    st = defaults[fallback_index % len(defaults)]
    return st.get("marker", "o"), st.get("color", "black"), st.get("mfc", "none"), curve_key

def load_matches(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"matches__{sanitize(inp)}__{sanitize(algo)}.npz")
    if not os.path.exists(f):
        raise RuntimeError(f"Missing match cache: {f}")
    return np.load(f)

def load_denoms(cache_dir):
    f = os.path.join(cache_dir, "denom_genjets.npz")
    if not os.path.exists(f):
        raise RuntimeError(f"Missing denom cache: {f} (run run_processing.py first)")
    return np.load(f)

def load_recomatch(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"recomatch__{sanitize(inp)}__{sanitize(algo)}.npz")
    if not os.path.exists(f):
        raise RuntimeError(f"Missing recomatch cache: {f} (run run_processing.py first)")
    return np.load(f)

def load_event_metrics(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"event_metrics__{sanitize(inp)}__{sanitize(algo)}.npz")
    if not os.path.exists(f):
        raise RuntimeError(f"Missing event_metrics cache: {f} (run run_processing.py first)")
    return np.load(f)


def get_region_defs(cfg):
    # Always have an explicit inclusive region label
    base = {"Inclusive eta gen jet": lambda eta: np.ones_like(eta, dtype=bool)}

    if getattr(cfg, "REGION_SPLIT", {}).get("enabled", True):
        defs = dict(getattr(cfg, "REGION_SPLIT", {}).get("definitions", {}))

        # If user already defined something called "Inclusive", drop/override it to avoid ambiguity
        if "Inclusive" in defs:
            defs.pop("Inclusive")

        defs["Inclusive eta gen jet"] = base["Inclusive eta gen jet"]
        return defs

    return base

def get_zcat_defs(cfg):
    cats = {"Inclusive dz": lambda dz_cat: np.ones_like(dz_cat, dtype=bool)}

    if getattr(cfg, "Z_SPLIT", {}).get("enabled", False):
        thr = float(getattr(cfg, "Z_SPLIT", {}).get("dz_cm", 1.0))
        cats[f"|dz| < {thr:.1f} cm"] = lambda dz_cat: (dz_cat == 0)
        cats[f"|dz| ≥ {thr:.1f} cm"] = lambda dz_cat: (dz_cat == 1)

    return cats

def _title_lines(proc_label, region_name, zcat_name, extra=None):
    lines = [proc_label]
    if extra:
        lines.append(extra)
    lines.append(region_name)
    if zcat_name != "Inclusive":
        lines.append(zcat_name)
    return "\n".join(lines)

def _bin_quantiles(x, y, bins, q=(16, 50, 84)):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    bins = np.asarray(bins, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])

    qvals = [np.full(len(centers), np.nan, dtype=float) for _ in q]

    b = np.digitize(x, bins) - 1
    for i in range(len(centers)):
        mask = (b == i) & np.isfinite(y)
        if np.sum(mask) < 2:
            continue
        qq = np.percentile(y[mask], list(q))
        for k in range(len(q)):
            qvals[k][i] = float(qq[k])

    return centers, qvals

def _eff_and_err(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    eff = np.divide(num, den, out=np.zeros_like(num), where=(den > 0))
    err = np.zeros_like(eff)
    ok = den > 0
    err[ok] = np.sqrt(eff[ok] * (1.0 - eff[ok]) / den[ok])
    return eff, err

def load_akcompat(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"akcompat__{sanitize(inp)}__{sanitize(algo)}.npz")
    return np.load(f) if os.path.exists(f) else None

def load_akmatch_ref(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"akmatch_ref__{sanitize(inp)}__{sanitize(algo)}.npz")
    return np.load(f) if os.path.exists(f) else None

def load_akmatch_alt(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"akmatch_alt__{sanitize(inp)}__{sanitize(algo)}.npz")
    return np.load(f) if os.path.exists(f) else None

def load_akcompat_gen(cache_dir, inp, algo):
    f = os.path.join(cache_dir, f"akcompat_gen__{sanitize(inp)}__{sanitize(algo)}.npz")
    return np.load(f) if os.path.exists(f) else None



def title_reco_reco(proc_label, input_name, ref_algo):
    """
    Clean title for reco–reco (AK-compat) comparisons.
    No region/zcat text.
    """
    return f"{proc_label}\n{input_name} (ref: {ref_algo})"


def run(cfg, cfg_tag: str):
    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    ensure_dir(out_root)

    enabled_inputs = [k for k, v in cfg.INPUTS.items() if v]
    enabled_algos = [name for name, a in cfg.ALGORITHMS.items() if a.get("enabled", False)]

    region_defs = get_region_defs(cfg)
    zcat_defs = get_zcat_defs(cfg)

    # F1 thresholds (optional). If not set, reuse jet_thresholds
    f1_thresholds = cfg.PT_BINS.get("f1_thresholds", None)
    if f1_thresholds is None:
        f1_thresholds = cfg.PT_BINS.get("jet_thresholds", np.array([20, 30, 40, 50], dtype=float))
    f1_thresholds = np.asarray(f1_thresholds, dtype=float)

    for proc, pinfo in cfg.PROCESSES.items():
        plabel = pinfo.get("label", proc)

        out_proc = os.path.join(out_root, proc)
        cache_dir = os.path.join(out_proc, "cache")
        if not os.path.isdir(cache_dir):
            raise RuntimeError(f"Cache directory not found: {cache_dir}. Run run_processing.py with the same --config.")

        denom = load_denoms(cache_dir)
        gen_pt_all = denom["gen_pt"].astype(np.float32)
        gen_eta_all = denom["gen_eta"].astype(np.float32)
        dz_cat_all = denom["dz_cat"].astype(np.int32)

        for zcat_name, zcat_mask_fn in zcat_defs.items():
            out_z = os.path.join(out_proc, sanitize(zcat_name))
            ensure_dir(out_z)

            zmask = zcat_mask_fn(dz_cat_all)

            for region_name, region_fn in region_defs.items():
                out_reg = os.path.join(out_z, region_name)
                ensure_dir(out_reg)

                rmask = region_fn(gen_eta_all)

                denom_mask = zmask & rmask
                gen_pt_denom = gen_pt_all[denom_mask]

                region_label = r"Inclusive $\eta^\mathrm{GEN}$" if region_name == "Inclusive" else region_name

                extra_lines = [plabel]
                if zcat_name != "Inclusive":
                    extra_lines.append(zcat_name)
                extra_text = "\n".join(extra_lines)

                # ---------------- Efficiency ----------------
                if cfg.STUDIES.get("efficiency", False):
                    pt_bins_eff = cfg.PT_BINS["efficiency"]
                    pt_cent = 0.5 * (pt_bins_eff[:-1] + pt_bins_eff[1:])
                    total = np.histogram(gen_pt_denom, bins=pt_bins_eff)[0].astype(float)

                    eff_curves = {}
                    keys = []

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m_mask = m_z & m_r

                            m_gen_pt = rec["gen_pt"].astype(np.float32)[m_mask]
                            matched = np.histogram(m_gen_pt, bins=pt_bins_eff)[0].astype(float)

                            eff, err = _eff_and_err(matched, total)

                            key = f"{inp} · {algo}"
                            eff_curves[key] = (pt_cent, eff, err)
                            keys.append(key)

                    markers, colors, mfc, labs = [], [], [], []
                    for i, k in enumerate(keys):
                        inp = k.split("·")[0].strip()
                        alg = k.split("·")[1].strip()
                        mk, col, face, lab = resolve_eff_style(cfg, inp, alg, k, i)
                        markers.append(mk)
                        colors.append(col)
                        mfc.append(face)
                        labs.append(lab)

                    plot_efficiencies_single_region(
                        {k: eff_curves[k] for k in keys},
                        pt_bins_eff,
                        region_label,
                        os.path.join(out_reg, "efficiency.png"),
                        llabel=cfg.PLOT_LABELS["llabel"],
                        rlabel=cfg.PLOT_LABELS["rlabel"],
                        markers=markers,
                        mfc=mfc,
                        colors=colors,
                        labs=labs,
                        extra_text=extra_text
                    )

                # ---------------- Ridgelines ----------------
                if cfg.STUDIES.get("response_ridgeline", False):
                    pt_bins_r = cfg.PT_BINS["ridgeline"]
                    out_ridge = os.path.join(out_reg, "ridgeline")
                    ensure_dir(out_ridge)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m = m_z & m_r

                            gen_pt = rec["gen_pt"].astype(np.float32)[m]
                            dpt_rel = rec["dpt_rel"].astype(np.float32)[m]
                            dr = rec["dr"].astype(np.float32)[m]

                            dpt_by_bin = [[] for _ in range(len(pt_bins_r) - 1)]
                            dr_by_bin = [[] for _ in range(len(pt_bins_r) - 1)]

                            b = np.digitize(gen_pt, pt_bins_r) - 1
                            for ii in range(len(b)):
                                bi = int(b[ii])
                                if 0 <= bi < len(dpt_by_bin):
                                    dpt_by_bin[bi].append(float(dpt_rel[ii]))
                                    dr_by_bin[bi].append(float(dr[ii]))

                            key = f"{inp} · {algo}"
                            plot_ridgeline(
                                [np.asarray(v, dtype=float) for v in dpt_by_bin],
                                pt_bins_r,
                                xlabel=r"$(p_T^{RECO}-p_T^{GEN})/p_T^{GEN}$",
                                outputfile=os.path.join(out_ridge, f"dptrel__{sanitize(key)}.png"),
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                title=f"{plabel} — {region_name} — {zcat_name}\n{key}",
                                xlim=(-1.0, 1.0),
                                title_anchor=(0.96, 0.98),
                                title_ha="right",
                            )
                            plot_ridgeline(
                                [np.asarray(v, dtype=float) for v in dr_by_bin],
                                pt_bins_r,
                                xlabel=r"$\Delta R(\mathrm{reco},\mathrm{gen})$",
                                outputfile=os.path.join(out_ridge, f"dr__{sanitize(key)}.png"),
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                title=f"{plabel} — {region_name} — {zcat_name}\n{key}",
                                xlim=(0.0, 0.4),
                                title_anchor=(0.96, 0.98),
                                title_ha="right",
                            )

                # ---------------- Violin ----------------
                if cfg.STUDIES.get("violin_response", False):
                    pt_bins_v = cfg.PT_BINS["violin"]
                    out_vio = os.path.join(out_reg, "violin")
                    ensure_dir(out_vio)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m = m_z & m_r

                            gen_pt = rec["gen_pt"].astype(np.float32)[m]
                            resp = rec["resp"].astype(np.float32)[m]

                            resp_by_bin = [[] for _ in range(len(pt_bins_v) - 1)]
                            b = np.digitize(gen_pt, pt_bins_v) - 1
                            for ii in range(len(b)):
                                bi = int(b[ii])
                                if 0 <= bi < len(resp_by_bin):
                                    resp_by_bin[bi].append(float(resp[ii]))

                            key = f"{inp} · {algo}"
                            z_for_title = "" if zcat_name == "Inclusive" else f" — {zcat_name}"
                            plot_violin_response(
                                [np.asarray(v, dtype=float) for v in resp_by_bin],
                                pt_bins_v,
                                outputfile=os.path.join(out_vio, f"resp__{sanitize(key)}.png"),
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                title=f"{plabel} — {region_label}{z_for_title}\n{key}",
                                ylim=(0.5, 3.0)
                            )


                # -- Purity / Fake vs pT^RECO
                if cfg.STUDIES.get("purity_vs_ptreco", False) or cfg.STUDIES.get("fake_rate_vs_ptreco", False):
                    out_p = os.path.join(out_reg, "purity_fake")
                    ensure_dir(out_p)
                    pt_bins_p = np.asarray(cfg.PT_BINS.get("purity", np.array([0,10,15,20,25,30,40,60,80,100,150,200,300])), dtype=float)
                    pt_cent = 0.5 * (pt_bins_p[:-1] + pt_bins_p[1:])

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            recm = load_recomatch(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(recm["dz_cat"].astype(np.int32))
                            m_r = region_fn(recm["reco_eta"].astype(np.float32))
                            m = m_z & m_r

                            pt = recm["reco_pt"].astype(np.float32)[m]
                            is_matched = recm["is_matched"].astype(np.int32)[m]

                            den = np.histogram(pt, bins=pt_bins_p)[0].astype(float)
                            num = np.histogram(pt[is_matched == 1], bins=pt_bins_p)[0].astype(float)

                            purity, perr = _eff_and_err(num, den)
                            fake = 1.0 - purity
                            ferr = perr

                            title = f"{plabel} — {region_name} — {zcat_name}\n{inp} · {algo}"

                            if cfg.STUDIES.get("purity_vs_ptreco", False):
                                plot_purity_or_fake_vs_ptreco(
                                    pt_cent, purity, perr,
                                    outputfile=os.path.join(out_p, f"purity__{sanitize(inp+'_'+algo)}.png"),
                                    ylabel="Purity (RECO→GEN match fraction)",
                                    title=title,
                                    llabel=cfg.PLOT_LABELS["llabel"],
                                    rlabel=cfg.PLOT_LABELS["rlabel"],
                                )
                            if cfg.STUDIES.get("fake_rate_vs_ptreco", False):
                                plot_purity_or_fake_vs_ptreco(
                                    pt_cent, fake, ferr,
                                    outputfile=os.path.join(out_p, f"fake__{sanitize(inp+'_'+algo)}.png"),
                                    ylabel="Fake rate (1 − purity)",
                                    title=title,
                                    llabel=cfg.PLOT_LABELS["llabel"],
                                    rlabel=cfg.PLOT_LABELS["rlabel"],
                                )

                # -- Event-level N(jets), HT, seeds
                if (cfg.STUDIES.get("njet_multiplicity", False)
                        or cfg.STUDIES.get("ht_distributions", False)
                        or cfg.STUDIES.get("seed_stats", False)):
                    out_evt = os.path.join(out_reg, "event_level")
                    ensure_dir(out_evt)

                    jet_thresholds = np.asarray(cfg.PT_BINS.get("jet_thresholds", np.array([20,30,40,50], dtype=float)))
                    ht_thresholds  = np.asarray(cfg.PT_BINS.get("ht_thresholds",  np.array([20,30,40,50], dtype=float)))

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            em = load_event_metrics(cache_dir, inp, algo)
                            m_z_evt = zcat_mask_fn(em["dz_cat"].astype(np.int32))
                            m_evt = m_z_evt

                            title = f"{plabel} — {region_name} — {zcat_name}\n{inp} · {algo}"

                            if cfg.STUDIES.get("seed_stats", False):
                                nseeds = em["nseeds"].astype(np.int32)[m_evt]
                                plot_multiplicity_hist(
                                    nseeds,
                                    outputfile=os.path.join(out_evt, f"nseeds__{sanitize(inp+'_'+algo)}.png"),
                                    xlabel="N(seeds) per event",
                                    title=title,
                                    llabel=cfg.PLOT_LABELS["llabel"],
                                    rlabel=cfg.PLOT_LABELS["rlabel"],
                                    max_n=60
                                )

                            if cfg.STUDIES.get("njet_multiplicity", False):
                                for T in jet_thresholds:
                                    key = f"njet_ge_{int(T)}"
                                    if key not in em.files:
                                        continue
                                    njet = em[key].astype(np.int32)[m_evt]
                                    plot_multiplicity_hist(
                                        njet,
                                        outputfile=os.path.join(out_evt, f"njet_ge{int(T)}__{sanitize(inp+'_'+algo)}.png"),
                                        xlabel=fr"$N_\mathrm{{jets}}(p_T>{int(T)}\ \mathrm{{GeV}})$ per event",
                                        title=title,
                                        llabel=cfg.PLOT_LABELS["llabel"],
                                        rlabel=cfg.PLOT_LABELS["rlabel"],
                                        max_n=25
                                    )

                            if cfg.STUDIES.get("ht_distributions", False):
                                for T in ht_thresholds:
                                    key = f"ht_ge_{int(T)}"
                                    if key not in em.files:
                                        continue
                                    ht = em[key].astype(np.float32)[m_evt]
                                    plot_ht_hist(
                                        ht,
                                        outputfile=os.path.join(out_evt, f"ht_ge{int(T)}__{sanitize(inp+'_'+algo)}.png"),
                                        xlabel=fr"$H_T=\sum p_T^\mathrm{{jet}}$ for jets with $p_T>{int(T)}$ GeV",
                                        title=title,
                                        llabel=cfg.PLOT_LABELS["llabel"],
                                        rlabel=cfg.PLOT_LABELS["rlabel"],
                                    )

                # -- Response quantiles vs pT^GEN
                if cfg.STUDIES.get("response_quantiles", False):
                    out_q = os.path.join(out_reg, "response_quantiles")
                    ensure_dir(out_q)

                    pt_bins_s = np.asarray(cfg.PT_BINS.get("summary_gen", cfg.PT_BINS["efficiency"]), dtype=float)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m = m_z & m_r

                            gen_pt = rec["gen_pt"].astype(np.float32)[m]
                            resp   = rec["resp"].astype(np.float32)[m]

                            pt_cent, (q16, q50, q84) = _bin_quantiles(gen_pt, resp, pt_bins_s, q=(16, 50, 84))

                            title = f"{plabel} — {region_name} — {zcat_name}\n{inp} · {algo}"
                            plot_quantiles_vs_ptgen(
                                pt_cent, q16, q50, q84,
                                outputfile=os.path.join(out_q, f"resp_q16_50_84__{sanitize(inp+'_'+algo)}.png"),
                                ylabel=r"$p_T^{RECO}/p_T^{GEN}$ (16–84% band)",
                                title=title,
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                ylim=(0.0, 3.0)
                            )

                # -- dR quantiles vs pT^GEN
                if cfg.STUDIES.get("dr_quantiles", False):
                    out_dr = os.path.join(out_reg, "dr_quantiles")
                    ensure_dir(out_dr)

                    pt_bins_s = np.asarray(cfg.PT_BINS.get("summary_gen", cfg.PT_BINS["efficiency"]), dtype=float)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)

                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m = m_z & m_r

                            gen_pt = rec["gen_pt"].astype(np.float32)[m]
                            dr     = rec["dr"].astype(np.float32)[m]

                            pt_cent, (q16, q50, q84) = _bin_quantiles(gen_pt, dr, pt_bins_s, q=(16, 50, 84))

                            title = f"{plabel} — {region_name} — {zcat_name}\n{inp} · {algo}"
                            plot_quantiles_vs_ptgen(
                                pt_cent, q16, q50, q84,
                                outputfile=os.path.join(out_dr, f"dr_q16_50_84__{sanitize(inp+'_'+algo)}.png"),
                                ylabel=r"$\Delta R(\mathrm{reco},\mathrm{gen})$ (16–84% band)",
                                title=title,
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                ylim=(0.0, float(cfg.MATCHING.get("dR_match", 0.3)) * 1.2)
                            )

                # -- Turn-ons: P(pT^RECO > thr | GEN jet in bin)
                if cfg.STUDIES.get("turnons", False):
                    out_t = os.path.join(out_reg, "turnons")
                    ensure_dir(out_t)

                    pt_bins_s = np.asarray(cfg.PT_BINS.get("summary_gen", cfg.PT_BINS["efficiency"]), dtype=float)
                    pt_cent = 0.5 * (pt_bins_s[:-1] + pt_bins_s[1:])
                    thresholds = np.asarray(cfg.PT_BINS.get("turnon_thresholds", np.array([30, 40, 50], dtype=float)), dtype=float)

                    total = np.histogram(gen_pt_denom, bins=pt_bins_s)[0].astype(float)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)
                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m = m_z & m_r

                            gpt = rec["gen_pt"].astype(np.float32)[m]
                            rpt = rec["reco_pt"].astype(np.float32)[m]

                            curves = {}
                            for thr in thresholds:
                                sel = (rpt > float(thr))
                                num = np.histogram(gpt[sel], bins=pt_bins_s)[0].astype(float)
                                eff, err = _eff_and_err(num, total)
                                curves[float(thr)] = (eff, err)

                            title = f"{plabel} — {region_name} — {zcat_name}\n{inp} · {algo}"
                            plot_turnon_curves(
                                pt_cent, curves,
                                outputfile=os.path.join(out_t, f"turnon__{sanitize(inp+'_'+algo)}.png"),
                                title=title,
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                            )


                if cfg.STUDIES.get("f1", False):
                    out_f1 = os.path.join(out_reg, "f1")
                    ensure_dir(out_f1)

                    # Truth jets in this slice: GEN jets with pT_GEN >= T
                    truth_counts = np.array([np.sum(gen_pt_denom >= float(T)) for T in f1_thresholds], dtype=float)

                    for inp in enabled_inputs:
                        for algo in enabled_algos:
                            rec = load_matches(cache_dir, inp, algo)
                            rm = load_recomatch(cache_dir, inp, algo)

                            # Slice match cache by dz + region (region defined on GEN eta)
                            m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                            m_r = region_fn(rec["gen_eta"].astype(np.float32))
                            m_mask = m_z & m_r

                            m_gen_pt = rec["gen_pt"].astype(np.float32)[m_mask]
                            m_reco_pt = rec["reco_pt"].astype(np.float32)[m_mask]

                            # Slice recomatch by dz and by GEN eta of matched gen jet.
                            # Note: unmatched reco entries have dummy gen_eta/gen_pt; they are excluded by region_fn(gen_eta)
                            rm_z = zcat_mask_fn(rm["dz_cat"].astype(np.int32))
                            rm_r = region_fn(rm["gen_eta"].astype(np.float32))
                            rm_mask = rm_z & rm_r

                            reco_pt = rm["reco_pt"].astype(np.float32)[rm_mask]
                            reco_matched_genpt = rm["gen_pt"].astype(np.float32)[rm_mask]

                            # TP(T): matched pairs with both gen_pt>=T and reco_pt>=T
                            TP = np.array(
                                [np.sum((m_gen_pt >= float(T)) & (m_reco_pt >= float(T))) for T in f1_thresholds],
                                dtype=float
                            )

                            # Predicted jets above T in this slice (as defined by rm_mask)
                            Npred = np.array([np.sum(reco_pt >= float(T)) for T in f1_thresholds], dtype=float)

                            # Of those predicted jets above T, count those matched to a truth jet above T
                            TPpred = np.array(
                                [np.sum((reco_pt >= float(T)) & (reco_matched_genpt >= float(T))) for T in f1_thresholds],
                                dtype=float
                            )

                            FN = np.maximum(truth_counts - TP, 0.0)
                            FP = np.maximum(Npred - TPpred, 0.0)

                            precision = np.divide(TPpred, TPpred + FP, out=np.zeros_like(TPpred), where=((TPpred + FP) > 0))
                            recall = np.divide(TP, TP + FN, out=np.zeros_like(TP), where=((TP + FN) > 0))
                            f1 = np.divide(
                                2.0 * precision * recall,
                                precision + recall,
                                out=np.zeros_like(precision),
                                where=((precision + recall) > 0)
                            )

                            title = _title_lines(plabel, region_name, zcat_name, extra=f"{inp} · {algo}")

                            plot_f1_vs_threshold(
                                thresholds=f1_thresholds,
                                f1=f1,
                                precision=precision,
                                recall=recall,
                                outputfile=os.path.join(out_f1, f"f1__{sanitize(inp)}__{sanitize(algo)}.png"),
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                title=title
                            )

                if cfg.STUDIES.get("ak_compat", False):
                    # ---- config
                    pt_bins = np.asarray(cfg.AK_COMPAT["pt_bins"], dtype=float)   # interpreted as GEN pT bins here
                    ref_alg = cfg.AK_COMPAT["ref_algo"]

                    out_ak = os.path.join(out_reg, "akcompat")
                    ensure_dir(out_ak)

                    # ---- (1) IoU overlays per GEN pT bin (weighted + unweighted)
                    metrics_to_plot = [
                        ("iou", "IoU (pT-weighted)"),
                        ("iou_unw", "IoU (unweighted)"),
                    ]

                    # ---- (2) Ratio distributions per GEN pT bin
                    ratio_metrics = [
                        ("ratio_n",  r"$N_\mathrm{const}(\mathrm{AK})/N_\mathrm{const}(\mathrm{Alt})$"),
                        ("ratio_pt", r"$\sum p_T^\mathrm{const}(\mathrm{AK})/\sum p_T^\mathrm{const}(\mathrm{Alt})$"),
                    ]

                    for inp in enabled_inputs:
                        # -------------------------
                        # A) IoU distributions in GEN pT bins
                        # -------------------------
                        for metric_key, metric_label in metrics_to_plot:
                            values_by_ptbin_by_algo = []

                            for i in range(len(pt_bins) - 1):
                                lo, hi = pt_bins[i], pt_bins[i + 1]
                                values_by_algo = {}

                                for algo in enabled_algos:
                                    if algo == ref_alg:
                                        continue

                                    rec = load_akcompat_gen(cache_dir, inp, algo)
                                    if rec is None:
                                        continue
                                    if metric_key not in rec.files:
                                        continue

                                    # dz + region cut (region on GEN eta in this GEN-anchored cache)
                                    m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                                    m_r = region_fn(rec["gen_eta"].astype(np.float32))
                                    m = m_z & m_r

                                    gpt = rec["gen_pt"].astype(np.float32)
                                    sel = (gpt >= lo) & (gpt < hi) & m

                                    vals = rec[metric_key][sel].astype(np.float32)
                                    values_by_algo[algo] = vals[np.isfinite(vals)]

                                values_by_ptbin_by_algo.append(values_by_algo)

                            plot_akcompat_overlay_hist_subplots(
                                values_by_ptbin_by_algo=values_by_ptbin_by_algo,
                                pt_bins=pt_bins,
                                outputfile=os.path.join(
                                    out_ak,
                                    f"akcompat_gen_subplots__{sanitize(inp)}__{sanitize(metric_key)}.png"
                                ),
                                xlabel=metric_label,
                                title=title_reco_reco(plabel, inp, ref_alg) + "\n(GEN-matched to both)",
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                bins=np.linspace(0, 1, 25),
                                xlim=(0, 1),
                            )

                        # -------------------------
                        # B) Ratio distributions in GEN pT bins
                        # -------------------------
                        for metric_key, metric_label in ratio_metrics:
                            values_by_ptbin_by_algo = []

                            for i in range(len(pt_bins) - 1):
                                lo, hi = pt_bins[i], pt_bins[i + 1]
                                values_by_algo = {}

                                for algo in enabled_algos:
                                    if algo == ref_alg:
                                        continue

                                    rec = load_akcompat_gen(cache_dir, inp, algo)
                                    if rec is None:
                                        continue
                                    if metric_key not in rec.files:
                                        continue

                                    m_z = zcat_mask_fn(rec["dz_cat"].astype(np.int32))
                                    m_r = region_fn(rec["gen_eta"].astype(np.float32))
                                    m = m_z & m_r

                                    gpt = rec["gen_pt"].astype(np.float32)
                                    sel = (gpt >= lo) & (gpt < hi) & m

                                    vals = rec[metric_key][sel].astype(np.float32)
                                    vals = vals[np.isfinite(vals)]
                                    # keep it sane; ratios can have long tails if alt jet is tiny
                                    vals = vals[(vals > 0) & (vals < 10)]
                                    values_by_algo[algo] = vals

                                values_by_ptbin_by_algo.append(values_by_algo)

                            # Choose x-range and bins that work for "AK/Alt"
                            # (centered at 1, with tails visible)
                            plot_akcompat_overlay_hist_subplots(
                                values_by_ptbin_by_algo=values_by_ptbin_by_algo,
                                pt_bins=pt_bins,
                                outputfile=os.path.join(
                                    out_ak,
                                    f"akcompat_gen_subplots__{sanitize(inp)}__{sanitize(metric_key)}.png"
                                ),
                                xlabel=metric_label,
                                title=title_reco_reco(plabel, inp, ref_alg) + "\n(GEN-matched to both)",
                                llabel=cfg.PLOT_LABELS["llabel"],
                                rlabel=cfg.PLOT_LABELS["rlabel"],
                                bins=np.linspace(0, 3, 31),
                                xlim=(0, 3),
                            )

        # ---------------- Agreement ----------------
        if cfg.STUDIES.get("ak4_agreement", False):
            out_ag = os.path.join(out_proc, "agreement_with_AK4")
            ensure_dir(out_ag)

            for inp in enabled_inputs:
                for algo in enabled_algos:
                    f = os.path.join(cache_dir, f"agreement__{sanitize(inp)}__{sanitize(algo)}.npz")
                    if not os.path.exists(f):
                        continue

                    fracs = np.load(f)["fractions"]
                    title = f"{plabel}\n{inp} · {algo} vs PF · AntiKt"
                    plot_agreement_fraction_hist(
                        fracs,
                        outputfile=os.path.join(out_ag, f"agreement__{sanitize(inp+'_'+algo)}.png"),
                        llabel=cfg.PLOT_LABELS["llabel"],
                        rlabel=cfg.PLOT_LABELS["rlabel"],
                        title=title
                    )

    print("All plots done.")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg_from_path(args.config)
    tag = config_tag_from_path(args.config)
    run(cfg, tag)
