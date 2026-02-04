# run_processing.py
import os
import argparse
import importlib.util
import numpy as np
import awkward as ak

from src.utils import (
    load_arrays, ensure_dir, match_gen_to_reco, match_reco_to_gen,
    match_reco_to_reco, pt_weighted_constituent_overlap,
    unweighted_constituent_overlap, save_columnar_npz,
    jet_constituent_count, jet_constituent_sumpt, safe_ratio,
)

from src.clustering_algorithms import (
    ALGO_REGISTRY, wrap_phi
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Config loading
# -----------------------------
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
    ap = argparse.ArgumentParser(description="Run processing and write caches for plotting.")
    ap.add_argument("--config", "-c", default="config.py",
                    help="Path to config file, e.g. configs/example_config.py (default: config.py)")
    return ap.parse_args()


def maybe_tqdm(cfg, it, total=None, desc=None):
    if cfg.RUNTIME.get("use_tqdm", True) and (tqdm is not None):
        return tqdm(it, total=total, desc=desc)
    return it


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


def select_event_indices(cfg, n_total: int) -> np.ndarray:
    max_events = cfg.RUNTIME.get("max_events", None)
    sampling = cfg.RUNTIME.get("event_sampling", "head")
    stride = int(cfg.RUNTIME.get("stride", 1))

    if (max_events is None) or (max_events <= 0) or (max_events >= n_total):
        return np.arange(n_total, dtype=int)

    if sampling == "stride":
        idx = np.arange(0, n_total, stride, dtype=int)
        return idx[:max_events]

    return np.arange(int(max_events), dtype=int)


def branch_list(cfg):
    bl = []
    for _, cmap in cfg.BRANCHES["cands"].items():
        for _, br in cmap.items():
            bl.append(br)
    for _, br in cfg.BRANCHES["genjets"].items():
        bl.append(br)
    vtx = cfg.BRANCHES.get("vtx", {})
    for k in ("z_gen", "z_reco", "reco_sumpt", "n_reco"):
        if vtx.get(k):
            bl.append(vtx[k])
    return sorted(set(bl))


def scalar_item(x, default=None):
    if x is None:
        return default
    if isinstance(x, ak.highlevel.Array):
        x = ak.to_numpy(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        if x.size == 0:
            return default
        return x.reshape(-1)[0].item()
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return default
        return scalar_item(x[0], default=default)
    return x


def compute_event_dz_cat(cfg, data, ievt: int) -> int:
    if not cfg.Z_SPLIT.get("enabled", False):
        return 0

    vtx = cfg.BRANCHES.get("vtx", {})
    zgen_br = vtx.get("z_gen")
    zreco_br = vtx.get("z_reco")
    sumpt_br = vtx.get("reco_sumpt")
    nreco_br = vtx.get("n_reco")

    if any(x is None for x in (zgen_br, zreco_br, sumpt_br, nreco_br)):
        return -1

    nreco = scalar_item(data[nreco_br][ievt], default=0)
    if int(nreco) <= 0:
        return -1

    sp = ak.to_numpy(data[sumpt_br][ievt])
    zz = ak.to_numpy(data[zreco_br][ievt])
    if sp.size == 0 or zz.size == 0:
        return -1

    pv_idx = int(np.argmax(sp))
    zreco = float(zz[pv_idx])

    zgen = scalar_item(data[zgen_br][ievt], default=None)
    if zgen is None:
        return -1

    dz = abs(float(zgen) - zreco)
    thr = float(cfg.Z_SPLIT.get("dz_cm", 1.0))
    return 0 if dz < thr else 1


def save_matches_npz(outpath, records):
    keys = [
        "event", "gen_idx", "reco_idx",
        "gen_pt", "gen_eta", "gen_phi",
        "reco_pt", "reco_eta", "reco_phi",
        "dr", "resp", "dpt_rel",
        "dz_cat",
    ]
    if len(records) == 0:
        np.savez_compressed(
            outpath,
            **{k: np.array([], dtype=(np.int32 if k in ("event", "gen_idx", "reco_idx", "dz_cat") else np.float32))
               for k in keys}
        )
        return

    cols = {k: [] for k in keys}
    for r in records:
        for k in keys:
            cols[k].append(r[k])

    out = {}
    for k, v in cols.items():
        if k in ("event", "gen_idx", "reco_idx", "dz_cat"):
            out[k] = np.asarray(v, dtype=np.int32)
        else:
            out[k] = np.asarray(v, dtype=np.float32)

    np.savez_compressed(outpath, **out)

def remap_assign_after_jet_pt_cut(assign, keep_mask):
    """
    Remap per-constituent jet assignment indices after removing jets.

    assign: int array of length Ncands, values in [-1, 0..Njets-1] (OLD jet indices)
    keep_mask: bool array of length Njets (OLD jet list), True for jets we keep

    Returns:
      new_assign: int array length Ncands, values in [-1, 0..Nkept-1]
      old_to_new: int array length Njets, mapping old jet idx -> new jet idx (or -1)
    """
    assign = np.asarray(assign, dtype=int)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    n_old = int(keep_mask.size)
    old_to_new = np.full(n_old, -1, dtype=int)
    kept_old = np.where(keep_mask)[0]
    for new_i, old_i in enumerate(kept_old):
        old_to_new[int(old_i)] = int(new_i)

    new_assign = np.full_like(assign, -1)
    ok = (assign >= 0) & (assign < n_old)
    if np.any(ok):
        mapped = old_to_new[assign[ok]]
        new_assign[ok] = mapped  # mapped is -1 for removed jets
    return new_assign, old_to_new


# -----------------------------
# Main
# -----------------------------
def run(cfg, cfg_tag: str):
    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    ensure_dir(out_root)

    enabled_inputs = [k for k, v in cfg.INPUTS.items() if v]
    enabled_algos = [(name, a) for name, a in cfg.ALGORITHMS.items() if a.get("enabled", False)]

    # --- AK-compat config
    do_akcompat = cfg.STUDIES.get("ak_compat", False)
    ak_ref_algo = cfg.AK_COMPAT.get("ref_algo", "AntiKt")
    ak_dR = float(cfg.AK_COMPAT.get("dR_match", 0.2))

    for proc, pinfo in cfg.PROCESSES.items():
        path = pinfo["path"]
        print(f"\n=== PROCESS: {proc} | file: {path} | config: {cfg_tag} ===")

        data = load_arrays(path, cfg.TREE_NAME, branch_list(cfg), library="ak")
        n_total = len(data[cfg.BRANCHES["genjets"]["pt"]])
        ev_idx = select_event_indices(cfg, n_total)
        print(f"Loaded {n_total} events (processing {len(ev_idx)})")

        out_proc = os.path.join(out_root, proc)
        out_cache = os.path.join(out_proc, "cache")
        ensure_dir(out_cache)

        gen_pt_all  = data[cfg.BRANCHES["genjets"]["pt"]]
        gen_eta_all = data[cfg.BRANCHES["genjets"]["eta"]]
        gen_phi_all = data[cfg.BRANCHES["genjets"]["phi"]]

        # containers-
        denom_gen_pt = []
        denom_gen_eta = []
        denom_dz_cat = []

        reco_denom = {
            (inp, aname): {"event": [], "reco_idx": [], "reco_pt": [], "reco_eta": [], "reco_phi": [], "dz_cat": []}
            for inp in enabled_inputs for aname, _ in enabled_algos
        }

        evt_metrics = {
            (inp, aname): {"event": [], "dz_cat": [],
                           "njet_ge_T": {}, "ht_ge_T": {},
                           "nseeds": []}
            for inp in enabled_inputs for aname, _ in enabled_algos
        }

        jet_thresholds = np.asarray(cfg.PT_BINS.get("jet_thresholds", np.array([20, 30, 40, 50], dtype=float)), dtype=float)
        ht_thresholds  = np.asarray(cfg.PT_BINS.get("ht_thresholds",  np.array([20, 30, 40, 50], dtype=float)), dtype=float)

        for T in jet_thresholds:
            for key in evt_metrics.keys():
                evt_metrics[key]["njet_ge_T"][float(T)] = []
        for T in ht_thresholds:
            for key in evt_metrics.keys():
                evt_metrics[key]["ht_ge_T"][float(T)] = []

        match_records = {(inp, aname): [] for inp in enabled_inputs for aname, _ in enabled_algos}
        reco_match_records = {(inp, aname): [] for inp in enabled_inputs for aname, _ in enabled_algos}
        unmatched_counts = {
            (inp, aname): {"event": [], "dz_cat": [], "nunmatched_gen": [], "nunmatched_reco": []}
            for inp in enabled_inputs for aname, _ in enabled_algos
        }

        agreement = {}
        do_agree = cfg.STUDIES.get("ak4_agreement", False)
        have_pf = ("PF" in enabled_inputs)
        have_antikt = any(an == "AntiKt" for an, _ in enabled_algos)
        if do_agree and not (have_pf and have_antikt):
            do_agree = False

        # --- AK-compat records (additive)
        akcompat_records = {
            (inp, aname): []
            for inp in enabled_inputs
            for aname, _ in enabled_algos
            if aname != ak_ref_algo
        }

        akcompat_gen_records = {
            (inp, aname): []
            for inp in enabled_inputs
            for aname, _ in enabled_algos
            if aname != ak_ref_algo
        }

        akmatch_ref_records = {
            (inp, aname): []
            for inp in enabled_inputs
            for aname, _ in enabled_algos
            if aname != ak_ref_algo
        }
        akmatch_alt_records = {
            (inp, aname): []
            for inp in enabled_inputs
            for aname, _ in enabled_algos
            if aname != ak_ref_algo
        }

        algo_fns = {aname: ALGO_REGISTRY[acfg["fn"]] for aname, acfg in enabled_algos}
        algo_params = {aname: acfg.get("params", {}) for aname, acfg in enabled_algos}

        cand_arrays = {}
        for inp in enabled_inputs:
            cdef = cfg.BRANCHES["cands"][inp]
            cand_arrays[inp] = {
                "pt":  data[cdef["pt"]],
                "eta": data[cdef["eta"]],
                "phi": data[cdef["phi"]],
            }

        # -----------------------------
        # Event loop
        # -----------------------------
        for ievt in maybe_tqdm(cfg, ev_idx, total=len(ev_idx), desc=f"{proc}: events"):
            ievt = int(ievt)
            dz_cat_evt = compute_event_dz_cat(cfg, data, ievt)

            gen_pt  = ak.to_numpy(gen_pt_all[ievt])
            gen_eta = ak.to_numpy(gen_eta_all[ievt])
            gen_phi = wrap_phi(ak.to_numpy(gen_phi_all[ievt]))

            gsel = (gen_pt >= float(cfg.MATCHING["pt_gen_min"]))
            gen_pt, gen_eta, gen_phi = gen_pt[gsel], gen_eta[gsel], gen_phi[gsel]
            if gen_pt.size == 0:
                continue

            denom_gen_pt.append(gen_pt.astype(np.float32))
            denom_gen_eta.append(gen_eta.astype(np.float32))
            denom_dz_cat.append(np.full(gen_pt.shape, dz_cat_evt, dtype=np.int32))

            # cluster
            reco_by_key = {}      # (inp, aname) -> (pt, eta, phi)
            assign_by_key = {}    # (inp, aname) -> assign
            seedmask_by_key = {}

            for inp in enabled_inputs:
                cpt  = ak.to_numpy(cand_arrays[inp]["pt"][ievt])
                ceta = ak.to_numpy(cand_arrays[inp]["eta"][ievt])
                cphi = wrap_phi(ak.to_numpy(cand_arrays[inp]["phi"][ievt]))

                for aname, _ in enabled_algos:
                    jets, assign, seed_mask = algo_fns[aname](ceta, cphi, cpt, **algo_params[aname])
                    rpt_all, reta_all, rphi_all, _ = jets

                    rpt_all = np.asarray(rpt_all, dtype=float)
                    reta_all = np.asarray(reta_all, dtype=float)
                    rphi_all = np.asarray(rphi_all, dtype=float)

                    # apply jet pT cut on the jet list
                    rsel = (rpt_all >= float(cfg.MATCHING["pt_reco_min"]))

                    rpt = rpt_all[rsel]
                    reta = reta_all[rsel]
                    rphi = rphi_all[rsel]

                    # remap assign old->new indices after cutting jets
                    assign_new, _old_to_new = remap_assign_after_jet_pt_cut(assign, rsel)

                    reco_by_key[(inp, aname)] = (rpt, reta, rphi)
                    assign_by_key[(inp, aname)] = np.asarray(assign_new, dtype=int)

                    seedmask_by_key[(inp, aname)] = (
                        np.asarray(seed_mask, dtype=bool)
                        if seed_mask is not None else np.array([], dtype=bool)
                    )


                    # reco denominators
                    for ir in range(len(rpt)):
                        reco_denom[(inp, aname)]["event"].append(ievt)
                        reco_denom[(inp, aname)]["reco_idx"].append(ir)
                        reco_denom[(inp, aname)]["reco_pt"].append(float(rpt[ir]))
                        reco_denom[(inp, aname)]["reco_eta"].append(float(reta[ir]))
                        reco_denom[(inp, aname)]["reco_phi"].append(float(rphi[ir]))
                        reco_denom[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))

                    # event metrics
                    nseeds = int(np.sum(seedmask_by_key[(inp, aname)])) if seedmask_by_key[(inp, aname)].size else 0
                    evt_metrics[(inp, aname)]["event"].append(ievt)
                    evt_metrics[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))
                    evt_metrics[(inp, aname)]["nseeds"].append(nseeds)

                    for T in jet_thresholds:
                        evt_metrics[(inp, aname)]["njet_ge_T"][float(T)].append(int(np.sum(rpt >= float(T))))
                    for T in ht_thresholds:
                        evt_metrics[(inp, aname)]["ht_ge_T"][float(T)].append(float(np.sum(rpt[rpt >= float(T)])))

            # ------------------------------------------------------------
            # per-event GEN->RECO maps for (inp, aname)
            # ------------------------------------------------------------
            gen2reco_evt = {}  # (inp, aname) -> dict(gen_idx -> reco_idx)

            # GEN<->RECO matching + fill gen2reco_evt
            for (inp, aname), (rpt, reta, rphi) in reco_by_key.items():
                matched, un_g, un_r = match_gen_to_reco(
                    gen_pt, gen_eta, gen_phi,
                    rpt, reta, rphi,
                    dR=float(cfg.MATCHING["dR_match"]),
                    pt_gen_min=float(cfg.MATCHING["pt_gen_min"]),
                    pt_reco_min=float(cfg.MATCHING["pt_reco_min"]),
                )

                gen2reco_evt[(inp, aname)] = {
                    int(m["gen_idx"]): int(m["reco_idx"])
                    for m in matched
                }

                unmatched_counts[(inp, aname)]["event"].append(ievt)
                unmatched_counts[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))
                unmatched_counts[(inp, aname)]["nunmatched_gen"].append(int(len(un_g)))
                unmatched_counts[(inp, aname)]["nunmatched_reco"].append(int(len(un_r)))

                for m in matched:
                    match_records[(inp, aname)].append({
                        "event": ievt,
                        "gen_idx": int(m["gen_idx"]),
                        "reco_idx": int(m["reco_idx"]),
                        "gen_pt": float(m["gen_pt"]),
                        "gen_eta": float(m["gen_eta"]),
                        "gen_phi": float(m["gen_phi"]),
                        "reco_pt": float(m["reco_pt"]),
                        "reco_eta": float(m["reco_eta"]),
                        "reco_phi": float(m["reco_phi"]),
                        "dr": float(m["dr"]),
                        "resp": float(m["resp"]),
                        "dpt_rel": float(m["dpt_rel"]),
                        "dz_cat": int(dz_cat_evt),
                    })

                # reco→gen matching
                m_r2g, _, _ = match_reco_to_gen(
                    rpt, reta, rphi,
                    gen_pt, gen_eta, gen_phi,
                    dR=float(cfg.MATCHING["dR_match"]),
                    pt_reco_min=float(cfg.MATCHING["pt_reco_min"]),
                    pt_gen_min=float(cfg.MATCHING["pt_gen_min"]),
                )

                reco_to_gen = {int(mm["reco_idx"]): mm for mm in m_r2g}

                for ir in range(len(rpt)):
                    if ir in reco_to_gen:
                        mm = reco_to_gen[ir]
                        is_matched = 1
                        gen_pt_m = float(mm["gen_pt"])
                        gen_eta_m = float(mm["gen_eta"])
                        gen_phi_m = float(mm["gen_phi"])
                        dr_m = float(mm["dr"])
                    else:
                        is_matched = 0
                        gen_pt_m = -1.0
                        gen_eta_m = 0.0
                        gen_phi_m = 0.0
                        dr_m = -1.0

                    reco_match_records[(inp, aname)].append({
                        "event": ievt,
                        "reco_idx": ir,
                        "reco_pt": float(rpt[ir]),
                        "reco_eta": float(reta[ir]),
                        "reco_phi": float(rphi[ir]),
                        "is_matched": int(is_matched),
                        "gen_pt": float(gen_pt_m),
                        "gen_eta": float(gen_eta_m),
                        "gen_phi": float(gen_phi_m),
                        "dr": float(dr_m),
                        "dz_cat": int(dz_cat_evt),
                    })

            # -----------------------------
            # AK-compatibility block
            # -----------------------------
            if do_akcompat:
                for inp in enabled_inputs:
                    ref_key = (inp, ak_ref_algo)
                    if ref_key not in reco_by_key:
                        continue

                    ref_pt, ref_eta, ref_phi = reco_by_key[ref_key]
                    ref_assign = assign_by_key[ref_key]

                    # which REF reco jets are GEN-matched, using gen2reco_evt
                    ref_genmatched = set(gen2reco_evt.get(ref_key, {}).values())

                    # Candidate pT for overlap metrics (PF/PUPPI array)
                    cand_pt_evt = ak.to_numpy(cand_arrays[inp]["pt"][ievt])

                    for aname, _ in enabled_algos:
                        if aname == ak_ref_algo:
                            continue

                        alt_key = (inp, aname)
                        if alt_key not in reco_by_key:
                            continue

                        # -------------------------
                        # DEFINE alt_* EARLY
                        # -------------------------
                        alt_pt, alt_eta, alt_phi = reco_by_key[alt_key]
                        alt_assign = assign_by_key[alt_key]
                        # -------------------------

                        # ============================================================
                        # (A) GEN-driven comparison: match AK and ALT to SAME gen jet
                        # ============================================================
                        ref_map = gen2reco_evt.get(ref_key, {})   # gen_idx -> ref_reco_idx
                        alt_map = gen2reco_evt.get(alt_key, {})   # gen_idx -> alt_reco_idx

                        if ref_map and alt_map:
                            common_gen = sorted(set(ref_map.keys()) & set(alt_map.keys()))
                            for ig in common_gen:
                                ir = int(ref_map[ig])
                                ia = int(alt_map[ig])

                                # IoU (pT-weighted)
                                f_ref_w, f_alt_w, iou_w = pt_weighted_constituent_overlap(
                                    cand_pt_evt,
                                    ref_assign, ir,
                                    alt_assign, ia,
                                )

                                # IoU (unweighted)
                                f_ref_u, f_alt_u, iou_u = unweighted_constituent_overlap(
                                    ref_assign, ir,
                                    alt_assign, ia,
                                )

                                # ratios: Nconst(AK)/Nconst(Alt) and SumPt(AK)/SumPt(Alt)
                                n_ref = jet_constituent_count(ref_assign, ir)
                                n_alt = jet_constituent_count(alt_assign, ia)
                                sumpt_ref = jet_constituent_sumpt(cand_pt_evt, ref_assign, ir)
                                sumpt_alt = jet_constituent_sumpt(cand_pt_evt, alt_assign, ia)

                                akcompat_gen_records[(inp, aname)].append({
                                    "event": ievt,
                                    "dz_cat": int(dz_cat_evt),

                                    "gen_idx": int(ig),
                                    "gen_pt": float(gen_pt[ig]),
                                    "gen_eta": float(gen_eta[ig]),
                                    "gen_phi": float(gen_phi[ig]),

                                    "ref_pt": float(ref_pt[ir]) if ir < len(ref_pt) else -1.0,
                                    "alt_pt": float(alt_pt[ia]) if ia < len(alt_pt) else -1.0,

                                    # IoU + fractions
                                    "iou": float(iou_w),
                                    "f_ref": float(f_ref_w),
                                    "f_alt": float(f_alt_w),

                                    "iou_unw": float(iou_u),
                                    "f_ref_unw": float(f_ref_u),
                                    "f_alt_unw": float(f_alt_u),

                                    # ratios
                                    "ratio_n": float(safe_ratio(n_ref, n_alt, default=np.nan)),
                                    "ratio_pt": float(safe_ratio(sumpt_ref, sumpt_alt, default=np.nan)),
                                })

                        # ============================================================
                        # (B) RECO<->RECO matching: REF-driven greedy 1-1
                        # ============================================================
                        matches, un_ref, un_alt = match_reco_to_reco(
                            ref_pt, ref_eta, ref_phi,
                            alt_pt, alt_eta, alt_phi,
                            dR=ak_dR,
                            pt_ref_min=float(cfg.MATCHING["pt_reco_min"]),
                            pt_alt_min=float(cfg.MATCHING["pt_reco_min"]),
                        )

                        matched_ref_set = {int(m["ref_idx"]) for m in matches}
                        matched_alt_set = {int(m["alt_idx"]) for m in matches}

                        # Store AK->ALT matching efficiency inputs (per REF jet)
                        for ir in range(len(ref_pt)):
                            if float(ref_pt[ir]) < float(cfg.MATCHING["pt_reco_min"]):
                                continue
                            akmatch_ref_records[(inp, aname)].append({
                                "event": ievt,
                                "dz_cat": int(dz_cat_evt),
                                "ref_idx": int(ir),
                                "ref_pt": float(ref_pt[ir]),
                                "ref_eta": float(ref_eta[ir]),
                                "is_matched": int(ir in matched_ref_set),
                            })

                        # Store ALT fake-rate inputs (per ALT jet)
                        for ia in range(len(alt_pt)):
                            if float(alt_pt[ia]) < float(cfg.MATCHING["pt_reco_min"]):
                                continue
                            akmatch_alt_records[(inp, aname)].append({
                                "event": ievt,
                                "dz_cat": int(dz_cat_evt),
                                "alt_idx": int(ia),
                                "alt_pt": float(alt_pt[ia]),
                                "alt_eta": float(alt_eta[ia]),
                                "is_fake": int(ia not in matched_alt_set),
                            })

                        # Constituent overlap metrics for matched jet pairs only
                        for m in matches:
                            ir = int(m["ref_idx"])
                            ia = int(m["alt_idx"])

                            f_ref_w, f_alt_w, iou_w = pt_weighted_constituent_overlap(
                                cand_pt_evt,
                                ref_assign, ir,
                                alt_assign, ia,
                            )

                            f_ref_u, f_alt_u, iou_u = unweighted_constituent_overlap(
                                ref_assign, ir,
                                alt_assign, ia,
                            )

                            akcompat_records[(inp, aname)].append({
                                "event": ievt,
                                "dz_cat": int(dz_cat_evt),

                                "ref_pt": float(m["ref_pt"]),
                                "ref_eta": float(m["ref_eta"]),
                                "ref_genmatched": int(ir in ref_genmatched),
                                "dr_ref_alt": float(m["dr"]),

                                "iou": float(iou_w),
                                "f_ref": float(f_ref_w),
                                "f_alt": float(f_alt_w),

                                "iou_unw": float(iou_u),
                                "f_ref_unw": float(f_ref_u),
                                "f_alt_unw": float(f_alt_u),
                            })

                            

        # -----------------------------
        # Write caches 
        # -----------------------------
        denom_gen_pt = np.concatenate(denom_gen_pt) if denom_gen_pt else np.array([], dtype=np.float32)
        denom_gen_eta = np.concatenate(denom_gen_eta) if denom_gen_eta else np.array([], dtype=np.float32)
        denom_dz_cat = np.concatenate(denom_dz_cat) if denom_dz_cat else np.array([], dtype=np.int32)

        np.savez_compressed(
            os.path.join(out_cache, "denom_genjets.npz"),
            gen_pt=denom_gen_pt,
            gen_eta=denom_gen_eta,
            dz_cat=denom_dz_cat,
        )

        print("Writing cache files...")

        for (inp, aname), recs in match_records.items():
            out = os.path.join(out_cache, f"matches__{sanitize(inp)}__{sanitize(aname)}.npz")
            save_matches_npz(out, recs)

        for (inp, aname), cols in reco_denom.items():
            out = os.path.join(out_cache, f"denom_recojets__{sanitize(inp)}__{sanitize(aname)}.npz")
            save_columnar_npz(
                out, cols,
                dtypes={
                    "event": np.int32,
                    "reco_idx": np.int32,
                    "reco_pt": np.float32,
                    "reco_eta": np.float32,
                    "reco_phi": np.float32,
                    "dz_cat": np.int32,
                }
            )

        for (inp, aname), recs in reco_match_records.items():
            out = os.path.join(out_cache, f"recomatch__{sanitize(inp)}__{sanitize(aname)}.npz")
            cols = {k: [] for k in recs[0].keys()} if recs else {}
            for r in recs:
                for k in cols:
                    cols[k].append(r[k])
            save_columnar_npz(
                out, cols,
                dtypes={
                    "event": np.int32,
                    "reco_idx": np.int32,
                    "reco_pt": np.float32,
                    "reco_eta": np.float32,
                    "reco_phi": np.float32,
                    "is_matched": np.int32,
                    "gen_pt": np.float32,
                    "gen_eta": np.float32,
                    "gen_phi": np.float32,
                    "dr": np.float32,
                    "dz_cat": np.int32,
                }
            )

        for (inp, aname), cols in unmatched_counts.items():
            out = os.path.join(out_cache, f"unmatched_counts__{sanitize(inp)}__{sanitize(aname)}.npz")
            save_columnar_npz(
                out, cols,
                dtypes={
                    "event": np.int32,
                    "dz_cat": np.int32,
                    "nunmatched_gen": np.int32,
                    "nunmatched_reco": np.int32,
                }
            )

        for (inp, aname), em in evt_metrics.items():
            out = os.path.join(out_cache, f"event_metrics__{sanitize(inp)}__{sanitize(aname)}.npz")

            cols = {"event": em["event"], "dz_cat": em["dz_cat"], "nseeds": em["nseeds"]}
            dtypes = {"event": np.int32, "dz_cat": np.int32, "nseeds": np.int32}

            for T, arr in em["njet_ge_T"].items():
                k = f"njet_ge_{int(T)}"
                cols[k] = arr
                dtypes[k] = np.int32
            for T, arr in em["ht_ge_T"].items():
                k = f"ht_ge_{int(T)}"
                cols[k] = arr
                dtypes[k] = np.float32

            save_columnar_npz(out, cols, dtypes)

        if do_agree:
            for (inp, aname), fracs in agreement.items():
                out = os.path.join(out_cache, f"agreement__{sanitize(inp)}__{sanitize(aname)}.npz")
                np.savez_compressed(out, fractions=np.asarray(fracs, dtype=np.float32))

        # -----------------------------
        # Write AK-compat caches
        # -----------------------------
        if do_akcompat:
            # (A) RECO<->RECO matched-pair overlap metrics (IoU etc.)
            for (inp, aname), recs in akcompat_records.items():
                if not recs:
                    continue
                out = os.path.join(out_cache, f"akcompat__{sanitize(inp)}__{sanitize(aname)}.npz")
                cols = {k: [] for k in recs[0].keys()}
                for r in recs:
                    for k in cols:
                        cols[k].append(r[k])

                save_columnar_npz(
                    out, cols,
                    dtypes={
                        "event": np.int32,
                        "dz_cat": np.int32,

                        "ref_pt": np.float32,
                        "ref_eta": np.float32,
                        "ref_genmatched": np.int32,
                        "dr_ref_alt": np.float32,

                        "iou": np.float32,
                        "f_ref": np.float32,
                        "f_alt": np.float32,

                        "iou_unw": np.float32,
                        "f_ref_unw": np.float32,
                        "f_alt_unw": np.float32,
                    }
                )

            # (B) GEN-common: AK and ALT matched to the SAME gen jet
            for (inp, aname), recs in akcompat_gen_records.items():
                if not recs:
                    continue
                out = os.path.join(out_cache, f"akcompat_gen__{sanitize(inp)}__{sanitize(aname)}.npz")
                cols = {k: [] for k in recs[0].keys()}
                for r in recs:
                    for k in cols:
                        cols[k].append(r[k])

                save_columnar_npz(
                    out, cols,
                    dtypes={
                        "event": np.int32,
                        "dz_cat": np.int32,

                        "gen_idx": np.int32,
                        "gen_pt": np.float32,
                        "gen_eta": np.float32,
                        "gen_phi": np.float32,

                        "ref_pt": np.float32,
                        "alt_pt": np.float32,

                        "iou": np.float32,
                        "f_ref": np.float32,
                        "f_alt": np.float32,

                        "iou_unw": np.float32,
                        "f_ref_unw": np.float32,
                        "f_alt_unw": np.float32,

                        "ratio_n": np.float32,
                        "ratio_pt": np.float32,
                    }
                )

            # (C) Ref-side AK->ALT matching efficiency inputs
            for (inp, aname), recs in akmatch_ref_records.items():
                if not recs:
                    continue
                out = os.path.join(out_cache, f"akmatch_ref__{sanitize(inp)}__{sanitize(aname)}.npz")
                cols = {k: [] for k in recs[0].keys()}
                for r in recs:
                    for k in cols:
                        cols[k].append(r[k])

                save_columnar_npz(
                    out, cols,
                    dtypes={
                        "event": np.int32,
                        "dz_cat": np.int32,
                        "ref_idx": np.int32,
                        "ref_pt": np.float32,
                        "ref_eta": np.float32,
                        "is_matched": np.int32,
                    }
                )

            # (D) Alt-side ALT fake-rate inputs
            for (inp, aname), recs in akmatch_alt_records.items():
                if not recs:
                    continue
                out = os.path.join(out_cache, f"akmatch_alt__{sanitize(inp)}__{sanitize(aname)}.npz")
                cols = {k: [] for k in recs[0].keys()}
                for r in recs:
                    for k in cols:
                        cols[k].append(r[k])

                save_columnar_npz(
                    out, cols,
                    dtypes={
                        "event": np.int32,
                        "dz_cat": np.int32,
                        "alt_idx": np.int32,
                        "alt_pt": np.float32,
                        "alt_eta": np.float32,
                        "is_fake": np.int32,
                    }
                )

        print(f"Done processing {proc}. Cache in: {out_cache}")

    print("\nAll processing done.")



if __name__ == "__main__":
    args = parse_args()
    cfg = load_cfg_from_path(args.config)
    tag = config_tag_from_path(args.config)
    run(cfg, tag)
