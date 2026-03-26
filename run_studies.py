# run_studies.py
import os
import argparse
import numpy as np
import awkward as ak

from src.config_utils import (
    load_cfg_from_path,
    config_tag_from_path,
    enabled_inputs,
    enabled_algos_with_cfg,
    snapshot_file_path,
    snapshot_tree_name,
    snapshot_branch_list,
    snapshot_branch_name_cand,
    snapshot_branch_name_algo,
    cache_dir_path,
    sanitize,
    maybe_tqdm,
    resolve_config_path,
)
from src.utils import (
    ensure_dir,
    load_arrays,
    compute_event_dz_cat_from_snapshot,
    match_gen_to_reco,
    match_reco_to_gen,
    match_reco_to_reco,
    pt_weighted_constituent_overlap,
    unweighted_constituent_overlap,
    save_columnar_npz,
    save_matches_npz,
    jet_constituent_count,
    jet_constituent_sumpt,
    safe_ratio,
    remap_assign_after_jet_pt_cut,
    wrap_phi_np,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Read clustered snapshot ROOT and derive study caches for plotting."
    )
    ap.add_argument(
        "--config", "-c", default="config.py",
        help="Config file (auto-resolves from configs/ if not found locally)"
    )
    return ap.parse_args()


def _load_snapshot_cands(data, inp, ievt):
    out = {
        "pt": np.asarray(ak.to_numpy(data[snapshot_branch_name_cand(inp, 'pt')][ievt]), dtype=float),
        "eta": np.asarray(ak.to_numpy(data[snapshot_branch_name_cand(inp, 'eta')][ievt]), dtype=float),
        "phi": wrap_phi_np(np.asarray(ak.to_numpy(data[snapshot_branch_name_cand(inp, 'phi')][ievt]), dtype=float)),
    }

    mass_br = snapshot_branch_name_cand(inp, "mass")
    charge_br = snapshot_branch_name_cand(inp, "charge")
    pid_br = snapshot_branch_name_cand(inp, "abs_pdgid")

    if mass_br in data.fields:
        out["mass"] = np.asarray(ak.to_numpy(data[mass_br][ievt]), dtype=float)
    if charge_br in data.fields:
        out["charge"] = np.asarray(ak.to_numpy(data[charge_br][ievt]))
    if pid_br in data.fields:
        out["abs_pdgid"] = np.asarray(ak.to_numpy(data[pid_br][ievt]), dtype=int)

    return out


def _load_snapshot_algo_raw(data, inp, algo, ievt):
    rpt = np.asarray(ak.to_numpy(data[snapshot_branch_name_algo(inp, algo, "jet_pt")][ievt]), dtype=float)
    reta = np.asarray(ak.to_numpy(data[snapshot_branch_name_algo(inp, algo, "jet_eta")][ievt]), dtype=float)
    rphi = wrap_phi_np(np.asarray(ak.to_numpy(data[snapshot_branch_name_algo(inp, algo, "jet_phi")][ievt]), dtype=float))
    assign = np.asarray(ak.to_numpy(data[snapshot_branch_name_algo(inp, algo, "cand_jetIdx")][ievt]), dtype=int)
    seedmask = np.asarray(ak.to_numpy(data[snapshot_branch_name_algo(inp, algo, "cand_isSeed")][ievt]), dtype=bool)
    return rpt, reta, rphi, assign, seedmask


def run(cfg, cfg_tag: str):
    inputs = enabled_inputs(cfg)
    algos = enabled_algos_with_cfg(cfg)

    do_akcompat = cfg.STUDIES.get("ak_compat", False)
    ak_ref_algo = cfg.AK_COMPAT.get("ref_algo", "AntiKt")
    ak_dR = float(cfg.AK_COMPAT.get("dR_match", 0.2))

    jet_thresholds = np.asarray(
        cfg.PT_BINS.get("jet_thresholds", np.array([20, 30, 40, 50], dtype=float)),
        dtype=float
    )
    ht_thresholds = np.asarray(
        cfg.PT_BINS.get("ht_thresholds", np.array([20, 30, 40, 50], dtype=float)),
        dtype=float
    )

    for proc, _pinfo in cfg.PROCESSES.items():
        in_snapshot = snapshot_file_path(cfg, cfg_tag, proc)
        out_cache = cache_dir_path(cfg, cfg_tag, proc)
        ensure_dir(out_cache)

        if not os.path.exists(in_snapshot):
            raise RuntimeError(
                f"Missing clustered snapshot: {in_snapshot}\n"
                f"Run run_clustering.py first with the same --config."
            )

        print(f"\n=== STUDIES: {proc} | snapshot: {in_snapshot} | config: {cfg_tag} ===")

        data = load_arrays(in_snapshot, snapshot_tree_name(cfg), snapshot_branch_list(cfg), library="ak")
        n_events = len(data["event_idx"])
        ev_idx = np.arange(n_events, dtype=int)

        print(f"Loaded {n_events} clustered events from snapshot")

        # -----------------------------
        # Denominators / containers
        # -----------------------------
        denom_gen_pt = []
        denom_gen_eta = []
        denom_dz_cat = []

        reco_denom = {
            (inp, aname): {"event": [], "reco_idx": [], "reco_pt": [], "reco_eta": [], "reco_phi": [], "dz_cat": []}
            for inp in inputs for aname, _ in algos
        }

        evt_metrics = {
            (inp, aname): {"event": [], "dz_cat": [], "njet_ge_T": {}, "ht_ge_T": {}, "nseeds": []}
            for inp in inputs for aname, _ in algos
        }
        for T in jet_thresholds:
            for key in evt_metrics:
                evt_metrics[key]["njet_ge_T"][float(T)] = []
        for T in ht_thresholds:
            for key in evt_metrics:
                evt_metrics[key]["ht_ge_T"][float(T)] = []

        match_records = {(inp, aname): [] for inp in inputs for aname, _ in algos}
        reco_match_records = {(inp, aname): [] for inp in inputs for aname, _ in algos}
        unmatched_counts = {
            (inp, aname): {"event": [], "dz_cat": [], "nunmatched_gen": [], "nunmatched_reco": []}
            for inp in inputs for aname, _ in algos
        }

        akcompat_records = {
            (inp, aname): []
            for inp in inputs
            for aname, _ in algos
            if aname != ak_ref_algo
        }
        akcompat_gen_records = {
            (inp, aname): []
            for inp in inputs
            for aname, _ in algos
            if aname != ak_ref_algo
        }
        akmatch_ref_records = {
            (inp, aname): []
            for inp in inputs
            for aname, _ in algos
            if aname != ak_ref_algo
        }
        akmatch_alt_records = {
            (inp, aname): []
            for inp in inputs
            for aname, _ in algos
            if aname != ak_ref_algo
        }

        # -----------------------------
        # Event loop
        # -----------------------------
        for isnap in maybe_tqdm(cfg, ev_idx, total=len(ev_idx), desc=f"{proc}: studies"):
            isnap = int(isnap)
            event_id = int(np.asarray(data["event_idx"][isnap]).reshape(-1)[0])

            dz_cat_evt = compute_event_dz_cat_from_snapshot(cfg, data, isnap)

            gen_pt = np.asarray(ak.to_numpy(data["GenJet_pt"][isnap]), dtype=float)
            gen_eta = np.asarray(ak.to_numpy(data["GenJet_eta"][isnap]), dtype=float)
            gen_phi = wrap_phi_np(np.asarray(ak.to_numpy(data["GenJet_phi"][isnap]), dtype=float))

            gsel = (gen_pt >= float(cfg.MATCHING["pt_gen_min"]))
            gen_pt = gen_pt[gsel]
            gen_eta = gen_eta[gsel]
            gen_phi = gen_phi[gsel]

            if gen_pt.size == 0:
                continue

            denom_gen_pt.append(gen_pt.astype(np.float32))
            denom_gen_eta.append(gen_eta.astype(np.float32))
            denom_dz_cat.append(np.full(gen_pt.shape, dz_cat_evt, dtype=np.int32))

            reco_by_key = {}
            assign_by_key = {}
            seedmask_by_key = {}
            cand_arrays = {}

            # -----------------------------
            # Per input / algo reconstruction info from snapshot
            # -----------------------------
            for inp in inputs:
                cand_arrays[inp] = _load_snapshot_cands(data, inp, isnap)
                cand_pt_evt = cand_arrays[inp]["pt"]

                for aname, _ in algos:
                    rpt_all, reta_all, rphi_all, assign_raw, seedmask_raw = _load_snapshot_algo_raw(
                        data, inp, aname, isnap
                    )

                    rsel = (rpt_all >= float(cfg.MATCHING["pt_reco_min"]))
                    rpt = rpt_all[rsel]
                    reta = reta_all[rsel]
                    rphi = rphi_all[rsel]

                    assign_new, _old_to_new = remap_assign_after_jet_pt_cut(assign_raw, rsel)

                    reco_by_key[(inp, aname)] = (rpt, reta, rphi)
                    assign_by_key[(inp, aname)] = np.asarray(assign_new, dtype=int)
                    seedmask_by_key[(inp, aname)] = np.asarray(seedmask_raw, dtype=bool)

                    for ir in range(len(rpt)):
                        reco_denom[(inp, aname)]["event"].append(event_id)
                        reco_denom[(inp, aname)]["reco_idx"].append(ir)
                        reco_denom[(inp, aname)]["reco_pt"].append(float(rpt[ir]))
                        reco_denom[(inp, aname)]["reco_eta"].append(float(reta[ir]))
                        reco_denom[(inp, aname)]["reco_phi"].append(float(rphi[ir]))
                        reco_denom[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))

                    nseeds = int(np.sum(seedmask_by_key[(inp, aname)])) if seedmask_by_key[(inp, aname)].size else 0
                    evt_metrics[(inp, aname)]["event"].append(event_id)
                    evt_metrics[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))
                    evt_metrics[(inp, aname)]["nseeds"].append(nseeds)

                    for T in jet_thresholds:
                        evt_metrics[(inp, aname)]["njet_ge_T"][float(T)].append(int(np.sum(rpt >= float(T))))
                    for T in ht_thresholds:
                        evt_metrics[(inp, aname)]["ht_ge_T"][float(T)].append(float(np.sum(rpt[rpt >= float(T)])))

            # -----------------------------
            # GEN<->RECO matching
            # -----------------------------
            gen2reco_evt = {}

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

                unmatched_counts[(inp, aname)]["event"].append(event_id)
                unmatched_counts[(inp, aname)]["dz_cat"].append(int(dz_cat_evt))
                unmatched_counts[(inp, aname)]["nunmatched_gen"].append(int(len(un_g)))
                unmatched_counts[(inp, aname)]["nunmatched_reco"].append(int(len(un_r)))

                for m in matched:
                    match_records[(inp, aname)].append({
                        "event": event_id,
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
                        "event": event_id,
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
            # AK-compatibility
            # -----------------------------
            if do_akcompat:
                for inp in inputs:
                    ref_key = (inp, ak_ref_algo)
                    if ref_key not in reco_by_key:
                        continue

                    ref_pt, ref_eta, ref_phi = reco_by_key[ref_key]
                    ref_assign = assign_by_key[ref_key]
                    ref_genmatched = set(gen2reco_evt.get(ref_key, {}).values())

                    cand_pt_evt = cand_arrays[inp]["pt"]

                    for aname, _ in algos:
                        if aname == ak_ref_algo:
                            continue

                        alt_key = (inp, aname)
                        if alt_key not in reco_by_key:
                            continue

                        alt_pt, alt_eta, alt_phi = reco_by_key[alt_key]
                        alt_assign = assign_by_key[alt_key]

                        # ---- GEN-driven: both matched to same gen jet
                        ref_map = gen2reco_evt.get(ref_key, {})
                        alt_map = gen2reco_evt.get(alt_key, {})

                        if ref_map and alt_map:
                            common_gen = sorted(set(ref_map.keys()) & set(alt_map.keys()))
                            for ig in common_gen:
                                ir = int(ref_map[ig])
                                ia = int(alt_map[ig])

                                f_ref_w, f_alt_w, iou_w = pt_weighted_constituent_overlap(
                                    cand_pt_evt, ref_assign, ir, alt_assign, ia
                                )
                                f_ref_u, f_alt_u, iou_u = unweighted_constituent_overlap(
                                    ref_assign, ir, alt_assign, ia
                                )

                                n_ref = jet_constituent_count(ref_assign, ir)
                                n_alt = jet_constituent_count(alt_assign, ia)
                                sumpt_ref = jet_constituent_sumpt(cand_pt_evt, ref_assign, ir)
                                sumpt_alt = jet_constituent_sumpt(cand_pt_evt, alt_assign, ia)

                                akcompat_gen_records[(inp, aname)].append({
                                    "event": event_id,
                                    "dz_cat": int(dz_cat_evt),
                                    "gen_idx": int(ig),
                                    "gen_pt": float(gen_pt[ig]),
                                    "gen_eta": float(gen_eta[ig]),
                                    "gen_phi": float(gen_phi[ig]),
                                    "ref_pt": float(ref_pt[ir]) if ir < len(ref_pt) else -1.0,
                                    "alt_pt": float(alt_pt[ia]) if ia < len(alt_pt) else -1.0,
                                    "iou": float(iou_w),
                                    "f_ref": float(f_ref_w),
                                    "f_alt": float(f_alt_w),
                                    "iou_unw": float(iou_u),
                                    "f_ref_unw": float(f_ref_u),
                                    "f_alt_unw": float(f_alt_u),
                                    "ratio_n": float(safe_ratio(n_ref, n_alt, default=np.nan)),
                                    "ratio_pt": float(safe_ratio(sumpt_ref, sumpt_alt, default=np.nan)),
                                })

                        # ---- RECO<->RECO matching, ref-driven
                        matches, un_ref, un_alt = match_reco_to_reco(
                            ref_pt, ref_eta, ref_phi,
                            alt_pt, alt_eta, alt_phi,
                            dR=ak_dR,
                            pt_ref_min=float(cfg.MATCHING["pt_reco_min"]),
                            pt_alt_min=float(cfg.MATCHING["pt_reco_min"]),
                        )

                        matched_ref_set = {int(m["ref_idx"]) for m in matches}
                        matched_alt_set = {int(m["alt_idx"]) for m in matches}

                        for ir in range(len(ref_pt)):
                            if float(ref_pt[ir]) < float(cfg.MATCHING["pt_reco_min"]):
                                continue
                            akmatch_ref_records[(inp, aname)].append({
                                "event": event_id,
                                "dz_cat": int(dz_cat_evt),
                                "ref_idx": int(ir),
                                "ref_pt": float(ref_pt[ir]),
                                "ref_eta": float(ref_eta[ir]),
                                "is_matched": int(ir in matched_ref_set),
                            })

                        for ia in range(len(alt_pt)):
                            if float(alt_pt[ia]) < float(cfg.MATCHING["pt_reco_min"]):
                                continue
                            akmatch_alt_records[(inp, aname)].append({
                                "event": event_id,
                                "dz_cat": int(dz_cat_evt),
                                "alt_idx": int(ia),
                                "alt_pt": float(alt_pt[ia]),
                                "alt_eta": float(alt_eta[ia]),
                                "is_fake": int(ia not in matched_alt_set),
                            })

                        for m in matches:
                            ir = int(m["ref_idx"])
                            ia = int(m["alt_idx"])

                            f_ref_w, f_alt_w, iou_w = pt_weighted_constituent_overlap(
                                cand_pt_evt, ref_assign, ir, alt_assign, ia
                            )
                            f_ref_u, f_alt_u, iou_u = unweighted_constituent_overlap(
                                ref_assign, ir, alt_assign, ia
                            )

                            akcompat_records[(inp, aname)].append({
                                "event": event_id,
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
        print("Writing study caches...")

        denom_gen_pt = np.concatenate(denom_gen_pt) if denom_gen_pt else np.array([], dtype=np.float32)
        denom_gen_eta = np.concatenate(denom_gen_eta) if denom_gen_eta else np.array([], dtype=np.float32)
        denom_dz_cat = np.concatenate(denom_dz_cat) if denom_dz_cat else np.array([], dtype=np.int32)

        np.savez_compressed(
            os.path.join(out_cache, "denom_genjets.npz"),
            gen_pt=denom_gen_pt,
            gen_eta=denom_gen_eta,
            dz_cat=denom_dz_cat,
        )

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

            cols = {
                "event": [],
                "reco_idx": [],
                "reco_pt": [],
                "reco_eta": [],
                "reco_phi": [],
                "is_matched": [],
                "gen_pt": [],
                "gen_eta": [],
                "gen_phi": [],
                "dr": [],
                "dz_cat": [],
            }
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

            cols = {
                "event": em["event"],
                "dz_cat": em["dz_cat"],
                "nseeds": em["nseeds"],
            }
            dtypes = {
                "event": np.int32,
                "dz_cat": np.int32,
                "nseeds": np.int32,
            }

            for T, arr in em["njet_ge_T"].items():
                k = f"njet_ge_{int(T)}"
                cols[k] = arr
                dtypes[k] = np.int32

            for T, arr in em["ht_ge_T"].items():
                k = f"ht_ge_{int(T)}"
                cols[k] = arr
                dtypes[k] = np.float32

            save_columnar_npz(out, cols, dtypes)

        if do_akcompat:
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

        print(f"Done studies for {proc}. Cache in: {out_cache}")

    print("\nAll studies done.")


if __name__ == "__main__":
    args = parse_args()

    cfg_path = resolve_config_path(args.config)
    cfg = load_cfg_from_path(cfg_path)
    tag = config_tag_from_path(cfg_path)

    run(cfg, tag)