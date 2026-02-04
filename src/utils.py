# src/utils.py
import os
import numpy as np
import awkward as ak
import uproot

# -------------------------
# Matching: greedy one-to-one (GEN->RECO)
# -------------------------
def match_gen_to_reco(gen_pt, gen_eta, gen_phi,
                      reco_pt, reco_eta, reco_phi,
                      dR=0.3,
                      pt_gen_min=0.0,
                      pt_reco_min=0.0):
    gen_pt = np.asarray(gen_pt)
    gen_eta = np.asarray(gen_eta)
    gen_phi = np.asarray(gen_phi)
    reco_pt = np.asarray(reco_pt)
    reco_eta = np.asarray(reco_eta)
    reco_phi = np.asarray(reco_phi)

    gen_sel = (gen_pt >= pt_gen_min)
    reco_sel = (reco_pt >= pt_reco_min)

    gen_idx_all = np.where(gen_sel)[0]
    reco_idx_all = np.where(reco_sel)[0]

    if len(gen_idx_all) == 0 or len(reco_idx_all) == 0:
        return [], gen_idx_all.tolist(), reco_idx_all.tolist()

    gen_sorted = gen_idx_all[np.argsort(gen_pt[gen_idx_all])[::-1]]

    used_reco = set()
    matched = []
    thr2 = dR * dR

    reco_phi_sel = reco_phi[reco_idx_all]
    reco_eta_sel = reco_eta[reco_idx_all]

    for ig in gen_sorted:
        dphi = np.arctan2(np.sin(reco_phi_sel - gen_phi[ig]),
                          np.cos(reco_phi_sel - gen_phi[ig]))
        deta = reco_eta_sel - gen_eta[ig]
        dr2 = deta * deta + dphi * dphi

        jbest = int(np.argmin(dr2))
        if dr2[jbest] >= thr2:
            continue

        ir = int(reco_idx_all[jbest])
        if ir in used_reco:
            continue

        used_reco.add(ir)
        dr = float(np.sqrt(dr2[jbest]))
        gpt = float(gen_pt[ig])
        rpt = float(reco_pt[ir])

        matched.append({
            "gen_idx": int(ig),
            "reco_idx": int(ir),
            "gen_pt": gpt,
            "gen_eta": float(gen_eta[ig]),
            "gen_phi": float(gen_phi[ig]),
            "reco_pt": rpt,
            "reco_eta": float(reco_eta[ir]),
            "reco_phi": float(reco_phi[ir]),
            "dr": dr,
            "resp": float(rpt / max(gpt, 1e-6)),
            "dpt_rel": float((rpt - gpt) / max(gpt, 1e-6)),
        })

    matched_gen = {m["gen_idx"] for m in matched}
    matched_reco = {m["reco_idx"] for m in matched}

    unmatched_gen = [int(i) for i in gen_idx_all if int(i) not in matched_gen]
    unmatched_reco = [int(i) for i in reco_idx_all if int(i) not in matched_reco]

    return matched, unmatched_gen, unmatched_reco

# -------------------------
# Matching: greedy one-to-one (RECO->GEN)
# -------------------------
def match_reco_to_gen(reco_pt, reco_eta, reco_phi,
                      gen_pt, gen_eta, gen_phi,
                      dR=0.3,
                      pt_reco_min=0.0,
                      pt_gen_min=0.0):
    """
    Greedy one-to-one matching driven by RECO jets (descending pT).
    Useful for RECO-side purity/fake diagnostics.
    Returns: matched_records(list), unmatched_reco(list), unmatched_gen(list)
    """
    reco_pt = np.asarray(reco_pt, dtype=float)
    reco_eta = np.asarray(reco_eta, dtype=float)
    reco_phi = np.asarray(reco_phi, dtype=float)

    gen_pt = np.asarray(gen_pt, dtype=float)
    gen_eta = np.asarray(gen_eta, dtype=float)
    gen_phi = np.asarray(gen_phi, dtype=float)

    reco_sel = (reco_pt >= float(pt_reco_min))
    gen_sel = (gen_pt >= float(pt_gen_min))

    reco_idx_all = np.where(reco_sel)[0]
    gen_idx_all = np.where(gen_sel)[0]

    if len(reco_idx_all) == 0 or len(gen_idx_all) == 0:
        return [], reco_idx_all.tolist(), gen_idx_all.tolist()

    reco_sorted = reco_idx_all[np.argsort(reco_pt[reco_idx_all])[::-1]]

    used_gen = set()
    matched = []
    thr2 = float(dR) * float(dR)

    gen_phi_sel = gen_phi[gen_idx_all]
    gen_eta_sel = gen_eta[gen_idx_all]

    for ir in reco_sorted:
        dphi = np.arctan2(np.sin(gen_phi_sel - reco_phi[ir]),
                          np.cos(gen_phi_sel - reco_phi[ir]))
        deta = gen_eta_sel - reco_eta[ir]
        dr2 = deta * deta + dphi * dphi

        jbest = int(np.argmin(dr2))
        if float(dr2[jbest]) >= thr2:
            continue

        ig = int(gen_idx_all[jbest])
        if ig in used_gen:
            continue

        used_gen.add(ig)
        dr = float(np.sqrt(dr2[jbest]))

        matched.append({
            "reco_idx": int(ir),
            "gen_idx": int(ig),
            "reco_pt": float(reco_pt[ir]),
            "reco_eta": float(reco_eta[ir]),
            "reco_phi": float(reco_phi[ir]),
            "gen_pt": float(gen_pt[ig]),
            "gen_eta": float(gen_eta[ig]),
            "gen_phi": float(gen_phi[ig]),
            "dr": dr,
        })

    matched_reco = {m["reco_idx"] for m in matched}
    matched_gen = {m["gen_idx"] for m in matched}

    unmatched_reco = [int(i) for i in reco_idx_all if int(i) not in matched_reco]
    unmatched_gen = [int(i) for i in gen_idx_all if int(i) not in matched_gen]

    return matched, unmatched_reco, unmatched_gen


# -------------------------
# Matching: greedy one-to-one (RECO->RECO (alt to ref))
# -------------------------
def match_reco_to_reco(ref_pt, ref_eta, ref_phi,
                       alt_pt, alt_eta, alt_phi,
                       dR=0.2,
                       pt_ref_min=0.0,
                       pt_alt_min=0.0):
    """
    Greedy one-to-one matching driven by REF jets (descending pT).
    """
    ref_pt  = np.asarray(ref_pt,  dtype=float)
    ref_eta = np.asarray(ref_eta, dtype=float)
    ref_phi = np.asarray(ref_phi, dtype=float)

    alt_pt  = np.asarray(alt_pt,  dtype=float)
    alt_eta = np.asarray(alt_eta, dtype=float)
    alt_phi = np.asarray(alt_phi, dtype=float)

    ref_sel = (ref_pt >= float(pt_ref_min))
    alt_sel = (alt_pt >= float(pt_alt_min))

    ref_idx_all = np.where(ref_sel)[0]
    alt_idx_all = np.where(alt_sel)[0]

    if len(ref_idx_all) == 0 or len(alt_idx_all) == 0:
        return [], ref_idx_all.tolist(), alt_idx_all.tolist()

    ref_sorted = ref_idx_all[np.argsort(ref_pt[ref_idx_all])[::-1]]

    used_alt = set()
    matched = []
    thr2 = float(dR) * float(dR)

    for ir in ref_sorted:
        dphi = np.arctan2(np.sin(alt_phi[alt_idx_all] - ref_phi[ir]),
                          np.cos(alt_phi[alt_idx_all] - ref_phi[ir]))
        deta = alt_eta[alt_idx_all] - ref_eta[ir]
        dr2 = deta*deta + dphi*dphi

        jbest = int(np.argmin(dr2))
        if float(dr2[jbest]) >= thr2:
            continue

        ia = int(alt_idx_all[jbest])
        if ia in used_alt:
            continue

        used_alt.add(ia)
        matched.append({
            "ref_idx": int(ir),
            "alt_idx": int(ia),
            "ref_pt": float(ref_pt[ir]),
            "ref_eta": float(ref_eta[ir]),
            "ref_phi": float(ref_phi[ir]),
            "alt_pt": float(alt_pt[ia]),
            "alt_eta": float(alt_eta[ia]),
            "alt_phi": float(alt_phi[ia]),
            "dr": float(np.sqrt(float(dr2[jbest]))),
        })

    matched_ref = {m["ref_idx"] for m in matched}
    matched_alt = {m["alt_idx"] for m in matched}

    unmatched_ref = [int(i) for i in ref_idx_all if int(i) not in matched_ref]
    unmatched_alt = [int(i) for i in alt_idx_all if int(i) not in matched_alt]

    return matched, unmatched_ref, unmatched_alt


# -------------------------
# pT-weighted constituent overlap metrics
# -------------------------
def pt_weighted_constituent_overlap(
    cand_pt,
    assign_ref, ref_idx,
    assign_alt, alt_idx,
):
    cand_pt = np.asarray(cand_pt, dtype=float)
    assign_ref = np.asarray(assign_ref, dtype=int)
    assign_alt = np.asarray(assign_alt, dtype=int)

    ref_mask = (assign_ref == ref_idx)
    alt_mask = (assign_alt == alt_idx)

    if not np.any(ref_mask) or not np.any(alt_mask):
        return 0.0, 0.0, 0.0

    ref_sum = float(np.sum(cand_pt[ref_mask]))
    alt_sum = float(np.sum(cand_pt[alt_mask]))
    if ref_sum <= 0 or alt_sum <= 0:
        return 0.0, 0.0, 0.0

    ref_ids = np.where(ref_mask)[0]
    alt_ids = np.where(alt_mask)[0]
    inter = np.intersect1d(ref_ids, alt_ids)

    shared = float(np.sum(cand_pt[inter])) if inter.size else 0.0
    union = ref_sum + alt_sum - shared

    f_ref = shared / ref_sum
    f_alt = shared / alt_sum
    iou   = shared / union if union > 0 else 0.0

    return float(f_ref), float(f_alt), float(iou)


def unweighted_constituent_overlap(
    assign_ref, ref_idx,
    assign_alt, alt_idx,
):
    assign_ref = np.asarray(assign_ref, dtype=int)
    assign_alt = np.asarray(assign_alt, dtype=int)

    ref_ids = np.where(assign_ref == int(ref_idx))[0]
    alt_ids = np.where(assign_alt == int(alt_idx))[0]

    if ref_ids.size == 0 or alt_ids.size == 0:
        return 0.0, 0.0, 0.0

    inter = np.intersect1d(ref_ids, alt_ids)
    n_ref = float(ref_ids.size)
    n_alt = float(alt_ids.size)
    n_int = float(inter.size)
    n_uni = n_ref + n_alt - n_int

    f_ref = n_int / n_ref if n_ref > 0 else 0.0
    f_alt = n_int / n_alt if n_alt > 0 else 0.0
    iou   = n_int / n_uni if n_uni > 0 else 0.0

    return float(f_ref), float(f_alt), float(iou)


def jet_constituent_ids(assign, jet_idx):
    assign = np.asarray(assign, dtype=int)
    jet_idx = int(jet_idx)
    return np.where(assign == jet_idx)[0]


def jet_constituent_count(assign, jet_idx) -> int:
    ids = jet_constituent_ids(assign, jet_idx)
    return int(ids.size)


def jet_constituent_sumpt(cand_pt, assign, jet_idx) -> float:
    cand_pt = np.asarray(cand_pt, dtype=float)
    ids = jet_constituent_ids(assign, jet_idx)
    if ids.size == 0:
        return 0.0
    return float(np.sum(cand_pt[ids]))


def safe_ratio(num, den, default=np.nan) -> float:
    den = float(den)
    if den <= 0.0:
        return float(default)
    return float(num) / den


# -------------------------
# IO helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_arrays(root_path, tree_name, branch_list, library="ak"):
    with uproot.open(root_path) as f:
        return f[tree_name].arrays(branch_list, library=library)

def save_columnar_npz(outpath: str, cols: dict, dtypes: dict):
    """
    cols: dict key -> python list
    dtypes: dict key -> numpy dtype
    """
    out = {}
    for k, v in cols.items():
        out[k] = np.asarray(v, dtype=dtypes[k])
    np.savez_compressed(outpath, **out)
