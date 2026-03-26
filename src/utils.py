# src/utils.py
import os
import inspect
import numpy as np
import awkward as ak
import uproot


# -------------------------
# General helpers
# -------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_arrays(root_path, tree_name, branch_list, library="ak"):
    with uproot.open(root_path) as f:
        return f[tree_name].arrays(branch_list, library=library)


def _to_root_writable_array(values):
    """
    Convert a python list of per-event objects into something uproot can write.
    - jagged content -> awkward Array
    - flat scalars   -> numpy array
    """
    if len(values) == 0:
        return ak.Array([])

    sample = values[0]
    if isinstance(sample, (list, tuple, np.ndarray, ak.highlevel.Array)):
        return ak.Array(values)
    return np.asarray(values)


def write_root_tree(outpath: str, tree_name: str, columns: dict):
    ensure_dir(os.path.dirname(outpath))
    writable = {k: _to_root_writable_array(v) for k, v in columns.items()}
    with uproot.recreate(outpath) as fout:
        fout[tree_name] = writable


def save_columnar_npz(outpath: str, cols: dict, dtypes: dict):
    out = {}
    for k, v in cols.items():
        out[k] = np.asarray(v, dtype=dtypes[k])
    np.savez_compressed(outpath, **out)


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


def wrap_phi_np(phi):
    phi = np.asarray(phi, dtype=float)
    return (phi + np.pi) % (2 * np.pi) - np.pi


# -------------------------
# Algorithm call helper
# -------------------------
def call_algo_with_supported_kwargs(fn, eta, phi, pt, extra_kwargs, algo_kwargs):
    """
    Call clustering function with only the kwargs it actually accepts.
    """
    sig = inspect.signature(fn)
    params = sig.parameters

    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    )

    merged = {}
    merged.update(extra_kwargs)
    merged.update(algo_kwargs)

    if accepts_var_kw:
        return fn(eta, phi, pt, **merged)

    allowed = {k: v for k, v in merged.items() if k in params}
    return fn(eta, phi, pt, **allowed)


# -------------------------
# dz category helper
# -------------------------
def compute_event_dz_cat_from_snapshot(cfg, data, ievt: int) -> int:
    if not cfg.Z_SPLIT.get("enabled", False):
        return 0

    zgen_br = "GenVtx_z"
    zreco_br = "L1Vtx_z"
    sumpt_br = "L1Vtx_sumpt"
    nreco_br = "nL1Vtx"

    needed = [zgen_br, zreco_br, sumpt_br, nreco_br]
    if any(br not in data.fields for br in needed):
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


# -------------------------
# Matching: greedy one-to-one (GEN->RECO)
# -------------------------
def match_gen_to_reco(gen_pt, gen_eta, gen_phi,
                      reco_pt, reco_eta, reco_phi,
                      dR=0.3,
                      pt_gen_min=0.0,
                      pt_reco_min=0.0):
    gen_pt = np.asarray(gen_pt, dtype=float)
    gen_eta = np.asarray(gen_eta, dtype=float)
    gen_phi = np.asarray(gen_phi, dtype=float)
    reco_pt = np.asarray(reco_pt, dtype=float)
    reco_eta = np.asarray(reco_eta, dtype=float)
    reco_phi = np.asarray(reco_phi, dtype=float)

    gen_sel = (gen_pt >= float(pt_gen_min))
    reco_sel = (reco_pt >= float(pt_reco_min))

    gen_idx_all = np.where(gen_sel)[0]
    reco_idx_all = np.where(reco_sel)[0]

    if len(gen_idx_all) == 0 or len(reco_idx_all) == 0:
        return [], gen_idx_all.tolist(), reco_idx_all.tolist()

    gen_sorted = gen_idx_all[np.argsort(gen_pt[gen_idx_all])[::-1]]

    used_reco = set()
    matched = []
    thr2 = float(dR) * float(dR)

    for ig in gen_sorted:
        available = np.array([ir for ir in reco_idx_all if int(ir) not in used_reco], dtype=int)
        if available.size == 0:
            break

        dphi = np.arctan2(np.sin(reco_phi[available] - gen_phi[ig]),
                          np.cos(reco_phi[available] - gen_phi[ig]))
        deta = reco_eta[available] - gen_eta[ig]
        dr2 = deta * deta + dphi * dphi

        jbest = int(np.argmin(dr2))
        if float(dr2[jbest]) >= thr2:
            continue

        ir = int(available[jbest])
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

    for ir in reco_sorted:
        available = np.array([ig for ig in gen_idx_all if int(ig) not in used_gen], dtype=int)
        if available.size == 0:
            break

        dphi = np.arctan2(np.sin(gen_phi[available] - reco_phi[ir]),
                          np.cos(gen_phi[available] - reco_phi[ir]))
        deta = gen_eta[available] - reco_eta[ir]
        dr2 = deta * deta + dphi * dphi

        jbest = int(np.argmin(dr2))
        if float(dr2[jbest]) >= thr2:
            continue

        ig = int(available[jbest])
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
# Matching: greedy one-to-one (RECO->RECO)
# -------------------------
def match_reco_to_reco(ref_pt, ref_eta, ref_phi,
                       alt_pt, alt_eta, alt_phi,
                       dR=0.2,
                       pt_ref_min=0.0,
                       pt_alt_min=0.0):
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
        available = np.array([ia for ia in alt_idx_all if int(ia) not in used_alt], dtype=int)
        if available.size == 0:
            break

        dphi = np.arctan2(np.sin(alt_phi[available] - ref_phi[ir]),
                          np.cos(alt_phi[available] - ref_phi[ir]))
        deta = alt_eta[available] - ref_eta[ir]
        dr2 = deta * deta + dphi * dphi

        jbest = int(np.argmin(dr2))
        if float(dr2[jbest]) >= thr2:
            continue

        ia = int(available[jbest])
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
# Constituent overlap metrics
# -------------------------
def pt_weighted_constituent_overlap(cand_pt, assign_ref, ref_idx, assign_alt, alt_idx):
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
    iou = shared / union if union > 0 else 0.0

    return float(f_ref), float(f_alt), float(iou)


def unweighted_constituent_overlap(assign_ref, ref_idx, assign_alt, alt_idx):
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
    iou = n_int / n_uni if n_uni > 0 else 0.0

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
# Assignment remapping after jet selection
# -------------------------
def remap_assign_after_jet_pt_cut(assign, keep_mask):
    """
    Remap per-constituent jet assignment indices after removing jets.

    assign: int array of length Ncands, values in [-1, 0..Njets-1] (old jet indices)
    keep_mask: bool array of length Njets (old jet list), True for jets to keep

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
        new_assign[ok] = mapped
    return new_assign, old_to_new


# -------------------------
# Cache writing helpers
# -------------------------
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
            **{
                k: np.array([], dtype=(np.int32 if k in ("event", "gen_idx", "reco_idx", "dz_cat") else np.float32))
                for k in keys
            }
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