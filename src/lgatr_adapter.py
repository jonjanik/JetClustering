# src/ml/lgatr_adapter.py
import os
import json
import numpy as np

# Lazy imports: only required when you actually run LGATr
def _lazy_import_torch():
    import torch
    return torch

def _lazy_import_hdbscan():
    import hdbscan
    return hdbscan


class _EventBatch:
    """
    Minimal stub compatible with LGATrModel.forward() usage:
      - input_vectors: (N,4) tensor
      - input_scalars: (N,S) tensor
      - batch_idx: (N,) tensor (ints, contiguous per event)
    """
    __slots__ = ("input_vectors", "input_scalars", "batch_idx")

    def __init__(self, input_vectors, input_scalars, batch_idx):
        self.input_vectors = input_vectors
        self.input_scalars = input_scalars
        self.batch_idx = batch_idx


# -----------------------------
# Feature building
# -----------------------------
def _wrap_phi(phi):
    return (phi + np.pi) % (2 * np.pi) - np.pi


def build_p4_from_ptetaphi(pt, eta, phi, mass=0.13957):
    """
    Returns numpy array (N,4) as (E,px,py,pz).
    """
    pt = np.asarray(pt, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)
    phi = _wrap_phi(np.asarray(phi, dtype=np.float32))
    m = np.full_like(pt, float(mass), dtype=np.float32) if np.isscalar(mass) else np.asarray(mass, dtype=np.float32)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px * px + py * py + pz * pz + m * m)
    return np.stack([E, px, py, pz], axis=1).astype(np.float32)


def default_pid9_onehot(abs_pdgid):
    """
    Produce 9-dim one-hot-ish vector per cand.

    Defaults are PF-ish categories; you can override by providing your own
    pid9_map in config/params.

    Indices (default):
      0: charged hadron (211,321,2212, ... -> anything charged hadron-ish)
      1: neutral hadron (130, 2112, 310, 3122, ...)
      2: photon (22)
      3: electron (11)
      4: muon (13)
      5: tau (15)
      6: HF hadron / heavy (411,421,431,511,521,531,...)
      7: neutrino (12,14,16)
      8: other/unknown

    Note: if your training used different PID encoding, override this.
    """
    abs_pdgid = np.asarray(abs_pdgid, dtype=np.int32)
    N = abs_pdgid.shape[0]
    out = np.zeros((N, 9), dtype=np.float32)

    pid = abs_pdgid

    is_photon = (pid == 22)
    is_e = (pid == 11)
    is_mu = (pid == 13)
    is_tau = (pid == 15)
    is_nu = np.isin(pid, [12, 14, 16])

    # very rough heavy-flavor hadron heuristic
    is_hf = ((pid // 100) % 10 == 4) | ((pid // 100) % 10 == 5) | ((pid // 1000) % 10 == 4) | ((pid // 1000) % 10 == 5)

    # neutral hadrons (rough)
    is_neutral_had = np.isin(pid, [130, 310, 311, 3122, 2112, 3212, 3222, 3322, 3312])

    # charged hadrons (fallback): common charged hadrons + anything not caught but not lepton/photon/nu/hf
    is_ch_had = np.isin(pid, [211, 321, 2212, 2112])  # 2112 is actually neutral; gets overridden by is_neutral_had
    is_ch_had = is_ch_had | (
        (~is_neutral_had) & (~is_photon) & (~is_e) & (~is_mu) & (~is_tau) & (~is_nu) & (~is_hf) & (pid > 0)
    )

    # fill categories in order, last is unknown
    out[is_ch_had, 0] = 1.0
    out[is_neutral_had, 1] = 1.0
    out[is_photon, 2] = 1.0
    out[is_e, 3] = 1.0
    out[is_mu, 4] = 1.0
    out[is_tau, 5] = 1.0
    out[is_hf, 6] = 1.0
    out[is_nu, 7] = 1.0

    known = is_ch_had | is_neutral_had | is_photon | is_e | is_mu | is_tau | is_hf | is_nu
    out[~known, 8] = 1.0
    return out


def build_input_scalars(pt, eta, phi, charge=None, abs_pdgid=None, *, no_pid=False, base_mode="logpt_etaphi"):
    """
    Returns (N,S) float32.

    For LGATrModel code you pasted:
      - no_pid=True  -> S = 3
      - no_pid=False -> S = 12 = 3 base + 9 pid-encoding
    """
    pt = np.asarray(pt, dtype=np.float32)
    eta = np.asarray(eta, dtype=np.float32)
    phi = _wrap_phi(np.asarray(phi, dtype=np.float32))

    if base_mode == "logpt_etaphi":
        base = np.stack([np.log(np.maximum(pt, 1e-6)), eta, phi], axis=1).astype(np.float32)
    elif base_mode == "pt_etaphi":
        base = np.stack([pt, eta, phi], axis=1).astype(np.float32)
    else:
        raise ValueError(f"Unknown base_mode: {base_mode}")

    if no_pid:
        return base

    # pid9 features (9 dims)
    if abs_pdgid is None:
        # if you truly don't have PID in your ntuple, you MUST train with --no_pid
        raise ValueError("abs_pdgid is required when no_pid=False (n_scalars=12 expected).")

    pid9 = default_pid9_onehot(np.asarray(abs_pdgid, dtype=np.int32))

    # You may also want charge in scalars; the original code says 12=3+9,
    # so we DON'T add charge unless you explicitly include it in base_mode.
    # If your training used charge, adapt base_mode or encoding accordingly.
    scalars = np.concatenate([base, pid9], axis=1).astype(np.float32)
    if scalars.shape[1] != 12:
        raise RuntimeError(f"Expected 12 scalars, got {scalars.shape[1]}")
    return scalars


# -----------------------------
# Model loading + caching
# -----------------------------
_MODEL_CACHE = {}  # key -> (model, meta)

def load_lgatr_bundle(model_path, device="cuda"):
    """
    model_path can be:
      - a .pt/.pth weights file
      - a directory containing weights + model_meta.json

    Expected:
      - weights: weights.pt (or model.pt)
      - meta: model_meta.json (optional but recommended)
    """
    torch = _lazy_import_torch()
    device = str(device)

    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        meta_path = os.path.join(model_path, "model_meta.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, "r"))
        else:
            meta = {}
        # resolve weights
        for cand in ["weights.pt", "model.pt", "checkpoint.pt", "weights.pth", "model.pth"]:
            w = os.path.join(model_path, cand)
            if os.path.exists(w):
                weights_path = w
                break
        else:
            raise FileNotFoundError(f"No weights found in directory: {model_path}")
    else:
        weights_path = model_path
        meta = {}

    cache_key = (os.path.realpath(weights_path), device, json.dumps(meta, sort_keys=True))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Import the LGATr model class from the transformer repo code
    # You will vendor/copy this file OR install their package such that this import works.
    # Option A: vendor lgatr.py into your repo and import from your own path
    # Option B: keep their repo on PYTHONPATH and import from there
    try:
        from src.models.LGATr.lgatr import LGATrModel  # if you copied it into your repo with same structure
    except Exception:
        # fallback: if you keep the transformer repo separately and add it to PYTHONPATH
        from jetclustering.src.models.LGATr.lgatr import LGATrModel  # adjust if needed

    # Meta defaults (must match training)
    no_pid = bool(meta.get("no_pid", True))
    n_scalars = int(meta.get("n_scalars", (3 if no_pid else 12)))
    hidden_mv_channels = int(meta.get("hidden_mv_channels", 8))
    hidden_s_channels = int(meta.get("hidden_s_channels", 16))
    blocks = int(meta.get("blocks", 5))
    n_scalars_out = int(meta.get("n_scalars_out", 8))
    return_scalar_coords = bool(meta.get("return_scalar_coords", False))
    obj_score = bool(meta.get("obj_score", False))
    global_features_copy = bool(meta.get("global_features_copy", False))

    model = LGATrModel(
        n_scalars=n_scalars,
        hidden_mv_channels=hidden_mv_channels,
        hidden_s_channels=hidden_s_channels,
        blocks=blocks,
        embed_as_vectors=False,
        n_scalars_out=n_scalars_out,
        return_scalar_coords=return_scalar_coords,
        obj_score=obj_score,
        global_featuers_copy=global_features_copy,
    )

    # load weights
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # handle DataParallel "module."
    clean = {}
    for k, v in state.items():
        clean[k[7:]] = v if k.startswith("module.") else v
    try:
        model.load_state_dict(clean, strict=False)
    except Exception:
        # if keys already clean
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)

    _MODEL_CACHE[cache_key] = (model, meta)
    return model, meta


# -----------------------------
# Inference + clustering
# -----------------------------
def predict_coords_single_event(
    model, meta,
    pt, eta, phi,
    mass=None,
    charge=None,
    abs_pdgid=None,
    device="cuda",
    base_mode="logpt_etaphi",
    cpu_demo=False,
):
    """
    Runs LGATrModel forward on ONE event.
    Returns:
      coords: (N, D) numpy float32
    """
    torch = _lazy_import_torch()

    p4 = build_p4_from_ptetaphi(pt, eta, phi, mass=(0.13957 if mass is None else mass))
    no_pid = bool(meta.get("no_pid", True))

    scalars = build_input_scalars(
        pt, eta, phi,
        charge=charge,
        abs_pdgid=abs_pdgid,
        no_pid=no_pid,
        base_mode=base_mode,
    )

    # tensors
    t_vec = torch.from_numpy(p4).to(device=device, dtype=torch.float32)
    t_sca = torch.from_numpy(scalars).to(device=device, dtype=torch.float32)
    t_bix = torch.zeros((t_vec.shape[0],), device=device, dtype=torch.int64)  # single event => all 0

    batch = _EventBatch(t_vec, t_sca, t_bix)

    with torch.no_grad():
        out = model(batch, cpu_demo=cpu_demo)
        # out: (N, D) or (N, 4) etc depending on beta and coords
        coords = out.detach().float().cpu().numpy()

    return coords


def cluster_coords_single_event(
    coords,
    *,
    method="hdbscan",
    min_cluster_size=10,
    min_samples=20,
    epsilon=0.1,
    dbscan_eps=0.1,
    dbscan_min_samples=10,
):
    """
    coords: (N, D) numpy
    Returns:
      labels: (N,) int, with -1 noise
    """
    coords = np.asarray(coords, dtype=np.float32)
    N = coords.shape[0]
    if N == 0:
        return np.array([], dtype=np.int32)

    if method.lower() == "hdbscan":
        hdbscan = _lazy_import_hdbscan()
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            cluster_selection_epsilon=float(epsilon),
        )
        try:
            labels = clusterer.fit_predict(coords)
        except Exception:
            labels = np.full((N,), -1, dtype=np.int32)
        return np.asarray(labels, dtype=np.int32)

    if method.lower() == "dbscan":
        from sklearn.cluster import DBSCAN
        try:
            labels = DBSCAN(eps=float(dbscan_eps), min_samples=int(dbscan_min_samples)).fit_predict(coords)
        except Exception:
            labels = np.full((N,), -1, dtype=np.int32)
        return np.asarray(labels, dtype=np.int32)

    raise ValueError(f"Unknown clustering method: {method}")


def reindex_labels_to_assign(labels):
    """
    Convert arbitrary labels to contiguous [0..K-1], keeping -1 as noise.
    """
    labels = np.asarray(labels, dtype=np.int32)
    assign = np.full_like(labels, -1)

    uniq = np.unique(labels)
    uniq = uniq[uniq >= 0]
    uniq = np.sort(uniq)
    mapping = {int(l): i for i, l in enumerate(uniq)}

    for i in range(labels.shape[0]):
        l = int(labels[i])
        if l >= 0:
            assign[i] = mapping[l]
    return assign
