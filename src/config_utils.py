# src/config_utils.py
import os
import importlib.util
import numpy as np


def resolve_config_path(cfg_arg: str) -> str:
    """
    Resolution order:
      1. exact path (absolute or relative)
      2. ./configs/<name>
      3. <name> (fallback)
    """
    import os

    # 1) exact path as given
    if os.path.exists(cfg_arg):
        return os.path.abspath(cfg_arg)

    # 2) configs/ prefix
    cfg_try = os.path.join("configs", cfg_arg)
    if os.path.exists(cfg_try):
        return os.path.abspath(cfg_try)

    # 3) fallback (will error later in loader)
    return os.path.abspath(cfg_arg)


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


def maybe_tqdm(cfg, it, total=None, desc=None):
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    if getattr(cfg, "RUNTIME", {}).get("use_tqdm", True) and (tqdm is not None):
        return tqdm(it, total=total, desc=desc)
    return it


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


def enabled_inputs(cfg):
    return [k for k, v in cfg.INPUTS.items() if v]


def enabled_algos_with_cfg(cfg):
    return [(name, a) for name, a in cfg.ALGORITHMS.items() if a.get("enabled", False)]


def enabled_algo_names(cfg):
    return [name for name, _ in enabled_algos_with_cfg(cfg)]


def source_branch_list(cfg):
    """
    Branches needed from the original input ntuple for run_clustering.
    """
    bl = []

    for _, cmap in cfg.BRANCHES["cands"].items():
        for _, br in cmap.items():
            if br:
                bl.append(br)

    for _, br in cfg.BRANCHES["genjets"].items():
        if br:
            bl.append(br)

    vtx = cfg.BRANCHES.get("vtx", {})
    for k in ("z_gen", "z_reco", "reco_sumpt", "n_reco"):
        if vtx.get(k):
            bl.append(vtx[k])

    return sorted(set(bl))


def snapshot_tree_name(cfg):
    return getattr(cfg, "SNAPSHOT_TREE_NAME", "Events")


def snapshot_file_path(cfg, cfg_tag: str, proc: str) -> str:
    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    return os.path.join(out_root, proc, "snapshot", "clustered_events.root")


def cache_dir_path(cfg, cfg_tag: str, proc: str) -> str:
    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    return os.path.join(out_root, proc, "cache")


def plots_dir_path(cfg, cfg_tag: str, proc: str) -> str:
    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    return os.path.join(out_root, proc, "plots")


def snapshot_branch_name_cand(inp: str, field: str) -> str:
    return f"{inp}_cand_{field}"


def snapshot_branch_name_algo(inp: str, algo: str, field: str) -> str:
    return f"{inp}__{algo}__{field}"


def snapshot_branch_list(cfg):
    """
    Branches needed from the intermediate clustered snapshot for run_studies / run_plotting.
    """
    bl = ["event_idx"]

    gj = cfg.BRANCHES["genjets"]
    if gj.get("pt"):
        bl.append("GenJet_pt")
    if gj.get("eta"):
        bl.append("GenJet_eta")
    if gj.get("phi"):
        bl.append("GenJet_phi")
    if gj.get("mass"):
        bl.append("GenJet_mass")

    vtx = cfg.BRANCHES.get("vtx", {})
    if vtx.get("z_gen"):
        bl.append("GenVtx_z")
    if vtx.get("z_reco"):
        bl.append("L1Vtx_z")
    if vtx.get("reco_sumpt"):
        bl.append("L1Vtx_sumpt")
    if vtx.get("n_reco"):
        bl.append("nL1Vtx")

    for inp in enabled_inputs(cfg):
        cdef = cfg.BRANCHES["cands"][inp]

        bl.extend([
            snapshot_branch_name_cand(inp, "pt"),
            snapshot_branch_name_cand(inp, "eta"),
            snapshot_branch_name_cand(inp, "phi"),
        ])

        if cdef.get("mass"):
            bl.append(snapshot_branch_name_cand(inp, "mass"))
        if cdef.get("charge"):
            bl.append(snapshot_branch_name_cand(inp, "charge"))
        if cdef.get("abs_pdgid") or cdef.get("pdgId"):
            bl.append(snapshot_branch_name_cand(inp, "abs_pdgid"))

        for algo, _ in enabled_algos_with_cfg(cfg):
            bl.extend([
                snapshot_branch_name_algo(inp, algo, "jet_pt"),
                snapshot_branch_name_algo(inp, algo, "jet_eta"),
                snapshot_branch_name_algo(inp, algo, "jet_phi"),
                snapshot_branch_name_algo(inp, algo, "cand_jetIdx"),
                snapshot_branch_name_algo(inp, algo, "cand_isSeed"),
            ])
            # optional
            bl.append(snapshot_branch_name_algo(inp, algo, "jet_mass"))

    return sorted(set(bl))