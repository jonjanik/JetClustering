# run_clustering.py
import os
import argparse
import numpy as np
import awkward as ak

from src.config_utils import (
    load_cfg_from_path,
    config_tag_from_path,
    enabled_inputs,
    enabled_algos_with_cfg,
    maybe_tqdm,
    select_event_indices,
    source_branch_list,
    snapshot_tree_name,
    snapshot_file_path,
    snapshot_branch_name_cand,
    snapshot_branch_name_algo,
    resolve_config_path
)
from src.utils import (
    ensure_dir,
    load_arrays,
    write_root_tree,
    call_algo_with_supported_kwargs,
    wrap_phi_np,
)
from src.clustering_algorithms import ALGO_REGISTRY


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run clustering once and write an intermediate clustered ROOT snapshot."
    )
    ap.add_argument(
        "--config", "-c", default="config.py",
        help="Config file (auto-resolves from configs/ if not found locally)"
    )
    return ap.parse_args()


def _append_optional_branch(columns, branch_name, value):
    if branch_name not in columns:
        columns[branch_name] = []
    columns[branch_name].append(value)


def run(cfg, cfg_tag: str):
    inputs = enabled_inputs(cfg)
    algos = enabled_algos_with_cfg(cfg)

    out_root = os.path.join(getattr(cfg, "OUTDIR", "outputs"), cfg_tag)
    ensure_dir(out_root)

    algo_fns = {aname: ALGO_REGISTRY[acfg["fn"]] for aname, acfg in algos}
    algo_params = {aname: acfg.get("params", {}) for aname, acfg in algos}

    for proc, pinfo in cfg.PROCESSES.items():
        src_path = pinfo["path"]
        out_snapshot = snapshot_file_path(cfg, cfg_tag, proc)
        ensure_dir(os.path.dirname(out_snapshot))

        print(f"\n=== CLUSTERING: {proc} | input: {src_path} | config: {cfg_tag} ===")

        data = load_arrays(src_path, cfg.TREE_NAME, source_branch_list(cfg), library="ak")
        n_total = len(data[cfg.BRANCHES["genjets"]["pt"]])
        ev_idx = select_event_indices(cfg, n_total)

        print(f"Loaded {n_total} events (clustering {len(ev_idx)})")

        # -----------------------------
        # Build output columns
        # -----------------------------
        cols = {
            "event_idx": [],
            "GenJet_pt": [],
            "GenJet_eta": [],
            "GenJet_phi": [],
        }

        if cfg.BRANCHES["genjets"].get("mass"):
            cols["GenJet_mass"] = []

        vtx = cfg.BRANCHES.get("vtx", {})
        if vtx.get("z_gen"):
            cols["GenVtx_z"] = []
        if vtx.get("z_reco"):
            cols["L1Vtx_z"] = []
        if vtx.get("reco_sumpt"):
            cols["L1Vtx_sumpt"] = []
        if vtx.get("n_reco"):
            cols["nL1Vtx"] = []

        for inp in inputs:
            cdef = cfg.BRANCHES["cands"][inp]
            cols[snapshot_branch_name_cand(inp, "pt")] = []
            cols[snapshot_branch_name_cand(inp, "eta")] = []
            cols[snapshot_branch_name_cand(inp, "phi")] = []

            if cdef.get("mass"):
                cols[snapshot_branch_name_cand(inp, "mass")] = []
            if cdef.get("charge"):
                cols[snapshot_branch_name_cand(inp, "charge")] = []
            if cdef.get("abs_pdgid") or cdef.get("pdgId"):
                cols[snapshot_branch_name_cand(inp, "abs_pdgid")] = []

            for aname, _ in algos:
                cols[snapshot_branch_name_algo(inp, aname, "jet_pt")] = []
                cols[snapshot_branch_name_algo(inp, aname, "jet_eta")] = []
                cols[snapshot_branch_name_algo(inp, aname, "jet_phi")] = []
                cols[snapshot_branch_name_algo(inp, aname, "jet_mass")] = []
                cols[snapshot_branch_name_algo(inp, aname, "cand_jetIdx")] = []
                cols[snapshot_branch_name_algo(inp, aname, "cand_isSeed")] = []

        # -----------------------------
        # Event loop
        # -----------------------------
        for ievt in maybe_tqdm(cfg, ev_idx, total=len(ev_idx), desc=f"{proc}: cluster"):
            ievt = int(ievt)

            # ---- original event index
            cols["event_idx"].append(ievt)

            # ---- truth
            gen_pt = ak.to_numpy(data[cfg.BRANCHES["genjets"]["pt"]][ievt])
            gen_eta = ak.to_numpy(data[cfg.BRANCHES["genjets"]["eta"]][ievt])
            gen_phi = wrap_phi_np(ak.to_numpy(data[cfg.BRANCHES["genjets"]["phi"]][ievt]))
            cols["GenJet_pt"].append(gen_pt.astype(np.float32))
            cols["GenJet_eta"].append(gen_eta.astype(np.float32))
            cols["GenJet_phi"].append(gen_phi.astype(np.float32))

            if cfg.BRANCHES["genjets"].get("mass"):
                gen_mass = ak.to_numpy(data[cfg.BRANCHES["genjets"]["mass"]][ievt])
                cols["GenJet_mass"].append(np.asarray(gen_mass, dtype=np.float32))

            # ---- vertex info
            if vtx.get("z_gen"):
                cols["GenVtx_z"].append(float(np.asarray(data[vtx["z_gen"]][ievt]).reshape(-1)[0]))
            if vtx.get("z_reco"):
                cols["L1Vtx_z"].append(np.asarray(ak.to_numpy(data[vtx["z_reco"]][ievt]), dtype=np.float32))
            if vtx.get("reco_sumpt"):
                cols["L1Vtx_sumpt"].append(np.asarray(ak.to_numpy(data[vtx["reco_sumpt"]][ievt]), dtype=np.float32))
            if vtx.get("n_reco"):
                cols["nL1Vtx"].append(int(np.asarray(data[vtx["n_reco"]][ievt]).reshape(-1)[0]))

            # ---- candidates and clustering outputs
            for inp in inputs:
                cdef = cfg.BRANCHES["cands"][inp]

                cpt = ak.to_numpy(data[cdef["pt"]][ievt]).astype(np.float32)
                ceta = ak.to_numpy(data[cdef["eta"]][ievt]).astype(np.float32)
                cphi = wrap_phi_np(ak.to_numpy(data[cdef["phi"]][ievt])).astype(np.float32)

                cols[snapshot_branch_name_cand(inp, "pt")].append(cpt)
                cols[snapshot_branch_name_cand(inp, "eta")].append(ceta)
                cols[snapshot_branch_name_cand(inp, "phi")].append(cphi)

                cmass = None
                cchg = None
                cpid = None

                if cdef.get("mass"):
                    cmass = ak.to_numpy(data[cdef["mass"]][ievt]).astype(np.float32)
                    cols[snapshot_branch_name_cand(inp, "mass")].append(cmass)

                if cdef.get("charge"):
                    cchg = ak.to_numpy(data[cdef["charge"]][ievt])
                    cols[snapshot_branch_name_cand(inp, "charge")].append(np.asarray(cchg))

                if cdef.get("abs_pdgid"):
                    cpid = np.abs(ak.to_numpy(data[cdef["abs_pdgid"]][ievt])).astype(np.int32)
                    cols[snapshot_branch_name_cand(inp, "abs_pdgid")].append(cpid)
                elif cdef.get("pdgId"):
                    cpid = np.abs(ak.to_numpy(data[cdef["pdgId"]][ievt])).astype(np.int32)
                    cols[snapshot_branch_name_cand(inp, "abs_pdgid")].append(cpid)

                extra_kwargs = {
                    "mass": cmass,
                    "charge": cchg,
                    "abs_pdgid": cpid,
                }

                for aname, _ in algos:
                    jets, assign, seed_mask = call_algo_with_supported_kwargs(
                        algo_fns[aname],
                        ceta, cphi, cpt,
                        extra_kwargs=extra_kwargs,
                        algo_kwargs=algo_params[aname],
                    )

                    rpt, reta, rphi, rmass = jets

                    rpt = np.asarray(rpt, dtype=np.float32)
                    reta = np.asarray(reta, dtype=np.float32)
                    rphi = wrap_phi_np(np.asarray(rphi, dtype=float)).astype(np.float32)

                    if rmass is None:
                        rmass = np.zeros_like(rpt, dtype=np.float32)
                    else:
                        rmass = np.asarray(rmass, dtype=np.float32)
                        if rmass.shape != rpt.shape:
                            rmass = np.zeros_like(rpt, dtype=np.float32)

                    assign = np.asarray(assign, dtype=np.int32)
                    if assign.shape[0] != cpt.shape[0]:
                        raise RuntimeError(
                            f"{proc} event {ievt}: assign length mismatch for {inp}/{aname}: "
                            f"{assign.shape[0]} != {cpt.shape[0]}"
                        )

                    if seed_mask is None:
                        seed_mask = np.zeros(cpt.shape[0], dtype=np.bool_)
                    else:
                        seed_mask = np.asarray(seed_mask, dtype=np.bool_)
                        if seed_mask.shape[0] != cpt.shape[0]:
                            seed_mask = np.zeros(cpt.shape[0], dtype=np.bool_)

                    cols[snapshot_branch_name_algo(inp, aname, "jet_pt")].append(rpt)
                    cols[snapshot_branch_name_algo(inp, aname, "jet_eta")].append(reta)
                    cols[snapshot_branch_name_algo(inp, aname, "jet_phi")].append(rphi)
                    cols[snapshot_branch_name_algo(inp, aname, "jet_mass")].append(rmass)
                    cols[snapshot_branch_name_algo(inp, aname, "cand_jetIdx")].append(assign)
                    cols[snapshot_branch_name_algo(inp, aname, "cand_isSeed")].append(seed_mask)

        write_root_tree(out_snapshot, snapshot_tree_name(cfg), cols)
        print(f"Written clustered snapshot: {out_snapshot}")

    print("\nAll clustering done.")


if __name__ == "__main__":
    args = parse_args()
    
    cfg_path = resolve_config_path(args.config)
    cfg = load_cfg_from_path(cfg_path)
    tag = config_tag_from_path(cfg_path)

    run(cfg, tag)