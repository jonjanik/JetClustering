# example_config.py 
import numpy as np

# -----------------------------
# Processes (samples)
# -----------------------------
PROCESSES = {
    "ttbar": {
        "path": "data/TT_PU200/l1Nano_merged.root",
        "label": r"$\mathrm{t}\bar{\mathrm{t}}$",
    },
    "caseA": {
        "path": "data/caseA/l1Nano_merged.root",
        "label": r"MSJ case A",
    },
}

TREE_NAME = "Events"

# -----------------------------
# Runtime control
# -----------------------------
RUNTIME = {
    "max_events": None,
    "event_sampling": "head",
    "stride": 1,
    "use_tqdm": True,
    "cache_format": "npz",
}

# -----------------------------
# Branch mapping
# -----------------------------
BRANCHES = {
    "cands": {
        "PF": {
            "pt": "PF_pt",
            "eta": "PF_eta",
            "phi": "PF_phi",
            "mass": "PF_mass",
        },
        "Puppi": {
            "pt": "Puppi_pt",
            "eta": "Puppi_eta",
            "phi": "Puppi_phi",
            "mass": "Puppi_mass",
        },
        "PFExtended": {
            "pt": "PFExtended_pt",
            "eta": "PFExtended_eta",
            "phi": "PFExtended_phi",
            "mass": "PFExtended_mass",
        },
        "PuppiExtended": {
            "pt": "PuppiExtended_pt",
            "eta": "PuppiExtended_eta",
            "phi": "PuppiExtended_phi",
            "mass": "PuppiExtended_mass",
        },
    },

    "genjets": {
        "pt": "GenJet_pt",
        "eta": "GenJet_eta",
        "phi": "GenJet_phi",
        "mass": "GenJet_mass",
    },

    "vtx": {
        "z_gen": "GenVtx_z",
        "z_reco": "L1Vtx_z",
        "reco_sumpt": "L1Vtx_sumpt",
        "n_reco": "nL1Vtx",
        "reco_strategy": "max_sumpt",
    },
}

# -----------------------------
# Inputs to run
# -----------------------------
INPUTS = {
    "PF": True,
    "Puppi": True,
    "PFExtended": False,
    "PuppiExtended": False,
}

# -----------------------------
# Jet algorithms
# -----------------------------
ALGORITHMS = {
    "AntiKt": {   #reference
        "enabled": True,
        "fn": "antikt",
        "params": {"R_clu": 0.4, "pt_min": 0.0},
        "label": r"anti-k_{\mathrm{T}}",
    },

    # SC family
    "SeededConeGreedy": {   #greedy (iterative) constituent assignment!
        "enabled": False,
        "fn": "seeded_cone_greedy",
        "params": {"R_clu": 0.4, "nseeds": 1024},
        "label": "SeededConeGreedy",
    },
    "SeededConeNMS": {
        "enabled": True,
        "fn": "seeded_cone_nms",
        "params": {"R_seed": 0.33, "R_cen": 0.4, "R_clu": 0.4},
        "label": "SeededConeNMS",
    },
    "SeededConeNMSWeighted": {
        "enabled": True,
        "fn": "seeded_cone_nms_weighted",
        "params": {"R_seed": 0.3, "R_cen": 0.4, "R_clu": 0.4, "alpha_seed": 2.0},
        "label": "SeededConeNMSWeighted",
    },

    # Simplified CLUE-like "nearest higher-pT link" family
    "LinkTree": {
        "enabled": True,
        "fn": "linktree",
        "params": {"R_link": 0.3, "pt_min": 0.0},
        "label": "LinkTree",
    },
    "LinkSeedCone": {   #greedy (iterative) constituent assignment!
        "enabled": False,
        "fn": "linkseed_cone",
        "params": {"R_link": 0.3, "R_jet": 0.4, "pt_min": 0.0},
        "label": "LinkSeedCone",
    },
    "LinkBaryCone": {   #greedy (iterative) constituent assignment!
        "enabled": False,
        "fn": "linkbary_cone",
        "params": {"R_link": 0.3, "R_jet": 0.4, "pt_min": 0.0},
        "label": "LinkBaryCone",
    },
    "LinkCentroidNearest": {
        "enabled": True,
        "fn": "link_centroid_nearest",
        "params": {"R_link": 0.3, "R_cen": 0.4, "R_clu": 0.4, "pt_min": 0.0},
        "label": "LinkCentroidNearest",
    },
    "LinkCentroidNearestWeighted": {
        "enabled": True,
        "fn": "link_centroid_nearest_weighted",
        "params": {
            "R_link": 0.3, "R_cen": 0.4, "R_clu": 0.4, "pt_min": 0.0,
            "alpha_seed": 2.0, "seed_strength": "cone_sumpt"
        },
        "label": "LinkCentroidNearestWeighted",
    },
    "LinkBaryNearestWeighted": {
        "enabled": True,
        "fn": "linkbary_nearest_weighted",
        "params": {
            "R_link": 0.3, "R_jet": 0.4, "pt_min": 0.0,
            "alpha_seed": 2.0, "strength_mode": "proto_sumpt"
        },
        "label": "LinkBaryNearestWeighted",
    },
}

# -----------------------------
# Studies (plotting stage)
#   - Existing keys kept as-is.
#   - New diagnostics are additive and OFF by default unless set True.
# -----------------------------
STUDIES = {
    "efficiency": True,
    "response_ridgeline": True,
    "ak4_agreement": False,
    "violin_response": False,   # redundant

    # reco-side / rate-proxy / stability diagnostics
    "purity_vs_ptreco": True,           # reco->gen match fraction vs pT^RECO
    "fake_rate_vs_ptreco": False,       # 1 - purity
    "njet_multiplicity": True,         # per-event N(jets) above thresholds
    "ht_distributions": True,          # per-event HT above thresholds

    # response summaries (median + quantiles) vs pT^GEN
    "response_quantiles": False,

    # dr summaries (quantiles) vs pT^GEN
    "dr_quantiles": False,

    # trigger turn-ons: P(pT^RECO > T | GEN jet in bin)
    "turnons": False,

    # seed statistics (for algos that produce seed_mask)
    "seed_stats": False,

    # F-score vs threshold 
    "f1": False,

    # ak compatibility
    "ak_compat": True,
}

AK_COMPAT = {
    "enabled": True,

    # Reference algorithm name (per INPUT). Must be enabled in ALGORITHMS.
    "ref_algo": "AntiKt",

    "dR_match": 0.3,

    # pT binning on reference jet pT (AK for that same input)
    "pt_bins": np.array([0, 20, 40, 60, 80, 100, 150, 200, 300], dtype=float),

    # "iou" (intersection over union) | "f_ref" | "f_alt"
    "metric": "iou",
}

# -----------------------------
# Matching / selections
# -----------------------------
MATCHING = {
    "dR_match": 0.3,
    "pt_gen_min": 0.0,
    "pt_reco_min": 0.0,
}

# -----------------------------
# Region split (optional) - ALWAYS includes Inclusive
# -----------------------------
REGION_SPLIT = {
    "enabled": True,
    "definitions": {
        "Inclusive": lambda eta: np.ones_like(eta, dtype=bool),
        "Barrel":    lambda eta: (np.abs(eta) < 1.52),
        "Endcap":    lambda eta: (np.abs(eta) >= 1.52) & (np.abs(eta) < 2.4),
    }
}

# -----------------------------
# Z split (optional)
# -----------------------------
Z_SPLIT = {
    "enabled": True,
    "dz_cm": 1.0,
}

# -----------------------------
# Per-task pt binnings
# -----------------------------
PT_BINS = {
    "efficiency": np.linspace(0, 100, 41),
    "ridgeline":  np.array([0, 20, 40, 60, 80, 100, 150, 200, 300]),
    "violin":     np.array([0, 20, 40, 60, 80, 100, 150, 200, 300]),

    # pT^RECO binning for purity/fake
    "purity":     np.linspace(0, 300, 41),

    # thresholds for multiplicity/HT and turn-ons
    "jet_thresholds": np.array([0, 10, 20, 30, 40, 50], dtype=float),
    "ht_thresholds":  np.array([0, 10, 20, 30, 40, 50], dtype=float),
    "turnon_thresholds": np.array([30, 40, 50], dtype=float),

    # pT^GEN binning for quantile summaries / turn-ons (reuse efficiency bins by default)
    "summary_gen": np.linspace(0, 300, 31),

    "f1_thresholds": np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float),
}

# -----------------------------
# Output directories
# -----------------------------
OUTDIR = "outputs"

PLOT_LABELS = {
    "llabel": "Phase-2 Simulation Preliminary",
    "rlabel": "PU 200 (14 TeV)",
}

# -----------------------------
# Style configuration for efficiency curves
# -----------------------------
EFF_STYLE_MAP = {}

EFF_STYLE_DEFAULTS = [
    {"marker": "o", "color": "black", "mfc": "none"},
    {"marker": "*", "color": "violet", "mfc": "none"},
    {"marker": "s", "color": "red", "mfc": "none"},
    {"marker": "v", "color": "orange", "mfc": "none"},
    {"marker": "^", "color": "blue", "mfc": "none"},
    {"marker": "D", "color": "green", "mfc": "none"},
    {"marker": "+", "color": "cyan", "mfc": "none"},
    {"marker": "P", "color": "yellow", "mfc": "none"},
    {"marker": "X", "color": "pink", "mfc": "none"},
]

def algo_family_name(algo_name: str) -> str:
    if algo_name.startswith("SeededConeGreedy"):
        return "SeededConeGreedy"
    return algo_name

def build_eff_style_key_map():
    key_map = {}
    algo_names = list(ALGORITHMS.keys())
    input_names = list(INPUTS.keys())

    for inp in input_names:
        for aname in algo_names:
            fam = algo_family_name(aname)
            st = EFF_STYLE_MAP.get((inp, aname), None)
            if st is None:
                st = EFF_STYLE_MAP.get((inp, fam), None)
            if st is not None:
                key_map[(inp, aname)] = dict(st)
    return key_map

EFF_STYLE_KEY_MAP = build_eff_style_key_map()
