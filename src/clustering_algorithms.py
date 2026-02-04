# src/clustering.py
import numpy as np

# -------------------------
# Geometry helpers
# -------------------------
def delta_phi(a, b):
    return np.mod(a - b + np.pi, 2*np.pi) - np.pi

def deltaR(eta1, phi1, eta2, phi2):
    dphi = delta_phi(phi1, phi2)
    deta = eta1 - eta2
    return np.sqrt(deta*deta + dphi*dphi)

def wrap_phi(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi


# -------------------------
# utility: Jet builders (p4 sum)
# -------------------------
def to_p4(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([e, px, py, pz], axis=-1)

def from_p4(p4):
    e, px, py, pz = p4.T
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    eta = np.arcsinh(pz / np.maximum(pt, 1e-6))
    mass = np.sqrt(np.maximum(e**2 - (px**2 + py**2 + pz**2), 0))
    return pt, eta, phi, mass

def build_clusters(pt, eta, phi, assign, mass=None, max_daus=9999, default_mass=0.13957):
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    assign = np.asarray(assign, dtype=int)

    clusters = np.unique(assign[assign >= 0])   # unassigned -1
    jets_pt, jets_eta, jets_phi, jets_mass = [], [], [], []

    for c in clusters:
        mask = (assign == c)

        p_pt  = pt[mask]
        p_eta = eta[mask]
        p_phi = phi[mask]
        if mass is None:
            p_mass = np.full_like(p_pt, default_mass)
        else:
            p_mass = np.asarray(mass, dtype=float)[mask]

        order = np.argsort(p_pt)[::-1][:max_daus]
        p_pt   = p_pt[order]
        p_eta  = p_eta[order]
        p_phi  = p_phi[order]
        p_mass = p_mass[order]

        p4 = to_p4(p_pt, p_eta, p_phi, p_mass)
        tot = np.sum(p4, axis=0)

        jpt, jeta, jphi, jmass = from_p4(tot.reshape(1, 4))
        jets_pt.append(jpt[0])
        jets_eta.append(jeta[0])
        jets_phi.append(wrap_phi(jphi[0]))
        jets_mass.append(jmass[0])

    return (np.array(jets_pt), np.array(jets_eta), np.array(jets_phi), np.array(jets_mass))


# -------------------------
# utility: pT-weighted (eta,phi) centroid with circular mean in phi
# -------------------------
def weighted_centroid_eta_phi(eta, phi, w, eta0=None, phi0=None):
    """
    Returns (eta_c, phi_c). If (eta0, phi0) provided, uses delta_phi/relative sums
    for numerical stability; otherwise uses circular mean.
    """
    w = np.asarray(w, dtype=float)
    sw = float(np.sum(w))
    if sw <= 0.0:
        if eta0 is None or phi0 is None:
            return float(np.mean(eta) if len(eta) else 0.0), float(wrap_phi(np.mean(phi) if len(phi) else 0.0))
        return float(eta0), float(wrap_phi(phi0))

    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    # case 1: reference axis (e.g. seed axis) given; numerically stable
    if eta0 is not None and phi0 is not None:
        dEta = eta - float(eta0)
        dPhi = delta_phi(phi, float(phi0))
        # move axis by weighted average displacement
        eta_c = float(float(eta0) + np.sum(w * dEta) / sw)
        phi_c = float(wrap_phi(float(phi0) + np.sum(w * dPhi) / sw))
        return eta_c, phi_c

    # case 2: no reference axis given -> global circular mean for phi, standard mean for eta
    x = np.sum(w * np.cos(phi)) / sw
    y = np.sum(w * np.sin(phi)) / sw
    phi_c = float(wrap_phi(np.arctan2(y, x)))
    eta_c = float(np.sum(w * eta) / sw)
    return eta_c, phi_c


# -------------------------
# Algorithm 1: seeded cone greedy
# -------------------------
def seeded_cone_greedy(eta, phi, pt, R_clu=0.4, nseeds=16):
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    assigns = np.full(len(pt), -1, dtype=int)

    jpt  = []
    jeta = []
    jphi = []

    for it in range(int(nseeds)):
        pt_masked = np.where(assigns == -1, pt, 0.0)
        seed = int(np.argmax(pt_masked))
        if pt_masked[seed] <= 0:
            break

        drs = deltaR(eta[seed], phi[seed], eta, phi)
        mask = (drs < float(R_clu)) & (pt_masked > 0)

        w = pt[mask]
        sumpt = float(np.sum(w))
        if sumpt <= 0:
            assigns[seed] = it
            continue

        dEta = eta[mask] - eta[seed]
        dPhi = delta_phi(phi[mask], phi[seed])

        jpt.append(sumpt)
        jeta.append(float(eta[seed] + np.sum(w * dEta) / sumpt))
        jphi.append(float(wrap_phi(phi[seed] + np.sum(w * dPhi) / sumpt)))

        assigns[mask] = it

    jets = (np.array(jpt), np.array(jeta), np.array(jphi), np.zeros(len(jpt), dtype=float))
    seed_mask = np.zeros(len(pt), dtype=bool)
    return jets, assigns, seed_mask


# -------------------------
# Algorithm 2: seeded cone NMS + centroid + nearest axis assignment
# -------------------------
def seeded_cone_nms(eta, phi, pt, R_seed=0.2, R_cen=0.4, R_clu=0.4):
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)
    seed_mask = np.ones(N, dtype=bool)

    # local-max seed finding
    for i in range(N):
        if not seed_mask[i]:
            continue
        for j in range(N):
            if j == i:
                continue
            dr = deltaR(eta[i], phi[i], eta[j], phi[j])
            if dr < float(R_seed):
                if (pt[j] > pt[i]) or (pt[j] == pt[i] and j < i):
                    seed_mask[i] = False
                    break

    seed_idx = np.sort(np.where(seed_mask)[0])
    seed_to_cluster = {int(old): int(new) for new, old in enumerate(seed_idx)}

    cent_eta, cent_phi = [], []
    for s in seed_idx:
        drs = deltaR(eta[s], phi[s], eta, phi)
        mask = (drs < float(R_cen))

        w = pt[mask]
        sw = float(np.sum(w))
        if sw <= 0:
            cent_eta.append(float(eta[s]))
            cent_phi.append(float(wrap_phi(phi[s])))
            continue

        dEta = eta[mask] - eta[s]
        dPhi = delta_phi(phi[mask], phi[s])

        cent_eta.append(float(eta[s] + np.sum(w * dEta) / sw))
        cent_phi.append(float(wrap_phi(phi[s] + np.sum(w * dPhi) / sw)))

    cent_eta = np.array(cent_eta, dtype=float)
    cent_phi = np.array(cent_phi, dtype=float)

    assign = np.full(N, -1, dtype=int)
    for i in range(N):
        if cent_eta.size == 0:
            continue
        drs = deltaR(eta[i], phi[i], cent_eta, cent_phi)
        j = int(np.argmin(drs))
        if float(drs[j]) < float(R_clu):
            assign[i] = seed_to_cluster[int(seed_idx[j])]

    for s in seed_idx:
        assign[int(s)] = seed_to_cluster[int(s)]

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask

# -------------------------
# Algorithm 3: seeded cone NMS + centroid + nearest (normalized by pT) axis assignment
# -------------------------
def seeded_cone_nms_weighted(
    eta, phi, pt,
    R_seed=0.2, R_cen=0.4, R_clu=0.4,
    alpha_seed=2.0,
):
    """
    SeededConeNMS variant with pT-weighted assignment:
      - seeds: local maxima within R_seed
      - centroids: computed in R_cen around each seed
      - assignment: choose cluster j that minimizes score_ij = dr2 / pt_seed^alpha among dr<R_clu
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)

    # seed finding (local pT maxima)
    seed_mask = np.ones(N, dtype=bool)

    thr2_seed = float(R_seed) * float(R_seed)
    idx = np.arange(N, dtype=int)

    for i in range(N):
        if not seed_mask[i]:
            continue
        dphi = np.arctan2(np.sin(phi - phi[i]), np.cos(phi - phi[i]))
        deta = eta - eta[i]
        dr2 = deta * deta + dphi * dphi

        neigh = (dr2 < thr2_seed)
        neigh[i] = False
        if not np.any(neigh):
            continue

        jcand = idx[neigh]
        higher = (pt[jcand] > pt[i]) | ((pt[jcand] == pt[i]) & (jcand < i))
        if np.any(higher):
            seed_mask[i] = False

    seed_idx = np.sort(np.where(seed_mask)[0])
    seed_to_cluster = {old: new for new, old in enumerate(seed_idx)}


    # centroid per seed
    cent_eta, cent_phi = [], []
    thr2_cen = float(R_cen) * float(R_cen)

    for s in seed_idx:
        dphi = np.arctan2(np.sin(phi - phi[s]), np.cos(phi - phi[s]))
        deta = eta - eta[s]
        dr2 = deta * deta + dphi * dphi

        mask = (dr2 < thr2_cen)
        w = pt[mask]
        sw = float(np.sum(w))
        if sw <= 0.0:
            cent_eta.append(float(eta[s]))
            cent_phi.append(float(wrap_phi(phi[s])))
            continue

        eta_c, phi_c = weighted_centroid_eta_phi(eta[mask], phi[mask], w, eta0=eta[s], phi0=phi[s])
        cent_eta.append(float(eta_c))
        cent_phi.append(float(phi_c))

    cent_eta = np.asarray(cent_eta, dtype=float)
    cent_phi = np.asarray(cent_phi, dtype=float)

    # assign constituents
    assign = np.full(N, -1, dtype=int)
    if cent_eta.size == 0:
        jets = (np.array([]), np.array([]), np.array([]), np.array([]))
        return jets, assign, seed_mask

    thr2_clu = float(R_clu) * float(R_clu)

    seed_pt = pt[seed_idx].astype(float)
    denom = np.power(np.maximum(seed_pt, 1e-6), float(alpha_seed))

    for i in range(N):
        dphi = np.arctan2(np.sin(cent_phi - phi[i]), np.cos(cent_phi - phi[i]))
        deta = cent_eta - eta[i]
        dr2  = deta * deta + dphi * dphi

        ok = (dr2 < thr2_clu)
        if not np.any(ok):
            continue

        score = np.full_like(dr2, np.inf, dtype=float)
        score[ok] = dr2[ok] / denom[ok]

        j = int(np.argmin(score))

        min_score = score[j]
        ties = np.where(score == min_score)[0]
        # sort ties by dR, if also tied sort by idx
        if ties.size > 1:
            dr2_t = dr2[ties]
            j = int(ties[np.lexsort((ties, dr2_t))[0]])

        if ok[j]:
            assign[i] = seed_to_cluster[seed_idx[j]]

    for s in seed_idx:
        assign[s] = seed_to_cluster[s]

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


# -------------------------
# Algorithm 4: anti-kT via FastJet
# -------------------------
def antikt(eta, phi, pt, R_clu=0.4, pt_min=0.0):
    import fastjet

    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    mass = np.full_like(pt, 0.13957)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    constituents = [
        fastjet.PseudoJet(float(x), float(y), float(z), float(e))
        for (x, y, z, e) in zip(px, py, pz, E)
    ]
    for i, c in enumerate(constituents):
        c.set_user_index(i)

    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, float(R_clu))
    cluster = fastjet.ClusterSequence(constituents, jetdef)

    assign = np.full(len(pt), -1, dtype=int)

    jets = cluster.inclusive_jets(float(pt_min))
    jets = fastjet.sorted_by_pt(jets)

    NJ = len(jets)
    jpt = np.zeros(NJ)
    jeta = np.zeros(NJ)
    jphi = np.zeros(NJ)
    jmass = np.zeros(NJ)

    for i, jet in enumerate(jets):
        jpt[i] = jet.pt()
        jeta[i] = jet.eta()
        jphi[i] = wrap_phi(jet.phi())
        jmass[i] = jet.m()
        for dau in jet.constituents():
            assign[int(dau.user_index())] = int(i)

    seed_mask = np.zeros(len(pt), dtype=bool)
    return (jpt, jeta, jphi, jmass), assign, seed_mask


# ============================================================
# Simplified CLUE-like "nearest higher-pT link" family
# ============================================================

def _nearest_higher_parent_indices(eta, phi, pt, R_link=0.2, pt_min=0.0):
    """
    For each i, find parent(i) = nearest j with pt[j] > pt[i] within dR < R_link.
    If no such j exists, parent(i) = i (seed/root).
    Tie-breaking: if pt equal, treat smaller index as "higher" to avoid ambiguity.
    Returns:
      parent: int array length N; list of indices of parents for each cand
      seed_mask: bool array length N (True for roots)
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)

    parent = np.arange(N, dtype=int)
    seed_mask = np.zeros(N, dtype=bool)

    active = (pt >= float(pt_min))
    thr2 = float(R_link) * float(R_link)
    idx = np.arange(N, dtype=int)

    for i in range(N):
        if not active[i]:
            parent[i] = -1
            continue

        dphi = np.arctan2(np.sin(phi - phi[i]), np.cos(phi - phi[i]))
        deta = eta - eta[i]
        dr2 = deta * deta + dphi * dphi

        neigh = (dr2 < thr2)
        neigh[i] = False

        if not np.any(neigh):
            parent[i] = i
            seed_mask[i] = True
            continue

        higher = neigh & ((pt > pt[i]) | ((pt == pt[i]) & (idx < i)))
        if not np.any(higher):
            parent[i] = i
            seed_mask[i] = True
            continue

        # nearest higher pT neighbor = parent
        j = int(np.argmin(np.where(higher, dr2, np.inf)))
        parent[i] = j

    return parent, seed_mask


def _root_assign_from_parent(parent):
    """
    Given parent pointers with parent[i] in [0..N-1] or -1 (ignored),
    compute root id per particle by pointer chasing.
    Returns:
      root_idx: length N, values are root indices (original indices) or -1
      roots: sorted unique list of root indices
      assign: length N, cluster id in [0..nroots-1] or -1
    """
    parent = np.asarray(parent, dtype=int)
    N = len(parent)

    root_idx = np.full(N, -1, dtype=int)

    for i in range(N):
        if parent[i] < 0:
            continue
        j = i
        # chase to fixed point
        while True:
            pj = parent[j]
            if pj < 0:
                root_idx[i] = -1
                break
            if pj == j:
                root_idx[i] = j
                break
            j = pj

    roots = np.unique(root_idx[root_idx >= 0])   # -1 unassigned
    roots = np.sort(roots)   # sorted list of unique root particle indices

    root_to_cid = {int(r): k for k, r in enumerate(roots)}   # dict {root_index: cluster_id}
    assign = np.full(N, -1, dtype=int)
    for i in range(N):
        r = int(root_idx[i])
        if r >= 0:
            assign[i] = int(root_to_cid[r])

    return root_idx, roots, assign


def _greedy_cone_assignment(eta, phi, pt, centers_eta, centers_phi, centers_pt,
                           R_jet=0.4, pt_min=0.0):
    """
    Assign each particle to at most one cone, processing cones in descending centers_pt.
    A particle is assigned to the first cone that contains it (greedy), stable/deterministic.
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    centers_eta = np.asarray(centers_eta, dtype=float)
    centers_phi = np.asarray(centers_phi, dtype=float)
    centers_pt  = np.asarray(centers_pt,  dtype=float)

    N = len(pt)
    M = len(centers_pt)
    assign = np.full(N, -1, dtype=int)

    if M == 0:
        return assign

    order = np.argsort(centers_pt)[::-1]
    thr2 = float(R_jet) * float(R_jet)

    active = (pt >= float(pt_min))

    for cid_new, k in enumerate(order):
        ceta = centers_eta[k]
        cphi = centers_phi[k]

        dphi = np.arctan2(np.sin(phi - cphi), np.cos(phi - cphi))
        deta = eta - ceta
        dr2 = deta * deta + dphi * dphi

        inside = (dr2 < thr2) & active & (assign < 0)
        assign[inside] = cid_new

    return assign


# -------------------------
# Algorithm 5: LinkTree (proto-clusters are jets)
# -------------------------
def linktree(eta, phi, pt, R_link=0.2, pt_min=0.0):
    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    _, _, assign = _root_assign_from_parent(parent)

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(np.asarray(pt), np.asarray(eta), np.asarray(phi), assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


# -------------------------
# Algorithm 6: Link -> seed -> cone around seed (greedy)
# -------------------------
def linkseed_cone(eta, phi, pt, R_link=0.2, R_jet=0.4, pt_min=0.0):
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    _, roots, _ = _root_assign_from_parent(parent)

    centers_eta = eta[roots] if len(roots) else np.array([], dtype=float)
    centers_phi = phi[roots] if len(roots) else np.array([], dtype=float)
    centers_pt  = pt[roots]  if len(roots) else np.array([], dtype=float)

    assign = _greedy_cone_assignment(
        eta, phi, pt,
        centers_eta, centers_phi, centers_pt,
        R_jet=R_jet,
        pt_min=pt_min
    )

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


# -------------------------
# Algorithm 7: Link -> barycenter -> cone around barycenter (greedy)
# -------------------------
def linkbary_cone(eta, phi, pt, R_link=0.2, R_jet=0.4, pt_min=0.0):
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    _, roots, proto_assign = _root_assign_from_parent(parent)

    if len(roots) == 0:
        assign = np.full(len(pt), -1, dtype=int)
        jets = (np.array([]), np.array([]), np.array([]), np.array([]))
        return jets, assign, seed_mask

    nclus = int(np.max(proto_assign)) + 1 if np.any(proto_assign >= 0) else 0
    centers_eta = np.zeros(nclus, dtype=float)
    centers_phi = np.zeros(nclus, dtype=float)
    centers_pt  = np.zeros(nclus, dtype=float)

    for cid in range(nclus):
        mask = (proto_assign == cid)
        if not np.any(mask):
            centers_eta[cid] = 0.0
            centers_phi[cid] = 0.0
            centers_pt[cid]  = 0.0
            continue

        w = pt[mask]
        centers_pt[cid] = float(np.max(w))  # greedy ordering; could also use sumpt
        eta_c, phi_c = weighted_centroid_eta_phi(eta[mask], phi[mask], w)
        centers_eta[cid] = float(eta_c)
        centers_phi[cid] = float(phi_c)

    assign = _greedy_cone_assignment(
        eta, phi, pt,
        centers_eta, centers_phi, centers_pt,
        R_jet=R_jet,
        pt_min=pt_min
    )

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask

# -------------------------
# Algorithm 8: Link -> cone around seed -> centroid -> nearest axis assignment
# -------------------------
def link_centroid_nearest(eta, phi, pt, R_link=0.2, R_cen=0.4, R_clu=0.4, pt_min=0.0):
    """
    Link-based analogue of SeededConeNMS:

      1) Seeds: roots via linking within R_link (seed if no higher-pT neighbor in R_link)
      2) Centroids: pT-weighted centroid from particles within R_cen of each seed direction
      3) Assignment: each particle assigned to nearest centroid within R_clu (no greedy ordering)
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)
    if N == 0:
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, np.array([], dtype=int), np.array([], dtype=bool)

    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    seed_idx = np.sort(np.where(seed_mask)[0])

    if seed_idx.size == 0:
        assign = np.full(N, -1, dtype=int)
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, assign, seed_mask

    cent_eta = np.zeros(len(seed_idx), dtype=float)
    cent_phi = np.zeros(len(seed_idx), dtype=float)

    thr2_cen = float(R_cen) * float(R_cen)
    active = (pt >= float(pt_min))

    for k, s in enumerate(seed_idx):
        dphi = np.arctan2(np.sin(phi - phi[s]), np.cos(phi - phi[s]))
        deta = eta - eta[s]
        dr2  = deta * deta + dphi * dphi

        mask = (dr2 < thr2_cen) & active
        if not np.any(mask):
            cent_eta[k] = float(eta[s])
            cent_phi[k] = float(wrap_phi(phi[s]))
            continue

        w = pt[mask]
        eta_c, phi_c = weighted_centroid_eta_phi(eta[mask], phi[mask], w, eta0=eta[s], phi0=phi[s])
        cent_eta[k] = float(eta_c)
        cent_phi[k] = float(phi_c)

    thr2_clu = float(R_clu) * float(R_clu)
    assign = np.full(N, -1, dtype=int)

    for i in range(N):
        if not active[i]:
            continue
        dphi = np.arctan2(np.sin(cent_phi - phi[i]), np.cos(cent_phi - phi[i]))
        deta = cent_eta - eta[i]
        dr2  = deta * deta + dphi * dphi

        j = int(np.argmin(dr2))
        if float(dr2[j]) < thr2_clu:
            assign[i] = j

    for k, s in enumerate(seed_idx):
        assign[s] = k

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


# -------------------------
# Algorithm 9: Link -> cone around seed -> centroid -> nearest (normalized by pT) axis assignment
# -------------------------
def link_centroid_nearest_weighted(
    eta, phi, pt,
    R_link=0.2, R_cen=0.4, R_clu=0.4,
    pt_min=0.0,
    alpha_seed=2.0,
    seed_strength="seed_pt",
):
    """
    Link-based analogue of SeededConeNMS with pT-weighted assignment.
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)
    active = (pt >= float(pt_min))

    if N == 0:
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, np.array([], dtype=int), np.array([], dtype=bool)

    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    seed_idx = np.sort(np.where(seed_mask)[0])

    if seed_idx.size == 0:
        assign = np.full(N, -1, dtype=int)
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, assign, seed_mask

    M = len(seed_idx)
    cent_eta = np.zeros(M, dtype=float)
    cent_phi = np.zeros(M, dtype=float)

    thr2_cen = float(R_cen) * float(R_cen)
    strength = np.zeros(M, dtype=float)

    for k, s in enumerate(seed_idx):
        dphi = np.arctan2(np.sin(phi - phi[s]), np.cos(phi - phi[s]))
        deta = eta - eta[s]
        dr2  = deta * deta + dphi * dphi

        mask = (dr2 < thr2_cen) & active
        if not np.any(mask):
            cent_eta[k] = float(eta[s])
            cent_phi[k] = float(wrap_phi(phi[s]))
            strength[k] = float(pt[s])
            continue

        w = pt[mask]
        eta_c, phi_c = weighted_centroid_eta_phi(eta[mask], phi[mask], w, eta0=eta[s], phi0=phi[s])
        cent_eta[k] = float(eta_c)
        cent_phi[k] = float(phi_c)

        if seed_strength == "cone_sumpt":
            strength[k] = float(np.sum(w))
        else:
            strength[k] = float(pt[s])

    denom = np.power(np.maximum(strength, 1e-6), float(alpha_seed))

    assign = np.full(N, -1, dtype=int)
    thr2_clu = float(R_clu) * float(R_clu)

    for i in range(N):
        if not active[i]:
            continue

        dphi = np.arctan2(np.sin(cent_phi - phi[i]), np.cos(cent_phi - phi[i]))
        deta = cent_eta - eta[i]
        dr2  = deta * deta + dphi * dphi

        ok = (dr2 < thr2_clu)
        if not np.any(ok):
            continue

        score = np.full_like(dr2, np.inf, dtype=float)
        score[ok] = dr2[ok] / denom[ok]

        j = int(np.argmin(score))

        min_score = score[j]
        ties = np.where(score == min_score)[0]
        if ties.size > 1:
            dr2_t = dr2[ties]
            j = int(ties[np.lexsort((ties, dr2_t))[0]])

        if ok[j]:
            assign[i] = j

    for k, s in enumerate(seed_idx):
        assign[s] = k

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


# -------------------------
# Algorithm 10: Link -> proto-cluster -> barycenter -> nearest (normalized by pT) assignment
# -------------------------
def linkbary_nearest_weighted(
    eta, phi, pt,
    R_link=0.2, R_jet=0.4,
    pt_min=0.0,
    alpha_seed=2.0,
    strength_mode="proto_sumpt",
):
    """
    Link -> proto-clusters -> barycenters -> weighted (non-greedy) assignment in R_jet.
    """
    eta = np.asarray(eta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    pt  = np.asarray(pt,  dtype=float)

    N = len(pt)
    active = (pt >= float(pt_min))

    if N == 0:
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, np.array([], dtype=int), np.array([], dtype=bool)

    parent, seed_mask = _nearest_higher_parent_indices(eta, phi, pt, R_link=R_link, pt_min=pt_min)
    _, roots, proto_assign = _root_assign_from_parent(parent)

    if len(roots) == 0 or not np.any(proto_assign >= 0):
        assign = np.full(N, -1, dtype=int)
        empty = (np.array([]), np.array([]), np.array([]), np.array([]))
        return empty, assign, seed_mask

    nclus = int(np.max(proto_assign)) + 1

    centers_eta = np.zeros(nclus, dtype=float)
    centers_phi = np.zeros(nclus, dtype=float)
    strength    = np.zeros(nclus, dtype=float)

    for cid in range(nclus):
        mask = (proto_assign == cid)
        if not np.any(mask):
            centers_eta[cid] = 0.0
            centers_phi[cid] = 0.0
            strength[cid] = 0.0
            continue

        w = pt[mask]
        eta_c, phi_c = weighted_centroid_eta_phi(eta[mask], phi[mask], w)
        centers_eta[cid] = float(eta_c)
        centers_phi[cid] = float(phi_c)

        if strength_mode == "proto_maxpt":
            strength[cid] = float(np.max(w))
        else:
            strength[cid] = float(np.sum(w))

    denom = np.power(np.maximum(strength, 1e-6), float(alpha_seed))
    thr2 = float(R_jet) * float(R_jet)

    assign = np.full(N, -1, dtype=int)
    for i in range(N):
        if not active[i]:
            continue

        dphi = np.arctan2(np.sin(centers_phi - phi[i]), np.cos(centers_phi - phi[i]))
        deta = centers_eta - eta[i]
        dr2  = deta * deta + dphi * dphi

        ok = (dr2 < thr2)
        if not np.any(ok):
            continue

        score = np.full_like(dr2, np.inf, dtype=float)
        score[ok] = dr2[ok] / denom[ok]

        j = int(np.argmin(score))

        min_score = score[j]
        ties = np.where(score == min_score)[0]
        if ties.size > 1:
            dr2_t = dr2[ties]
            j = int(ties[np.lexsort((ties, dr2_t))[0]])

        assign[i] = j

    jets_pt, jets_eta, jets_phi, jets_mass = build_clusters(pt, eta, phi, assign)
    jets = (jets_pt, jets_eta, jets_phi, jets_mass)
    return jets, assign, seed_mask


ALGO_REGISTRY = {
    # reference
    "antikt": antikt,

    # SC family
    "seeded_cone_greedy": seeded_cone_greedy,
    "seeded_cone_nms": seeded_cone_nms,
    "seeded_cone_nms_weighted": seeded_cone_nms_weighted,

    # simplified CLUE-like family
    "linktree": linktree,
    "linkseed_cone": linkseed_cone,
    "linkbary_cone": linkbary_cone,
    "link_centroid_nearest": link_centroid_nearest,
    "link_centroid_nearest_weighted": link_centroid_nearest_weighted,
    "linkbary_nearest_weighted": linkbary_nearest_weighted,
}
