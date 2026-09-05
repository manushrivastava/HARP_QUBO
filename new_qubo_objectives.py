"""New QUBO objectives for beating greedy sweep in Solomon VRPTW assignment.

This file provides:
- the original distance-all-pairs slack-bit QUBO builder
- a new sweep-angle compactness QUBO builder
- a new nearest-neighbor continuity QUBO builder
- a sampler that selects the best feasible solution from all reads
- the existing greedy sweep baseline and evaluation helpers

Usage:
    python new_qubo_objectives.py --instance r101 --keep 30 --method sweep_angle
"""

import math
import os
from pathlib import Path

import numpy as np
from neal import SimulatedAnnealingSampler

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# -----------------------------------------------------------------------------
# Solomon instance helpers
# -----------------------------------------------------------------------------

def load_solomon_txt(path: str):
    """Load Solomon VRPTW instance from lowercase or standard .txt format."""
    with open(path, "r") as f:
        raw_lines = f.readlines()

    lines = [ln.strip() for ln in raw_lines if ln.strip()]
    table_start = None
    for i, ln in enumerate(lines):
        up = ln.upper()
        if ("CUST" in up and "NO" in up) or ("CUSTOMER" in up and "NUMBER" in up):
            table_start = i + 1
            break

    first_row_idx = None
    for i, ln in enumerate(lines):
        parts = ln.split()
        if len(parts) == 7 and parts[0].lstrip("-").isdigit():
            first_row_idx = i
            break

    if table_start is None:
        table_start = first_row_idx
    if table_start is None:
        raise ValueError(f"Could not find customer table in: {path}")

    cap = None
    max_vehicles = None
    search_upto = table_start
    for i in range(search_upto):
        parts = lines[i].split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            max_vehicles = int(parts[0])
            cap = float(parts[1])
            break

    if cap is None:
        for i in range(search_upto):
            if "CAPACITY" in lines[i].upper():
                for j in range(i + 1, min(i + 10, search_upto)):
                    parts = lines[j].split()
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        max_vehicles = int(parts[0])
                        cap = float(parts[1])
                        break
                if cap is not None:
                    break

    if cap is None:
        raise ValueError(f"Capacity/max_vehicles not found in header area of: {path}")

    rows = []
    for ln in lines[table_start:]:
        parts = ln.split()
        if len(parts) == 7 and parts[0].lstrip("-").isdigit():
            rows.append([float(x) for x in parts])

    if not rows:
        for ln in lines:
            parts = ln.split()
            if len(parts) == 7 and parts[0].lstrip("-").isdigit():
                rows.append([float(x) for x in parts])

    if not rows:
        raise ValueError(f"Could not parse 7-column customer rows from: {path}")

    customers = {}
    for r in rows:
        cid = int(r[0])
        customers[cid] = {
            "id": cid,
            "x": float(r[1]),
            "y": float(r[2]),
            "demand": float(r[3]),
            "ready": float(r[4]),
            "due": float(r[5]),
            "service": float(r[6]),
        }

    first_cid = int(rows[0][0])
    if first_cid != 0:
        customers[0] = customers.pop(first_cid)
        customers[0]["id"] = 0

    return {
        "path": path,
        "capacity": cap,
        "max_vehicles": max_vehicles,
        "customers": customers,
    }


def reduce_instance(sol, keep: int, slack_vehicles: int = 0, customer_ids=None):
    customers = sol["customers"]
    if customer_ids is None:
        kept_ids = list(range(1, keep + 1))
    else:
        kept_ids = list(customer_ids)[:keep]

    cap = float(sol["capacity"])
    total_demand = sum(customers[i]["demand"] for i in kept_ids)
    K = int(math.ceil(total_demand / cap)) + int(slack_vehicles)

    cust_red = {cid: customers[cid] for cid in [0] + kept_ids}
    coords = {cid: (cust_red[cid]["x"], cust_red[cid]["y"]) for cid in [0] + kept_ids}

    return {
        "capacity": cap,
        "n_vehicles": K,
        "kept_ids": kept_ids,
        "customers": cust_red,
        "coords": coords,
        "total_demand_kept": float(total_demand),
    }


def euclid(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def pairwise_dist(inst):
    cached = inst.get("_pairwise_dist")
    if cached is not None:
        return cached

    ids = [0] + inst["kept_ids"]
    D = {}
    for a, i in enumerate(ids):
        for j in ids[a:]:
            d = 0.0 if i == j else euclid(inst["coords"][i], inst["coords"][j])
            D[(i, j)] = d
            D[(j, i)] = d
    inst["_pairwise_dist"] = D
    return D


def tsp_nn_route_length_from_D(D, cluster):
    cluster = list(cluster)
    if not cluster:
        return 0.0
    unvisited = set(cluster)
    cur = 0
    dist = 0.0
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[(cur, j)])
        dist += D[(cur, nxt)]
        cur = nxt
        unvisited.remove(nxt)
    dist += D[(cur, 0)]
    return float(dist)


def total_distance_nn(inst, sets):
    D = pairwise_dist(inst)
    return sum(tsp_nn_route_length_from_D(D, s) for s in sets)


def total_distance_exact_or_nn(inst, sets, use_exact=False, exact_max_cluster_size=15):
    if not use_exact:
        return total_distance_nn(inst, sets)
    if "tsp_optimal_route_length" not in globals():
        return total_distance_nn(inst, sets)
    total = 0.0
    for s in sets:
        if len(s) <= exact_max_cluster_size:
            total += float(tsp_optimal_route_length(inst, s))
        else:
            D = pairwise_dist(inst)
            total += tsp_nn_route_length_from_D(D, s)
    return float(total)


def greedy_assignment_sweep(inst):
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    cap = inst["capacity"]
    cust = inst["customers"]
    dx, dy = inst["coords"][0]

    def angle(cid):
        x, y = inst["coords"][cid]
        return math.atan2(y - dy, x - dx)

    order = sorted(kept, key=angle)
    sets = [[] for _ in range(K)]
    loads = [0.0] * K
    k = 0
    feasible = True

    for cid in order:
        d = cust[cid]["demand"]
        placed = False
        for kk in range(k, K):
            if loads[kk] + d <= cap + 1e-9:
                sets[kk].append(cid)
                loads[kk] += d
                k = kk
                placed = True
                break
        if not placed:
            feasible = False
            sets[K - 1].append(cid)
            loads[K - 1] += d

    return sets, loads, order, feasible


def greedy_initial_state(inst, varmap, zmap=None):
    sets, _, _, _ = greedy_assignment_sweep(inst)
    state = {}
    for k, cluster in enumerate(sets):
        for cid in cluster:
            state[varmap[(cid, k)]] = 1
    # ensure all other x variables are explicitly zero
    for (cid, k), name in varmap.items():
        state.setdefault(name, 0)
    if zmap is not None:
        for key, name in zmap.items():
            state.setdefault(name, 0)
    return state


def angle_order_weights(inst):
    depot = inst["coords"][0]
    angles = {}
    for cid in inst["kept_ids"]:
        x, y = inst["coords"][cid]
        theta = math.atan2(y - depot[1], x - depot[0])
        if theta < 0:
            theta += 2.0 * math.pi
        angles[cid] = theta
    return angles


def nearest_neighbor_pairs(inst, neighbors=4):
    D = pairwise_dist(inst)
    ids = inst["kept_ids"]
    pairs = set()
    for i in ids:
        neigh = sorted((D[(i, j)], j) for j in ids if j != i)[:neighbors]
        for _, j in neigh:
            a, b = (i, j) if i < j else (j, i)
            pairs.add((a, b))
    return sorted(pairs)


def build_qubo_distance_allpairs_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
):
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]
    if slack_bits is None:
        L = max(1, int(math.ceil(math.log2(int(cap) + 1))))
    else:
        L = int(slack_bits)
        if L < 1:
            raise ValueError("slack_bits must be >= 1")

    var = {(cid, k): f"x_{cid}_{k}" for cid in kept for k in range(K)}
    z = {(k, b): f"z_{k}_{b}" for k in range(K) for b in range(L)}
    Q = {}

    def add(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(val)

    for cid in kept:
        for k in range(K):
            add(var[(cid, k)], var[(cid, k)], -A)
        for k in range(K):
            for l in range(k + 1, K):
                add(var[(cid, k)], var[(cid, l)], 2.0 * A)

    cap_norm = 1.0 if normalize_capacity else cap
    for k in range(K):
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            add(xi, xi, B * (d * d - 2.0 * cap_norm * d))
        for a in range(len(kept)):
            ci = kept[a]
            di = float(cust[ci]["demand"])
            if normalize_capacity:
                di /= cap
            for b in range(a + 1, len(kept)):
                cj = kept[b]
                dj = float(cust[cj]["demand"])
                if normalize_capacity:
                    dj /= cap
                add(var[(ci, k)], var[(cj, k)], B * (2.0 * di * dj))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            zb = z[(k, b)]
            add(zb, zb, B * (wb * wb - 2.0 * cap_norm * wb))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            for c in range(b + 1, L):
                wc = float(1 << c)
                if normalize_capacity:
                    wc /= cap
                add(z[(k, b)], z[(k, c)], B * (2.0 * wb * wc))
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            for b in range(L):
                wb = float(1 << b)
                if normalize_capacity:
                    wb /= cap
                add(xi, z[(k, b)], B * (2.0 * d * wb))

    for a in range(len(kept)):
        i = kept[a]
        xi, yi = inst["coords"][i]
        for b in range(a + 1, len(kept)):
            j = kept[b]
            xj, yj = inst["coords"][j]
            w_ij = math.hypot(xj - xi, yj - yi)
            for k in range(K):
                add(var[(i, k)], var[(j, k)], gamma * w_ij)

    return Q, var, z, order


def build_qubo_sweep_angle_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
):
    kept = inst["kept_ids"]
    angles = angle_order_weights(inst)
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]
    if slack_bits is None:
        L = max(1, int(math.ceil(math.log2(int(cap) + 1))))
    else:
        L = int(slack_bits)
        if L < 1:
            raise ValueError("slack_bits must be >= 1")

    var = {(cid, k): f"x_{cid}_{k}" for cid in kept for k in range(K)}
    z = {(k, b): f"z_{k}_{b}" for k in range(K) for b in range(L)}
    Q = {}

    def add(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(val)

    for cid in kept:
        for k in range(K):
            add(var[(cid, k)], var[(cid, k)], -A)
        for k in range(K):
            for l in range(k + 1, K):
                add(var[(cid, k)], var[(cid, l)], 2.0 * A)

    cap_norm = 1.0 if normalize_capacity else cap
    for k in range(K):
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            add(xi, xi, B * (d * d - 2.0 * cap_norm * d))
        for a in range(len(kept)):
            ci = kept[a]
            di = float(cust[ci]["demand"])
            if normalize_capacity:
                di /= cap
            for b in range(a + 1, len(kept)):
                cj = kept[b]
                dj = float(cust[cj]["demand"])
                if normalize_capacity:
                    dj /= cap
                add(var[(ci, k)], var[(cj, k)], B * (2.0 * di * dj))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            zb = z[(k, b)]
            add(zb, zb, B * (wb * wb - 2.0 * cap_norm * wb))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            for c in range(b + 1, L):
                wc = float(1 << c)
                if normalize_capacity:
                    wc /= cap
                add(z[(k, b)], z[(k, c)], B * (2.0 * wb * wc))
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            for b in range(L):
                wb = float(1 << b)
                if normalize_capacity:
                    wb /= cap
                add(xi, z[(k, b)], B * (2.0 * d * wb))

    for a in range(len(kept)):
        i = kept[a]
        for b in range(a + 1, len(kept)):
            j = kept[b]
            angle_i = angles[i]
            angle_j = angles[j]
            diff = abs(angle_i - angle_j)
            diff = min(diff, 2.0 * math.pi - diff)
            weight = diff / math.pi
            for k in range(K):
                add(var[(i, k)], var[(j, k)], gamma * weight)

    return Q, var, z, order


def build_qubo_nearest_neighbor_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
    neighbors=4,
):
    kept = inst["kept_ids"]
    nn_pairs = nearest_neighbor_pairs(inst, neighbors=neighbors)
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]
    if slack_bits is None:
        L = max(1, int(math.ceil(math.log2(int(cap) + 1))))
    else:
        L = int(slack_bits)
        if L < 1:
            raise ValueError("slack_bits must be >= 1")

    var = {(cid, k): f"x_{cid}_{k}" for cid in kept for k in range(K)}
    z = {(k, b): f"z_{k}_{b}" for k in range(K) for b in range(L)}
    Q = {}

    def add(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(val)

    for cid in kept:
        for k in range(K):
            add(var[(cid, k)], var[(cid, k)], -A)
        for k in range(K):
            for l in range(k + 1, K):
                add(var[(cid, k)], var[(cid, l)], 2.0 * A)

    cap_norm = 1.0 if normalize_capacity else cap
    for k in range(K):
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            add(xi, xi, B * (d * d - 2.0 * cap_norm * d))
        for a in range(len(kept)):
            ci = kept[a]
            di = float(cust[ci]["demand"])
            if normalize_capacity:
                di /= cap
            for b in range(a + 1, len(kept)):
                cj = kept[b]
                dj = float(cust[cj]["demand"])
                if normalize_capacity:
                    dj /= cap
                add(var[(ci, k)], var[(cj, k)], B * (2.0 * di * dj))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            zb = z[(k, b)]
            add(zb, zb, B * (wb * wb - 2.0 * cap_norm * wb))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            for c in range(b + 1, L):
                wc = float(1 << c)
                if normalize_capacity:
                    wc /= cap
                add(z[(k, b)], z[(k, c)], B * (2.0 * wb * wc))
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            for b in range(L):
                wb = float(1 << b)
                if normalize_capacity:
                    wb /= cap
                add(xi, z[(k, b)], B * (2.0 * d * wb))

    D = pairwise_dist(inst)
    for (i, j) in nn_pairs:
        w_ij = D[(i, j)]
        for k in range(K):
            add(var[(i, k)], var[(j, k)], gamma * w_ij)

    return Q, var, z, order


def build_qubo_nn_angle_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
    neighbors=4,
):
    kept = inst["kept_ids"]
    nn_pairs = nearest_neighbor_pairs(inst, neighbors=neighbors)
    angles = angle_order_weights(inst)
    sorted_by_angle = sorted(kept, key=lambda cid: angles[cid])
    max_dist = max(pairwise_dist(inst).values()) if kept else 1.0
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]
    if slack_bits is None:
        L = max(1, int(math.ceil(math.log2(int(cap) + 1))))
    else:
        L = int(slack_bits)
        if L < 1:
            raise ValueError("slack_bits must be >= 1")

    var = {(cid, k): f"x_{cid}_{k}" for cid in kept for k in range(K)}
    z = {(k, b): f"z_{k}_{b}" for k in range(K) for b in range(L)}
    Q = {}

    def add(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(val)

    for cid in kept:
        for k in range(K):
            add(var[(cid, k)], var[(cid, k)], -A)
        for k in range(K):
            for l in range(k + 1, K):
                add(var[(cid, k)], var[(cid, l)], 2.0 * A)

    cap_norm = 1.0 if normalize_capacity else cap
    for k in range(K):
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            add(xi, xi, B * (d * d - 2.0 * cap_norm * d))
        for a in range(len(kept)):
            ci = kept[a]
            di = float(cust[ci]["demand"])
            if normalize_capacity:
                di /= cap
            for b in range(a + 1, len(kept)):
                cj = kept[b]
                dj = float(cust[cj]["demand"])
                if normalize_capacity:
                    dj /= cap
                add(var[(ci, k)], var[(cj, k)], B * (2.0 * di * dj))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            zb = z[(k, b)]
            add(zb, zb, B * (wb * wb - 2.0 * cap_norm * wb))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            for c in range(b + 1, L):
                wc = float(1 << c)
                if normalize_capacity:
                    wc /= cap
                add(z[(k, b)], z[(k, c)], B * (2.0 * wb * wc))
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            for b in range(L):
                wb = float(1 << b)
                if normalize_capacity:
                    wb /= cap
                add(xi, z[(k, b)], B * (2.0 * d * wb))

    D = pairwise_dist(inst)
    for (i, j) in nn_pairs:
        w_ij = D[(i, j)] / max_dist
        for k in range(K):
            add(var[(i, k)], var[(j, k)], gamma * w_ij)

    for a in range(len(sorted_by_angle) - 1):
        i = sorted_by_angle[a]
        j = sorted_by_angle[a + 1]
        diff = abs(angles[i] - angles[j])
        diff = min(diff, 2.0 * math.pi - diff)
        angle_weight = diff / math.pi
        for k in range(K):
            add(var[(i, k)], var[(j, k)], gamma * angle_weight)

    return Q, var, z, order


def build_qubo_depot_proxy_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    beta=0.05,
    slack_bits=None,
    normalize_capacity=True,
    neighbors=4,
):
    """Combine nearest-neighbor compactness with a depot-distance linear proxy.

    The `beta` parameter adds a linear penalty proportional to the distance
    from the depot to each customer when assigned to any vehicle. This
    approximates depot entry/exit costs and encourages vehicles to pick
    customers closer to the depot.
    """
    # build base nearest-neighbor Q
    Q, var, z, order = build_qubo_nearest_neighbor_slack(
        inst,
        A=A,
        B=B,
        gamma=gamma,
        slack_bits=slack_bits,
        normalize_capacity=normalize_capacity,
        neighbors=neighbors,
    )

    # add linear depot-distance proxy terms to diagonals
    D = pairwise_dist(inst)
    for cid in inst["kept_ids"]:
        d0 = D[(0, cid)]
        for k in range(inst["n_vehicles"]):
            xi = var[(cid, k)]
            add_key = (xi, xi) if xi <= xi else (xi, xi)
            # accumulate onto Q; follow same internal structure as other builders
            Q[(xi, xi)] = Q.get((xi, xi), 0.0) + float(beta * d0)

    return Q, var, z, order


def build_qubo_firstlast_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
    neighbors=4,
    alpha_end=0.5,
    P_end=200.0,
):
    """Nearest-neighbor base plus auxiliary endpoint variables per vehicle.

    For each vehicle k we create endpoint binary variables `e_{cid,k}` and
    enforce (1) e_{cid,k} => x_{cid,k} and (2) sum_c e_{cid,k} == 2 via
    quadratic penalties. The objective adds `alpha_end * distance(depot,cid)`
    for endpoint variables to approximate first/last depot entry/exit cost.
    """
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]

    # start with nearest-neighbor base
    Q, var, z, order = build_qubo_nearest_neighbor_slack(
        inst,
        A=A,
        B=B,
        gamma=gamma,
        slack_bits=slack_bits,
        normalize_capacity=normalize_capacity,
        neighbors=neighbors,
    )

    # add endpoint variables
    e = {(cid, k): f"e_{cid}_{k}" for cid in kept for k in range(K)}

    def add_to_Q(u, v, val):
        if abs(val) < 1e-12:
            return
        if u > v:
            u, v = v, u
        Q[(u, v)] = Q.get((u, v), 0.0) + float(val)

    D = pairwise_dist(inst)

    # For each endpoint variable: penalize e*(1 - x) via term P_end*e - P_end*e*x
    for cid in kept:
        d0 = D[(0, cid)]
        for k in range(K):
            ei = e[(cid, k)]
            xi = var[(cid, k)]
            # linear term P_end * e
            add_to_Q(ei, ei, P_end * 1.0)
            # coupling -P_end * e * x  (discourages e=1 when x=0)
            add_to_Q(ei, xi, -P_end * 1.0)
            # objective: endpoint cost proportional to depot distance
            add_to_Q(ei, ei, alpha_end * d0)

    # For each vehicle enforce sum_e == 2 (first and last) with quadratic penalty
    for k in range(K):
        ids = list(kept)
        # expand (sum_e_k - 2)^2 = sum_i sum_j e_i e_j - 4 sum_i e_i + 4
        for i in range(len(ids)):
            ei = e[(ids[i], k)]
            # diagonal contribution from -4*e_i
            add_to_Q(ei, ei, -4.0 * P_end)
            for j in range(i, len(ids)):
                ej = e[(ids[j], k)]
                add_to_Q(ei, ej, P_end * 1.0)
        # constant term 4*P_end can be ignored for optimization

    # Ensure Q includes endpoint variable names in var/z lists for Qmat building order
    # Merge e map into varmap-like mapping by returning var extended
    # We'll append e names into z map when building Qmat in run_qubo
    # To support run_qubo expectations, return var and z; include endpoints in z
    # by merging them into zmap (they are auxiliary)
    # convert dictionaries to include e variables into zmap to be returned
    # existing callers expect zmap to be mapping of slack bits; we can return e's there
    z_ext = dict(z)
    for (cid, k), name in e.items():
        z_ext[(cid, k)] = name

    return Q, var, z_ext, order


def build_qubo_sweep_contiguous_slack(
    inst,
    A=20.0,
    B=30.0,
    gamma=0.01,
    slack_bits=None,
    normalize_capacity=True,
):
    """Sweep-based QUBO that encourages contiguous assignments along sweep order.

    Adds a negative adjacency coupling between consecutive customers in
    polar-angle order to encourage them to be assigned to the same vehicle.
    """
    kept = inst["kept_ids"]
    angles = angle_order_weights(inst)
    sorted_by_angle = sorted(kept, key=lambda cid: angles[cid])
    K = inst["n_vehicles"]
    cap = float(inst["capacity"])
    cust = inst["customers"]
    order = kept[:]
    if slack_bits is None:
        L = max(1, int(math.ceil(math.log2(int(cap) + 1))))
    else:
        L = int(slack_bits)
        if L < 1:
            raise ValueError("slack_bits must be >= 1")

    var = {(cid, k): f"x_{cid}_{k}" for cid in kept for k in range(K)}
    z = {(k, b): f"z_{k}_{b}" for k in range(K) for b in range(L)}
    Q = {}

    def add(i, j, val):
        if abs(val) < 1e-12:
            return
        if i > j:
            i, j = j, i
        Q[(i, j)] = Q.get((i, j), 0.0) + float(val)

    for cid in kept:
        for k in range(K):
            add(var[(cid, k)], var[(cid, k)], -A)
        for k in range(K):
            for l in range(k + 1, K):
                add(var[(cid, k)], var[(cid, l)], 2.0 * A)

    cap_norm = 1.0 if normalize_capacity else cap
    for k in range(K):
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            add(xi, xi, B * (d * d - 2.0 * cap_norm * d))
        for a in range(len(kept)):
            ci = kept[a]
            di = float(cust[ci]["demand"])
            if normalize_capacity:
                di /= cap
            for b in range(a + 1, len(kept)):
                cj = kept[b]
                dj = float(cust[cj]["demand"])
                if normalize_capacity:
                    dj /= cap
                add(var[(ci, k)], var[(cj, k)], B * (2.0 * di * dj))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            zb = z[(k, b)]
            add(zb, zb, B * (wb * wb - 2.0 * cap_norm * wb))
        for b in range(L):
            wb = float(1 << b)
            if normalize_capacity:
                wb /= cap
            for c in range(b + 1, L):
                wc = float(1 << c)
                if normalize_capacity:
                    wc /= cap
                add(z[(k, b)], z[(k, c)], B * (2.0 * wb * wc))
        for cid in kept:
            d = float(cust[cid]["demand"])
            if normalize_capacity:
                d /= cap
            xi = var[(cid, k)]
            for b in range(L):
                wb = float(1 << b)
                if normalize_capacity:
                    wb /= cap
                add(xi, z[(k, b)], B * (2.0 * d * wb))

    for a in range(len(kept)):
        i = kept[a]
        for b in range(a + 1, len(kept)):
            j = kept[b]
            angle_i = angles[i]
            angle_j = angles[j]
            diff = abs(angle_i - angle_j)
            diff = min(diff, 2.0 * math.pi - diff)
            weight = diff / math.pi
            for k in range(K):
                add(var[(i, k)], var[(j, k)], gamma * weight)

    # encourage contiguous sweep neighbors to be assigned together
    # add a negative coupling for consecutive pairs in angle order
    for idx in range(len(sorted_by_angle) - 1):
        i = sorted_by_angle[idx]
        j = sorted_by_angle[idx + 1]
        for k in range(K):
            add(var[(i, k)], var[(j, k)], -2.0 * gamma)

    return Q, var, z, order  
    


def decode_assignment_argmax(sample, inst, varmap):
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    sets = [[] for _ in range(K)]
    for cid in kept:
        vals = [int(sample.get(varmap[(cid, k)], 0)) for k in range(K)]
        kbest = int(np.argmax(vals))
        sets[kbest].append(cid)
    loads = [sum(inst["customers"][cid]["demand"] for cid in sets[k]) for k in range(K)]
    return sets, loads


def onehot_ok(sample, inst, varmap):
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    for cid in kept:
        if sum(int(sample.get(varmap[(cid, k)], 0)) for k in range(K)) != 1:
            return False
    return True


def loads_from_sample(sample, inst, varmap):
    kept = inst["kept_ids"]
    K = inst["n_vehicles"]
    cust = inst["customers"]
    loads = [0.0] * K
    for cid in kept:
        d = float(cust[cid]["demand"])
        for k in range(K):
            if int(sample.get(varmap[(cid, k)], 0)) == 1:
                loads[k] += d
    return loads


def capacity_ok_from_sample(sample, inst, varmap):
    if not onehot_ok(sample, inst, varmap):
        return False
    loads = loads_from_sample(sample, inst, varmap)
    return all(ld <= inst["capacity"] + 1e-6 for ld in loads)


def best_feasible_sample(sampleset, inst, varmap, zmap, use_exact=False):
    best = None
    best_dist = float("inf")
    for sample_data in sampleset.data(["sample", "energy"]):
        sample = dict(sample_data.sample)
        if not onehot_ok(sample, inst, varmap):
            continue
        if not capacity_ok_from_sample(sample, inst, varmap):
            continue
        sets, loads = decode_assignment_argmax(sample, inst, varmap)
        dist = total_distance_exact_or_nn(inst, sets, use_exact=use_exact)
        if dist < best_dist:
            best_dist = dist
            best = {
                "sample": sample,
                "energy": float(sample_data.energy),
                "sets": sets,
                "loads": loads,
                "distance": dist,
            }
    return best


def run_qubo(
    inst,
    build_qubo_fn,
    seed=0,
    A=20.0,
    B=30.0,
    gamma=0.01,
    num_reads=300,
    num_sweeps=3000,
    slack_bits=None,
    normalize_capacity=True,
    use_exact=False,
    use_greedy_seed=False,
    build_qmat=True,
):
    Q, varmap, zmap, order = build_qubo_fn(
        inst,
        A=A,
        B=B,
        gamma=gamma,
        slack_bits=slack_bits,
        normalize_capacity=normalize_capacity,
    )

    sampler = SimulatedAnnealingSampler()
    if use_greedy_seed:
        initial_state = greedy_initial_state(inst, varmap, zmap=zmap)
        sampleset = sampler.sample_qubo(
            Q,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed,
            initial_states=[initial_state],
            initial_states_generator='tile',
            randomize_order=False,
        )
    else:
        sampleset = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    best_entry = best_feasible_sample(sampleset, inst, varmap, zmap, use_exact=use_exact)
    if best_entry is None:
        best_entry = {
            "sample": dict(sampleset.first.sample),
            "energy": float(sampleset.first.energy),
            "sets": decode_assignment_argmax(sampleset.first.sample, inst, varmap)[0],
            "loads": loads_from_sample(sampleset.first.sample, inst, varmap),
            "distance": total_distance_exact_or_nn(inst, decode_assignment_argmax(sampleset.first.sample, inst, varmap)[0], use_exact=use_exact),
        }

    info = {
        "energy": best_entry["energy"],
        "assign_once_ok": onehot_ok(best_entry["sample"], inst, varmap),
        "assign_cap_ok": capacity_ok_from_sample(best_entry["sample"], inst, varmap),
        "loads_from_bits": best_entry["loads"],
        "sum_loads_bits": float(sum(best_entry["loads"])),
        "total_demand_kept": float(inst["total_demand_kept"]),
        "order": order,
    }

    Qmat, n_vars = None, None
    if build_qmat:
        all_names = list(varmap.values()) + list(zmap.values())
        name_to_idx = {name: i for i, name in enumerate(all_names)}
        n_vars = len(all_names)
        Qmat = np.zeros((n_vars, n_vars), dtype=float)
        for (u, v), val in Q.items():
            i = name_to_idx[u]
            j = name_to_idx[v]
            Qmat[i, j] += val
            if i != j:
                Qmat[j, i] += val

    return best_entry["sets"], best_entry["loads"], info, Qmat, n_vars, best_entry["distance"], best_entry["sample"], varmap, zmap


def build_qmat(Q, varmap, zmap):
    all_names = list(varmap.values()) + list(zmap.values())
    name_to_idx = {name: i for i, name in enumerate(all_names)}
    n_vars = len(all_names)
    Qmat = np.zeros((n_vars, n_vars), dtype=float)
    for (u, v), val in Q.items():
        i = name_to_idx[u]
        j = name_to_idx[v]
        Qmat[i, j] += val
        if i != j:
            Qmat[j, i] += val
    return Qmat, n_vars


def print_summary(inst, sets, loads, info, dist):
    print("--- Summary ---")
    print("Vehicles:", inst['n_vehicles'])
    print("Kept customers:", len(inst['kept_ids']))
    print("Total demand kept:", inst['total_demand_kept'])
    print("Distance:", dist)
    print("Feasible one-hot:", info['assign_once_ok'])
    print("Feasible capacity:", info['assign_cap_ok'])
    print("Loads:", loads)


def parse_seeds(seed_arg):
    if isinstance(seed_arg, int):
        return [seed_arg]
    s = str(seed_arg).strip()
    if not s:
        return []
    seeds = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            start, end = token.split('-', 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(token))
    return seeds


def run_qubo_over_seeds(
    inst,
    build_qubo_fn,
    seeds,
    A=20.0,
    B=30.0,
    gamma=0.01,
    num_reads=300,
    num_sweeps=3000,
    slack_bits=None,
    normalize_capacity=True,
    use_exact=False,
    use_greedy_seed=False,
    build_qmat=False,
):
    best_result = None
    for seed in seeds:
        sets, loads, info, Qmat, n_vars, dist, sample, varmap, zmap = run_qubo(
            inst,
            build_qubo_fn=build_qubo_fn,
            seed=seed,
            A=A,
            B=B,
            gamma=gamma,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            slack_bits=slack_bits,
            normalize_capacity=normalize_capacity,
            use_exact=use_exact,
            use_greedy_seed=use_greedy_seed,
            build_qmat=build_qmat,
        )
        feasible = info['assign_once_ok'] and info['assign_cap_ok']
        if not feasible:
            continue
        if best_result is None or dist < best_result['dist']:
            best_result = {
                'seed': seed,
                'sets': sets,
                'loads': loads,
                'info': info,
                'dist': dist,
                'sample': sample,
                'varmap': varmap,
                'zmap': zmap,
                'Qmat': Qmat,
                'n_vars': n_vars,
            }
    return best_result


def tsp_nn_route(inst, cluster):
    cluster = list(cluster)
    if not cluster:
        return [0]
    D = pairwise_dist(inst)
    unvisited = set(cluster)
    cur = 0
    route = [0]
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[(cur, j)])
        route.append(nxt)
        cur = nxt
        unvisited.remove(nxt)
    route.append(0)
    return route


def annotate_points(ax, coords, cluster):
    for cid in cluster:
        x, y = coords[cid]
        ax.text(x + 0.3, y + 0.3, str(cid), fontsize=7, color='black')


def plot_assignment(inst, sets, title="Assignment", save_path=None):
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for plotting. Install it with pip install matplotlib")

    colors = plt.colormaps['tab20'] if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab20', max(1, inst['n_vehicles']))
    coords = inst['coords']
    plt.figure(figsize=(10, 10))
    for k, cluster in enumerate(sets):
        xs = [coords[cid][0] for cid in cluster]
        ys = [coords[cid][1] for cid in cluster]
        plt.scatter(xs, ys, label=f'V{k+1}', color=colors(k), s=60, edgecolor='k', alpha=0.8)
        route = tsp_nn_route(inst, cluster)
        rx = [coords[cid][0] for cid in route]
        ry = [coords[cid][1] for cid in route]
        plt.plot(rx, ry, color=colors(k), linestyle='--', linewidth=1, alpha=0.6)
        annotate_points(plt.gca(), coords, cluster)
    depot_x, depot_y = coords[0]
    plt.scatter([depot_x], [depot_y], marker='*', color='black', s=220, label='Depot')
    plt.text(depot_x + 0.3, depot_y + 0.3, '0', fontsize=8, color='black')
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.5)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_assignments_side_by_side(inst, sets_a, sets_b, title_a='A', title_b='B', save_path=None):
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError("matplotlib is required for plotting. Install it with pip install matplotlib")

    colors = plt.colormaps['tab20'] if hasattr(plt, 'colormaps') else plt.cm.get_cmap('tab20', max(1, inst['n_vehicles']))
    coords = inst['coords']

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), sharex=False, sharey=False)
    for ax, sets, title in zip(axes, [sets_a, sets_b], [title_a, title_b]):
        for k, cluster in enumerate(sets):
            xs = [coords[cid][0] for cid in cluster]
            ys = [coords[cid][1] for cid in cluster]
            ax.scatter(xs, ys, label=f'V{k+1}', color=colors(k), s=60, edgecolor='k', alpha=0.8)
            route = tsp_nn_route(inst, cluster)
            rx = [coords[cid][0] for cid in route]
            ry = [coords[cid][1] for cid in route]
            ax.plot(rx, ry, color=colors(k), linestyle='--', linewidth=1, alpha=0.6)
            annotate_points(ax, coords, cluster)
        depot_x, depot_y = coords[0]
        ax.scatter([depot_x], [depot_y], marker='*', color='black', s=220, label='Depot')
        ax.text(depot_x + 0.3, depot_y + 0.3, '0', fontsize=8, color='black')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle=':', alpha=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=min(6, len(labels)), fontsize='small')
    fig.suptitle(f'Assignment comparison: {title_a} vs {title_b}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run new QUBO objectives for Solomon assignment')
    parser.add_argument('--instance', type=str, default='r101')
    parser.add_argument('--keep', type=int, default=30)
    parser.add_argument('--slack-vehicles', type=int, default=0)
    parser.add_argument('--method', type=str, choices=['allpairs', 'sweep_angle', 'nearest_neighbor', 'nn_angle', 'sweep_contiguous', 'depot_proxy', 'firstlast'], default='sweep_angle')
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--A', type=float, default=200.0)
    parser.add_argument('--B', type=float, default=30.0)
    parser.add_argument('--beta', type=float, default=0.05, help='Depot-distance linear penalty weight')
    parser.add_argument('--alpha-end', type=float, default=0.5, help='Endpoint depot-distance weight')
    parser.add_argument('--P-end', type=float, default=200.0, help='Penalty weight for endpoint constraints')
    parser.add_argument('--seed-greedy', action='store_true', help='Use the greedy sweep assignment as the initial QUBO state')
    parser.add_argument('--num-reads', type=int, default=300)
    parser.add_argument('--num-sweeps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--slack-bits', type=int, default=None)
    parser.add_argument('--use-exact', action='store_true')
    parser.add_argument('--sol-dir', type=str, default='data/solomon-100')
    parser.add_argument('--seeds', type=str, default='100', help='Comma-separated seeds or inclusive ranges like 100-104')
    parser.add_argument('--plot', action='store_true', help='Show a plot of the QUBO assignment')
    parser.add_argument('--plot-greedy', action='store_true', help='Show a comparison plot against greedy sweep')
    parser.add_argument('--plot-file', type=str, default=None, help='Save plot to this file path instead of showing it')
    args = parser.parse_args()

    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{args.instance}.txt"))
    inst = reduce_instance(sol, keep=args.keep, slack_vehicles=args.slack_vehicles)

    builder = {
        'allpairs': build_qubo_distance_allpairs_slack,
        'sweep_angle': build_qubo_sweep_angle_slack,
        'sweep_contiguous': build_qubo_sweep_contiguous_slack,
        'nearest_neighbor': build_qubo_nearest_neighbor_slack,
        'nn_angle': build_qubo_nn_angle_slack,
        'depot_proxy': build_qubo_depot_proxy_slack,
        'firstlast': build_qubo_firstlast_slack,
    }[args.method]

    seed_list = parse_seeds(args.seeds)
    if not seed_list:
        raise ValueError('At least one seed must be provided via --seeds')

    if len(seed_list) == 1:
        sets, loads, info, Qmat, n_vars, dist, sample, varmap, zmap = run_qubo(
            inst,
            build_qubo_fn=builder,
            seed=seed_list[0],
            A=args.A,
            B=args.B,
            gamma=args.gamma,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            slack_bits=args.slack_bits,
            normalize_capacity=True,
            use_exact=args.use_exact,
            use_greedy_seed=args.seed_greedy,
            build_qmat=False,
        )
        best_seed = seed_list[0]
    else:
        best = run_qubo_over_seeds(
            inst,
            build_qubo_fn=builder,
            seeds=seed_list,
            A=args.A,
            B=args.B,
            gamma=args.gamma,
            num_reads=args.num_reads,
            num_sweeps=args.num_sweeps,
            slack_bits=args.slack_bits,
            normalize_capacity=True,
            use_exact=args.use_exact,
            use_greedy_seed=args.seed_greedy,
            build_qmat=False,
        )
        if best is None:
            raise RuntimeError('No feasible QUBO solution found across given seeds')
        sets = best['sets']
        loads = best['loads']
        info = best['info']
        Qmat = best['Qmat']
        n_vars = best['n_vars']
        dist = best['dist']
        sample = best['sample']
        varmap = best['varmap']
        zmap = best['zmap']
        best_seed = best['seed']

    print(f"Method: {args.method}")
    print(f"Instance: {args.instance}, keep={args.keep}, gamma={args.gamma}")
    print(f"Seed(s): {args.seeds} -> best seed: {best_seed}")
    print_summary(inst, sets, loads, info, dist)

    if args.plot_greedy:
        greedy_sets, greedy_loads, _, greedy_feasible = greedy_assignment_sweep(inst)
        filename = args.plot_file or None
        plot_assignments_side_by_side(inst, greedy_sets, sets, title_a='Greedy Sweep', title_b=f'QUBO {args.method}', save_path=filename)
    elif args.plot:
        filename = args.plot_file or None
        plot_assignment(inst, sets, title=f'QUBO {args.method}', save_path=filename)
