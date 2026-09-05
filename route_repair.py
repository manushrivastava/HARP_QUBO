"""
route_repair.py
=====================================================================
Classical route-set repair/local-search utilities, extracted from the
original run_self_supervised_ml_qubo.py (which pulls in a much heavier
set of ML-experiment dependencies -- sklearn, feedback-tuning modules
-- that this "why quantum" storyline does not need).

Only the functions actually used by probe_strong_route_pool_repair.py
are kept here: intra-route local search, cross-route relocate/swap
local search, and the combined standalone_improve_routes repair loop
used on the top LNS route sets ("guarded repair").
=====================================================================
"""
from decomposed_vrptw_qubo import evaluate_routes, route_metrics


def route_loads(inst, routes):
    return [
        sum(float(inst["customers"][cid]["demand"]) for cid in route if cid != 0)
        for route in routes
    ]


def score_routes(inst, routes, tw_weight):
    metrics = evaluate_routes(inst, routes)
    return float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])


def single_route_score(inst, route, tw_weight):
    metrics = route_metrics(inst, route)
    return float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])


def compact_routes(routes):
    compacted = []
    for route in routes:
        customers = [cid for cid in route if cid != 0]
        if customers:
            compacted.append([0] + customers + [0])
    return compacted


def local_search_route_order(inst, route, tw_weight, max_passes=2):
    """Small deterministic intra-route search (2-opt + relocate) used by standalone repair."""
    if len(route) <= 4:
        return list(route)

    best = list(route)
    best_score = single_route_score(inst, best, tw_weight)
    for _ in range(max_passes):
        improved = False
        n = len(best)

        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                candidate = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                candidate_score = single_route_score(inst, candidate, tw_weight)
                if candidate_score + 1e-9 < best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True

        n = len(best)
        for i in range(1, n - 1):
            cid = best[i]
            remaining = best[:i] + best[i + 1 :]
            for j in range(1, len(remaining)):
                if j == i:
                    continue
                candidate = remaining[:j] + [cid] + remaining[j:]
                candidate_score = single_route_score(inst, candidate, tw_weight)
                if candidate_score + 1e-9 < best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True

        if not improved:
            break
    return best


def local_search_route_set(inst, routes, tw_weight, max_passes=2):
    return [local_search_route_order(inst, route, tw_weight, max_passes=max_passes) for route in routes]


def cross_route_relocate_search(inst, routes, tw_weight, max_passes=2):
    """Move customers across routes when capacity and score improve."""
    capacity = float(inst["capacity"])
    routes = compact_routes(routes)
    if len(routes) <= 1:
        return routes

    for _ in range(max_passes):
        loads = route_loads(inst, routes)
        route_scores = [single_route_score(inst, route, tw_weight) for route in routes]
        best_move = None
        best_gain = 0.0

        for src_idx, src in enumerate(routes):
            for src_pos in range(1, len(src) - 1):
                cid = src[src_pos]
                demand = float(inst["customers"][cid]["demand"])
                reduced_src = src[:src_pos] + src[src_pos + 1 :]
                reduced_src_score = single_route_score(inst, reduced_src, tw_weight) if len(reduced_src) > 2 else 0.0

                for dst_idx, dst in enumerate(routes):
                    if dst_idx == src_idx or loads[dst_idx] + demand > capacity + 1e-9:
                        continue
                    for dst_pos in range(1, len(dst)):
                        expanded_dst = dst[:dst_pos] + [cid] + dst[dst_pos:]
                        new_score = reduced_src_score + single_route_score(inst, expanded_dst, tw_weight)
                        old_score = route_scores[src_idx] + route_scores[dst_idx]
                        gain = old_score - new_score
                        if gain > best_gain + 1e-9:
                            best_gain = gain
                            best_move = (src_idx, src_pos, dst_idx, dst_pos)

        if best_move is None:
            break
        src_idx, src_pos, dst_idx, dst_pos = best_move
        cid = routes[src_idx][src_pos]
        routes[src_idx] = routes[src_idx][:src_pos] + routes[src_idx][src_pos + 1 :]
        insert_pos = dst_pos if dst_idx != src_idx else dst_pos
        routes[dst_idx] = routes[dst_idx][:insert_pos] + [cid] + routes[dst_idx][insert_pos:]
        routes = compact_routes(routes)

    return routes


def cross_route_swap_search(inst, routes, tw_weight, max_passes=2):
    """Swap customers between routes when both capacity and score improve."""
    capacity = float(inst["capacity"])
    cust = inst["customers"]
    routes = compact_routes(routes)
    if len(routes) <= 1:
        return routes

    for _ in range(max_passes):
        loads = route_loads(inst, routes)
        route_scores = [single_route_score(inst, route, tw_weight) for route in routes]
        best_move = None
        best_gain = 0.0

        for a_idx in range(len(routes)):
            route_a = routes[a_idx]
            for b_idx in range(a_idx + 1, len(routes)):
                route_b = routes[b_idx]
                old_score = route_scores[a_idx] + route_scores[b_idx]

                for a_pos in range(1, len(route_a) - 1):
                    cid_a = route_a[a_pos]
                    demand_a = float(cust[cid_a]["demand"])
                    a_without = route_a[:a_pos] + route_a[a_pos + 1 :]
                    for b_pos in range(1, len(route_b) - 1):
                        cid_b = route_b[b_pos]
                        demand_b = float(cust[cid_b]["demand"])
                        if loads[a_idx] - demand_a + demand_b > capacity + 1e-9:
                            continue
                        if loads[b_idx] - demand_b + demand_a > capacity + 1e-9:
                            continue
                        b_without = route_b[:b_pos] + route_b[b_pos + 1 :]
                        new_a = a_without[:a_pos] + [cid_b] + a_without[a_pos:]
                        new_b = b_without[:b_pos] + [cid_a] + b_without[b_pos:]
                        new_score = single_route_score(inst, new_a, tw_weight) + single_route_score(inst, new_b, tw_weight)
                        gain = old_score - new_score
                        if gain > best_gain + 1e-9:
                            best_gain = gain
                            best_move = (a_idx, a_pos, b_idx, b_pos, new_a, new_b)

        if best_move is None:
            break
        a_idx, a_pos, b_idx, b_pos, new_a, new_b = best_move
        routes[a_idx] = new_a
        routes[b_idx] = new_b
        routes = compact_routes(routes)

    return routes


def standalone_improve_routes(inst, routes, args):
    """Guarded repair loop applied to the top few LNS-produced route sets."""
    repair_tw_weight = args.repair_tw_weight if args.repair_tw_weight is not None else args.tw_weight
    routes = compact_routes(routes)
    routes = local_search_route_set(inst, routes, repair_tw_weight, max_passes=args.local_search_passes)
    for _ in range(max(1, args.inter_route_passes)):
        before_score = score_routes(inst, routes, repair_tw_weight)
        routes = cross_route_relocate_search(inst, routes, repair_tw_weight, max_passes=args.local_search_passes)
        routes = cross_route_swap_search(inst, routes, repair_tw_weight, max_passes=args.local_search_passes)
        routes = local_search_route_set(inst, routes, repair_tw_weight, max_passes=1)
        after_score = score_routes(inst, routes, repair_tw_weight)
        if after_score >= before_score - 1e-9:
            break
    return compact_routes(routes)
