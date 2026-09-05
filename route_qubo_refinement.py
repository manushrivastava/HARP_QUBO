"""Sliding-window QUBO refinement for already assigned VRPTW routes."""

from neal import SimulatedAnnealingSampler

from decomposed_vrptw_qubo import add_qubo, route_metrics
from new_qubo_objectives import pairwise_dist


def route_score(inst, route, tw_weight=6.0):
    metrics = route_metrics(inst, route)
    return float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])


def departure_time_before_index(inst, route, index):
    """Return simulator time just before traveling into route[index]."""
    D = pairwise_dist(inst)
    cust = inst["customers"]
    time_now = 0.0
    for prev, cid in zip(route[: index - 1], route[1:index]):
        travel = D[(prev, cid)]
        if cid == 0:
            time_now += travel
            continue
        arrival = time_now + travel
        start = max(arrival, cust[cid]["ready"])
        time_now = start + cust[cid]["service"]
    return time_now


def decode_window_sample(sample, y, customers, m):
    order = []
    used = set()
    for pos in range(m):
        chosen = [cid for cid in customers if int(sample.get(y[(cid, pos)], 0)) == 1]
        if len(chosen) != 1:
            return None
        cid = chosen[0]
        if cid in used:
            return None
        used.add(cid)
        order.append(cid)
    return order


def solve_window_reorder_qubo(
    inst,
    route,
    start_index,
    window_size=6,
    constraint_weight=120.0,
    distance_weight=1.0,
    tw_weight=6.0,
    score_tw_weight=None,
    due_order_weight=0.15,
    num_reads=300,
    num_sweeps=500,
    seed=0,
):
    """Try to improve one fixed-boundary route window with a small ordering QUBO."""
    if score_tw_weight is None:
        score_tw_weight = tw_weight

    end_index = min(start_index + window_size, len(route) - 1)
    customers = [cid for cid in route[start_index:end_index] if cid != 0]
    m = len(customers)
    if m <= 2:
        return route, {"accepted": False, "valid_samples": 0, "vars": m * m}

    prev_cid = route[start_index - 1]
    next_cid = route[end_index]
    prefix_time = departure_time_before_index(inst, route, start_index)
    D = pairwise_dist(inst)
    cust = inst["customers"]

    y = {(cid, pos): f"w_{cid}_{pos}" for cid in customers for pos in range(m)}
    Q = {}

    # Every customer appears once.
    for cid in customers:
        for pos in range(m):
            add_qubo(Q, y[(cid, pos)], y[(cid, pos)], -constraint_weight)
        for a in range(m):
            for b in range(a + 1, m):
                add_qubo(Q, y[(cid, a)], y[(cid, b)], 2.0 * constraint_weight)

    # Every position has one customer.
    for pos in range(m):
        for cid in customers:
            add_qubo(Q, y[(cid, pos)], y[(cid, pos)], -constraint_weight)
        for a in range(m):
            for b in range(a + 1, m):
                add_qubo(Q, y[(customers[a], pos)], y[(customers[b], pos)], 2.0 * constraint_weight)

    # First/last fixed-boundary costs.
    for cid in customers:
        arrival = prefix_time + D[(prev_cid, cid)]
        start = max(arrival, cust[cid]["ready"])
        late = max(0.0, start - cust[cid]["due"])
        add_qubo(
            Q,
            y[(cid, 0)],
            y[(cid, 0)],
            distance_weight * D[(prev_cid, cid)] + tw_weight * late,
        )
        add_qubo(Q, y[(cid, m - 1)], y[(cid, m - 1)], distance_weight * D[(cid, next_cid)])

        if next_cid != 0:
            rough_depart = max(prefix_time, cust[cid]["ready"]) + cust[cid]["service"]
            arrival_next = rough_depart + D[(cid, next_cid)]
            start_next = max(arrival_next, cust[next_cid]["ready"])
            late_next = max(0.0, start_next - cust[next_cid]["due"])
            add_qubo(Q, y[(cid, m - 1)], y[(cid, m - 1)], tw_weight * late_next)

    avg_step = (
        sum(cust[cid]["service"] for cid in customers) / max(1, m)
        + sum(D[(i, j)] for i in customers for j in customers if i != j) / max(1, m * (m - 1))
    )

    # Adjacent transition and approximate time-window costs.
    for pos in range(m - 1):
        rough_time = prefix_time + avg_step * pos
        for i in customers:
            start_i = max(rough_time, cust[i]["ready"])
            depart_i = start_i + cust[i]["service"]
            for j in customers:
                if i == j:
                    continue
                arrival_j = depart_i + D[(i, j)]
                start_j = max(arrival_j, cust[j]["ready"])
                late_j = max(0.0, start_j - cust[j]["due"])
                add_qubo(
                    Q,
                    y[(i, pos)],
                    y[(j, pos + 1)],
                    distance_weight * D[(i, j)] + tw_weight * late_j,
                )

    # Soft precedence: earlier due dates prefer earlier positions.
    if due_order_weight > 0:
        for i in customers:
            for j in customers:
                if i == j or cust[i]["due"] >= cust[j]["due"]:
                    continue
                due_gap = min(1.0, (cust[j]["due"] - cust[i]["due"]) / 100.0)
                penalty = due_order_weight * due_gap
                for late_pos in range(1, m):
                    for early_pos in range(late_pos):
                        add_qubo(Q, y[(i, late_pos)], y[(j, early_pos)], penalty)

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    original_score = route_score(inst, route, tw_weight=score_tw_weight)
    best_route = route
    best_score = original_score
    valid_samples = 0

    for row in sampleset.data(["sample", "energy"]):
        order = decode_window_sample(row.sample, y, customers, m)
        if order is None:
            continue
        valid_samples += 1
        candidate = route[:start_index] + order + route[end_index:]
        candidate_score = route_score(inst, candidate, tw_weight=score_tw_weight)
        if candidate_score + 1e-9 < best_score:
            best_route = candidate
            best_score = candidate_score

    return best_route, {
        "accepted": best_route != route,
        "valid_samples": valid_samples,
        "vars": m * m,
        "old_score": original_score,
        "new_score": best_score,
    }


def refine_route_sliding_qubo(
    inst,
    route,
    tw_weight=6.0,
    qubo_tw_weight=None,
    window_size=6,
    passes=2,
    constraint_weight=120.0,
    distance_weight=1.0,
    due_order_weight=0.15,
    num_reads=300,
    num_sweeps=500,
    seed=0,
):
    """Apply accepted sliding-window QUBO reorderings to one route."""
    if qubo_tw_weight is None:
        qubo_tw_weight = tw_weight

    refined = list(route)
    total_windows = 0
    accepted_windows = 0
    valid_samples = 0
    max_vars = 0

    for pass_idx in range(passes):
        changed = False
        last_start = max(1, len(refined) - 1)
        for start_index in range(1, last_start):
            if start_index >= len(refined) - 2:
                continue
            candidate, info = solve_window_reorder_qubo(
                inst,
                refined,
                start_index,
                window_size=window_size,
                constraint_weight=constraint_weight,
                distance_weight=distance_weight,
                tw_weight=qubo_tw_weight,
                score_tw_weight=tw_weight,
                due_order_weight=due_order_weight,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed + 1000 * pass_idx + start_index,
            )
            total_windows += 1
            valid_samples += info["valid_samples"]
            max_vars = max(max_vars, info["vars"])
            if info["accepted"]:
                refined = candidate
                accepted_windows += 1
                changed = True
        if not changed:
            break

    return refined, {
        "windows": total_windows,
        "accepted_windows": accepted_windows,
        "valid_samples": valid_samples,
        "max_vars": max_vars,
    }


def refine_routes_sliding_qubo(
    inst,
    routes,
    tw_weight=6.0,
    qubo_tw_weight=None,
    window_size=6,
    passes=2,
    constraint_weight=120.0,
    distance_weight=1.0,
    due_order_weight=0.15,
    num_reads=300,
    num_sweeps=500,
    seed=0,
):
    refined_routes = []
    infos = []
    for idx, route in enumerate(routes):
        refined, info = refine_route_sliding_qubo(
            inst,
            route,
            tw_weight=tw_weight,
            qubo_tw_weight=qubo_tw_weight,
            window_size=window_size,
            passes=passes,
            constraint_weight=constraint_weight,
            distance_weight=distance_weight,
            due_order_weight=due_order_weight,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed + 10000 * idx,
        )
        refined_routes.append(refined)
        infos.append(info)
    return refined_routes, infos


def intervals_conflict(a, b):
    return not (a["end"] < b["start"] or b["end"] < a["start"])


def solve_move_selection_qubo(moves, conflict_fn, num_reads=200, num_sweeps=300, seed=0):
    """Select a compatible set of candidate route moves with a small QUBO."""
    if not moves:
        return [], {"vars": 0, "selected": 0}

    max_gain = max(move["gain"] for move in moves) or 1.0
    Q = {}
    x = {idx: f"m_{idx}" for idx in range(len(moves))}

    for idx, move in enumerate(moves):
        add_qubo(Q, x[idx], x[idx], -move["gain"] / max_gain)

    for i in range(len(moves)):
        for j in range(i + 1, len(moves)):
            if conflict_fn(moves[i], moves[j]):
                add_qubo(Q, x[i], x[j], 2.0)

    sampleset = SimulatedAnnealingSampler().sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)
    selected_sets = []
    for row in sampleset.data(["sample", "energy"]):
        chosen = [idx for idx in range(len(moves)) if int(row.sample.get(x[idx], 0)) == 1]
        valid = True
        for a, idx in enumerate(chosen):
            for other in chosen[a + 1 :]:
                if conflict_fn(moves[idx], moves[other]):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            selected_sets.append(chosen)
    return selected_sets, {"vars": len(moves), "selected": max((len(s) for s in selected_sets), default=0)}


def apply_two_opt_moves(route, moves):
    candidate = list(route)
    for move in sorted(moves, key=lambda item: item["start"]):
        start = move["start"]
        end = move["end"]
        candidate[start : end + 1] = reversed(candidate[start : end + 1])
    return candidate


def two_opt_candidates(inst, route, tw_weight=6.0, max_candidates=40):
    base_score = route_score(inst, route, tw_weight=tw_weight)
    moves = []
    last_customer_index = len(route) - 2
    for start in range(1, last_customer_index):
        for end in range(start + 1, last_customer_index + 1):
            candidate = route[:start] + list(reversed(route[start : end + 1])) + route[end + 1 :]
            candidate_score = route_score(inst, candidate, tw_weight=tw_weight)
            gain = base_score - candidate_score
            if gain > 1e-9:
                moves.append(
                    {
                        "type": "two_opt",
                        "start": start,
                        "end": end,
                        "gain": gain,
                        "candidate_score": candidate_score,
                    }
                )
    moves.sort(key=lambda item: item["gain"], reverse=True)
    return moves[:max_candidates]


def refine_route_two_opt_move_qubo(
    inst,
    route,
    tw_weight=6.0,
    passes=3,
    max_candidates=40,
    num_reads=200,
    num_sweeps=300,
    seed=0,
):
    refined = list(route)
    total_moves = 0
    accepted_moves = 0
    max_vars = 0

    for pass_idx in range(passes):
        moves = two_opt_candidates(inst, refined, tw_weight=tw_weight, max_candidates=max_candidates)
        total_moves += len(moves)
        if not moves:
            break
        selected_sets, info = solve_move_selection_qubo(
            moves,
            intervals_conflict,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed + pass_idx,
        )
        max_vars = max(max_vars, info["vars"])
        current_score = route_score(inst, refined, tw_weight=tw_weight)
        best_route = refined
        best_score = current_score
        best_selected = []
        for selected in selected_sets:
            selected_moves = [moves[idx] for idx in selected]
            candidate = apply_two_opt_moves(refined, selected_moves)
            candidate_score = route_score(inst, candidate, tw_weight=tw_weight)
            if candidate_score + 1e-9 < best_score:
                best_route = candidate
                best_score = candidate_score
                best_selected = selected_moves
        if best_route == refined:
            break
        refined = best_route
        accepted_moves += len(best_selected)

    return refined, {"moves": total_moves, "accepted_moves": accepted_moves, "max_vars": max_vars}


def apply_relocate_or_swap_move(route, move):
    candidate = list(route)
    if move["type"] == "swap":
        candidate[move["i"]], candidate[move["j"]] = candidate[move["j"]], candidate[move["i"]]
        return candidate

    if move["type"] == "relocate":
        i = move["i"]
        j = move["j"]
        cid = candidate.pop(i)
        if j > i:
            j -= 1
        candidate.insert(j, cid)
        return candidate

    raise ValueError(f"Unknown move type: {move['type']}")


def apply_relocate_swap_moves(route, moves):
    candidate = list(route)
    for move in sorted(moves, key=lambda item: item["end"], reverse=True):
        candidate = apply_relocate_or_swap_move(candidate, move)
    return candidate


def relocate_swap_candidates(inst, route, tw_weight=6.0, max_candidates=50):
    base_score = route_score(inst, route, tw_weight=tw_weight)
    moves = []
    last_customer_index = len(route) - 2

    for i in range(1, last_customer_index + 1):
        for j in range(i + 1, last_customer_index + 1):
            candidate = list(route)
            candidate[i], candidate[j] = candidate[j], candidate[i]
            candidate_score = route_score(inst, candidate, tw_weight=tw_weight)
            gain = base_score - candidate_score
            if gain > 1e-9:
                moves.append(
                    {
                        "type": "swap",
                        "i": i,
                        "j": j,
                        "start": i,
                        "end": j,
                        "gain": gain,
                        "candidate_score": candidate_score,
                    }
                )

    for i in range(1, last_customer_index + 1):
        for j in range(1, last_customer_index + 2):
            if j == i or j == i + 1:
                continue
            candidate = apply_relocate_or_swap_move(route, {"type": "relocate", "i": i, "j": j})
            candidate_score = route_score(inst, candidate, tw_weight=tw_weight)
            gain = base_score - candidate_score
            if gain > 1e-9:
                moves.append(
                    {
                        "type": "relocate",
                        "i": i,
                        "j": j,
                        "start": min(i, j),
                        "end": max(i, j),
                        "gain": gain,
                        "candidate_score": candidate_score,
                    }
                )

    moves.sort(key=lambda item: item["gain"], reverse=True)
    return moves[:max_candidates]


def refine_route_relocate_swap_move_qubo(
    inst,
    route,
    tw_weight=6.0,
    passes=3,
    max_candidates=50,
    num_reads=200,
    num_sweeps=300,
    seed=0,
):
    refined = list(route)
    total_moves = 0
    accepted_moves = 0
    max_vars = 0

    for pass_idx in range(passes):
        moves = relocate_swap_candidates(inst, refined, tw_weight=tw_weight, max_candidates=max_candidates)
        total_moves += len(moves)
        if not moves:
            break
        selected_sets, info = solve_move_selection_qubo(
            moves,
            intervals_conflict,
            num_reads=num_reads,
            num_sweeps=num_sweeps,
            seed=seed + pass_idx,
        )
        max_vars = max(max_vars, info["vars"])
        current_score = route_score(inst, refined, tw_weight=tw_weight)
        best_route = refined
        best_score = current_score
        best_selected = []
        for selected in selected_sets:
            selected_moves = [moves[idx] for idx in selected]
            candidate = apply_relocate_swap_moves(refined, selected_moves)
            candidate_score = route_score(inst, candidate, tw_weight=tw_weight)
            if candidate_score + 1e-9 < best_score:
                best_route = candidate
                best_score = candidate_score
                best_selected = selected_moves
        if best_route == refined:
            break
        refined = best_route
        accepted_moves += len(best_selected)

    return refined, {"moves": total_moves, "accepted_moves": accepted_moves, "max_vars": max_vars}


def refine_routes_move_qubo(
    inst,
    routes,
    method,
    tw_weight=6.0,
    passes=3,
    max_candidates=50,
    num_reads=200,
    num_sweeps=300,
    seed=0,
):
    refined_routes = []
    infos = []
    for idx, route in enumerate(routes):
        if method == "two_opt":
            refined, info = refine_route_two_opt_move_qubo(
                inst,
                route,
                tw_weight=tw_weight,
                passes=passes,
                max_candidates=max_candidates,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed + 10000 * idx,
            )
        elif method == "relocate_swap":
            refined, info = refine_route_relocate_swap_move_qubo(
                inst,
                route,
                tw_weight=tw_weight,
                passes=passes,
                max_candidates=max_candidates,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                seed=seed + 10000 * idx,
            )
        else:
            raise ValueError(f"Unknown move QUBO method: {method}")
        refined_routes.append(refined)
        infos.append(info)
    return refined_routes, infos
