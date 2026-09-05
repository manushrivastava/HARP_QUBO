"""Route-pool set-partitioning QUBO for VRPTW.

This experiment addresses cases where customer-to-vehicle assignment creates
clusters whose internal route timing is structurally late. Instead of assigning
customers directly to vehicles, it generates a pool of complete candidate routes
with route-level time-window scores, then solves a QUBO that selects routes to
cover every customer exactly once.
"""

import argparse
import csv
import math
import os
import random
import statistics
import time

from neal import SimulatedAnnealingSampler

from compare_with_ortools import solve_case
from decomposed_vrptw_qubo import add_qubo, evaluate_routes, route_metrics
from new_qubo_objectives import load_solomon_txt, pairwise_dist, reduce_instance
from route_qubo_refinement import refine_routes_sliding_qubo

try:
    from tqdm.auto import tqdm
except ImportError:  # tqdm is optional; scripts still run without progress bars.
    tqdm = None


def progress_iter(iterable, enabled=False, **kwargs):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def parse_cases(value):
    cases = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        instance, keep = part.split(":", 1)
        cases.append((instance.strip(), int(keep)))
    return cases


def parse_csv_list(value, cast=str):
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def route_score(inst, route, tw_weight):
    metrics = route_metrics(inst, route)
    return float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])


def routes_score(inst, routes, tw_weight):
    metrics = evaluate_routes(inst, routes)
    return float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])


def route_load(inst, route):
    return sum(float(inst["customers"][cid]["demand"]) for cid in route if cid != 0)


def route_key(route):
    return tuple(cid for cid in route if cid != 0)


def ordered_customers(inst, key_name, rng=None):
    rng = rng or random.Random(0)
    kept = list(inst["kept_ids"])
    cust = inst["customers"]
    coords = inst["coords"]
    depot = coords[0]
    D = pairwise_dist(inst)

    def midpoint(cid):
        return 0.5 * (float(cust[cid]["ready"]) + float(cust[cid]["due"]))

    def angle(cid):
        x, y = coords[cid]
        value = math.atan2(y - depot[1], x - depot[0])
        return value if value >= 0 else value + 2.0 * math.pi

    if key_name == "random":
        rng.shuffle(kept)
        return kept
    if key_name == "ready":
        return sorted(kept, key=lambda cid: (cust[cid]["ready"], cust[cid]["due"], cid))
    if key_name == "due":
        return sorted(kept, key=lambda cid: (cust[cid]["due"], cust[cid]["ready"], cid))
    if key_name == "midpoint":
        return sorted(kept, key=lambda cid: (midpoint(cid), cust[cid]["due"], cid))
    if key_name == "angle":
        return sorted(kept, key=lambda cid: (angle(cid), midpoint(cid), cid))
    if key_name == "distance":
        return sorted(kept, key=lambda cid: (D[(0, cid)], midpoint(cid), cid))
    if key_name == "far_due":
        return sorted(kept, key=lambda cid: (cust[cid]["due"], -D[(0, cid)], cid))
    if key_name == "x":
        return sorted(kept, key=lambda cid: (coords[cid][0], coords[cid][1], cid))
    if key_name == "y":
        return sorted(kept, key=lambda cid: (coords[cid][1], coords[cid][0], cid))
    raise ValueError(f"Unknown order key: {key_name}")


def split_order_into_k(inst, order):
    K = inst["n_vehicles"]
    cust = inst["customers"]
    capacity = float(inst["capacity"])
    total_demand = sum(float(cust[cid]["demand"]) for cid in order)
    groups = []
    current = []
    load = 0.0
    remaining_demand = total_demand

    for idx, cid in enumerate(order):
        demand = float(cust[cid]["demand"])
        remaining_routes = max(1, K - len(groups))
        target = remaining_demand / remaining_routes
        customers_left = len(order) - idx
        can_close = current and len(groups) < K - 1 and customers_left > remaining_routes - 1
        if can_close and load + demand > max(target, 0.7 * capacity):
            groups.append(current)
            remaining_demand -= load
            current = []
            load = 0.0
        if load + demand > capacity + 1e-9 and current and len(groups) < K - 1:
            groups.append(current)
            remaining_demand -= load
            current = []
            load = 0.0
        current.append(cid)
        load += demand

    if current:
        groups.append(current)
    if len(groups) != K:
        return None
    if any(sum(float(cust[cid]["demand"]) for cid in group) > capacity + 1e-9 for group in groups):
        return None
    return groups


def directed_lateness_from_state(inst, D, cid, time_now, prev):
    cust = inst["customers"]
    arrival = time_now + D[(prev, cid)]
    start = max(arrival, float(cust[cid]["ready"]))
    late = max(0.0, start - float(cust[cid]["due"]))
    wait = max(0.0, float(cust[cid]["ready"]) - arrival)
    finish = start + float(cust[cid]["service"])
    return late, wait, finish


def time_greedy_route(inst, cluster, late_weight=6.0, wait_weight=0.02, due_weight=0.0):
    D = pairwise_dist(inst)
    cust = inst["customers"]
    unvisited = set(cluster)
    route = [0]
    cur = 0
    time_now = 0.0
    horizon = max(float(cust[cid]["due"]) for cid in [0] + list(cluster)) or 1.0

    while unvisited:
        def key(cid):
            late, wait, _ = directed_lateness_from_state(inst, D, cid, time_now, cur)
            due_norm = float(cust[cid]["due"]) / horizon
            return D[(cur, cid)] + late_weight * late + wait_weight * wait + due_weight * due_norm

        nxt = min(unvisited, key=key)
        late, wait, finish = directed_lateness_from_state(inst, D, nxt, time_now, cur)
        route.append(nxt)
        cur = nxt
        time_now = finish
        unvisited.remove(nxt)

    route.append(0)
    return route


def sorted_route(inst, cluster, key_name):
    cust = inst["customers"]
    coords = inst["coords"]
    depot = coords[0]

    def midpoint(cid):
        return 0.5 * (float(cust[cid]["ready"]) + float(cust[cid]["due"]))

    def angle(cid):
        x, y = coords[cid]
        value = math.atan2(y - depot[1], x - depot[0])
        return value if value >= 0 else value + 2.0 * math.pi

    if key_name == "ready":
        order = sorted(cluster, key=lambda cid: (cust[cid]["ready"], cust[cid]["due"], cid))
    elif key_name == "due":
        order = sorted(cluster, key=lambda cid: (cust[cid]["due"], cust[cid]["ready"], cid))
    elif key_name == "midpoint":
        order = sorted(cluster, key=lambda cid: (midpoint(cid), cust[cid]["due"], cid))
    elif key_name == "angle":
        order = sorted(cluster, key=lambda cid: (angle(cid), midpoint(cid), cid))
    else:
        order = list(cluster)
    return [0] + order + [0]


def best_route_for_cluster(inst, cluster, tw_weight):
    cluster = list(cluster)
    if not cluster:
        return [0, 0]
    candidates = [
        sorted_route(inst, cluster, "ready"),
        sorted_route(inst, cluster, "due"),
        sorted_route(inst, cluster, "midpoint"),
        sorted_route(inst, cluster, "angle"),
        time_greedy_route(inst, cluster, late_weight=2.0, wait_weight=0.0, due_weight=0.0),
        time_greedy_route(inst, cluster, late_weight=6.0, wait_weight=0.01, due_weight=0.0),
        time_greedy_route(inst, cluster, late_weight=12.0, wait_weight=0.02, due_weight=0.1),
        time_greedy_route(inst, cluster, late_weight=24.0, wait_weight=0.02, due_weight=0.2),
        time_greedy_route(inst, cluster, late_weight=60.0, wait_weight=0.02, due_weight=0.3),
    ]
    return min(candidates, key=lambda route: route_score(inst, route, tw_weight))


def anchor_partition(inst, key_name, rng):
    K = inst["n_vehicles"]
    cust = inst["customers"]
    capacity = float(inst["capacity"])
    D = pairwise_dist(inst)
    ordered = ordered_customers(inst, key_name, rng)
    if len(ordered) < K:
        return None

    anchors = []
    for k in range(K):
        idx = round(k * (len(ordered) - 1) / max(1, K - 1))
        anchors.append(ordered[int(idx)])

    groups = [[anchor] for anchor in anchors]
    loads = [float(cust[anchor]["demand"]) for anchor in anchors]
    assigned = set(anchors)
    horizon = max(float(cust[cid]["due"]) for cid in [0] + inst["kept_ids"]) or 1.0

    def midpoint(cid):
        return 0.5 * (float(cust[cid]["ready"]) + float(cust[cid]["due"]))

    for cid in [c for c in ordered if c not in assigned]:
        demand = float(cust[cid]["demand"])
        best_k = None
        best_cost = float("inf")
        for k, anchor in enumerate(anchors):
            if loads[k] + demand > capacity + 1e-9:
                continue
            time_gap = abs(midpoint(cid) - midpoint(anchor)) / horizon
            due_gap = abs(float(cust[cid]["due"]) - float(cust[anchor]["due"])) / horizon
            cost = D[(cid, anchor)] + 50.0 * time_gap + 20.0 * due_gap + 0.05 * loads[k]
            if cost < best_cost:
                best_cost = cost
                best_k = k
        if best_k is None:
            return None
        groups[best_k].append(cid)
        loads[best_k] += demand
    return groups


def insertion_delta(inst, route, cid, tw_weight):
    old_score = route_score(inst, route, tw_weight)
    best = None
    for pos in range(1, len(route)):
        candidate = route[:pos] + [cid] + route[pos:]
        score = route_score(inst, candidate, tw_weight)
        delta = score - old_score
        if best is None or delta < best["delta"]:
            best = {"route": candidate, "delta": delta, "score": score, "position": pos}
    return best


def insertion_partition(inst, key_name, mode, tw_weight, rng):
    K = inst["n_vehicles"]
    cust = inst["customers"]
    capacity = float(inst["capacity"])
    ordered = ordered_customers(inst, key_name, rng)
    if len(ordered) < K:
        return None

    anchors = []
    for k in range(K):
        idx = round(k * (len(ordered) - 1) / max(1, K - 1))
        anchors.append(ordered[int(idx)])
    anchors = list(dict.fromkeys(anchors))
    if len(anchors) < K:
        for cid in ordered:
            if cid not in anchors:
                anchors.append(cid)
            if len(anchors) == K:
                break

    routes = [[0, anchor, 0] for anchor in anchors]
    loads = [float(cust[anchor]["demand"]) for anchor in anchors]
    remaining = [cid for cid in ordered if cid not in anchors]
    if mode == "reverse":
        remaining = list(reversed(remaining))
    elif mode == "random":
        rng.shuffle(remaining)

    while remaining:
        choices = []
        for cid in remaining:
            demand = float(cust[cid]["demand"])
            feasible = []
            for k, route in enumerate(routes):
                if loads[k] + demand > capacity + 1e-9:
                    continue
                insert = insertion_delta(inst, route, cid, tw_weight)
                feasible.append((insert["delta"], insert["score"], k, insert["route"]))
            if not feasible:
                continue
            feasible.sort(key=lambda item: item[0])
            best = feasible[0]
            regret = (feasible[1][0] - best[0]) if len(feasible) > 1 else 1e6
            choices.append(
                {
                    "cid": cid,
                    "delta": best[0],
                    "new_score": best[1],
                    "route_index": best[2],
                    "route": best[3],
                    "regret": regret,
                }
            )
        if not choices:
            return None

        if mode == "global_best":
            choice = min(choices, key=lambda item: (item["delta"], item["new_score"]))
        elif mode == "regret":
            choice = max(choices, key=lambda item: (item["regret"], -item["delta"]))
        elif mode == "random":
            top = sorted(choices, key=lambda item: item["delta"])[: max(1, min(5, len(choices)))]
            choice = rng.choice(top)
        else:
            next_cid = remaining[0]
            matching = [item for item in choices if item["cid"] == next_cid]
            choice = matching[0] if matching else min(choices, key=lambda item: item["delta"])

        k = choice["route_index"]
        routes[k] = choice["route"]
        loads[k] += float(cust[choice["cid"]]["demand"])
        remaining.remove(choice["cid"])

    return [route_key(route) for route in routes]


def build_route_pool(inst, tw_weight=6.0, random_partitions=40, max_routes=350, seed=100, progress=False):
    rng = random.Random(seed)
    route_map = {}
    partition_rows = []
    keys = ["ready", "due", "midpoint", "angle", "distance", "far_due", "x", "y"]

    def add_partition(groups, source):
        if groups is None:
            return
        routes = [best_route_for_cluster(inst, group, tw_weight) for group in groups]
        if any(route_load(inst, route) > inst["capacity"] + 1e-9 for route in routes):
            return
        partition_rows.append({"source": source, "routes": routes, "score": routes_score(inst, routes, tw_weight)})
        for route in routes:
            customers = route_key(route)
            if not customers:
                continue
            score = route_score(inst, route, tw_weight)
            old = route_map.get(customers)
            if old is None or score < old["score"]:
                route_map[customers] = {
                    "route": route,
                    "score": score,
                    "distance": route_metrics(inst, route)["distance"],
                    "tw_violation_amount": route_metrics(inst, route)["tw_violation_amount"],
                    "source": source,
                }

    construction_weights = [tw_weight, max(tw_weight, 12.0), max(tw_weight, 24.0), max(tw_weight, 60.0)]

    for key in progress_iter(keys, enabled=progress, desc="route-pool deterministic", unit="key"):
        add_partition(split_order_into_k(inst, ordered_customers(inst, key, rng)), f"split_{key}")
        add_partition(anchor_partition(inst, key, rng), f"anchor_{key}")
        for construction_tw in construction_weights:
            for mode in ["ordered", "reverse", "global_best", "regret"]:
                add_partition(
                    insertion_partition(inst, key, mode, construction_tw, rng),
                    f"insertion_{key}_{mode}_tw{construction_tw:g}",
                )

    for idx in progress_iter(range(random_partitions), enabled=progress, desc="route-pool random", unit="partition"):
        add_partition(split_order_into_k(inst, ordered_customers(inst, "random", rng)), f"random_split_{idx}")
        add_partition(anchor_partition(inst, rng.choice(keys), rng), f"random_anchor_{idx}")
        construction_tw = rng.choice(construction_weights)
        add_partition(
            insertion_partition(inst, rng.choice(keys), rng.choice(["random", "global_best", "regret"]), construction_tw, rng),
            f"random_insertion_{idx}_tw{construction_tw:g}",
        )

    routes = sorted(route_map.values(), key=lambda row: row["score"])
    if len(routes) > max_routes:
        routes = routes[:max_routes]
    return routes, sorted(partition_rows, key=lambda row: row["score"])


def build_route_pool_qubo(inst, route_pool, tw_weight, coverage_weight, vehicle_weight, route_score_scale):
    Q = {}
    y = {idx: f"r_{idx}" for idx in range(len(route_pool))}
    K = inst["n_vehicles"]

    for idx, row in enumerate(route_pool):
        add_qubo(Q, y[idx], y[idx], row["score"] / route_score_scale)

    for cid in inst["kept_ids"]:
        containing = [idx for idx, row in enumerate(route_pool) if cid in route_key(row["route"])]
        for idx in containing:
            add_qubo(Q, y[idx], y[idx], -coverage_weight)
        for a, idx in enumerate(containing):
            for jdx in containing[a + 1 :]:
                add_qubo(Q, y[idx], y[jdx], 2.0 * coverage_weight)

    for idx in range(len(route_pool)):
        add_qubo(Q, y[idx], y[idx], vehicle_weight * (1.0 - 2.0 * K))
    for i in range(len(route_pool)):
        for j in range(i + 1, len(route_pool)):
            add_qubo(Q, y[i], y[j], 2.0 * vehicle_weight)

    return Q, y


def decode_selected(sample, y):
    return [idx for idx, name in y.items() if int(sample.get(name, 0)) == 1]


def selection_valid(inst, route_pool, selected):
    if len(selected) != inst["n_vehicles"]:
        return False
    counts = {cid: 0 for cid in inst["kept_ids"]}
    for idx in selected:
        for cid in route_key(route_pool[idx]["route"]):
            counts[cid] = counts.get(cid, 0) + 1
    return all(counts[cid] == 1 for cid in inst["kept_ids"])


def selection_diagnostics(inst, route_pool, selected):
    counts = {cid: 0 for cid in inst["kept_ids"]}
    for idx in selected:
        for cid in route_key(route_pool[idx]["route"]):
            counts[cid] = counts.get(cid, 0) + 1
    uncovered = sum(1 for cid in inst["kept_ids"] if counts[cid] == 0)
    duplicate = sum(1 for cid in inst["kept_ids"] if counts[cid] > 1)
    return {
        "selected_route_count": len(selected),
        "route_count_error": abs(len(selected) - inst["n_vehicles"]),
        "uncovered_customers": uncovered,
        "duplicate_customers": duplicate,
        "coverage_error": uncovered + duplicate,
    }


def solve_route_pool_qubo(inst, route_pool, args):
    Q, y = build_route_pool_qubo(
        inst,
        route_pool,
        tw_weight=args.tw_weight,
        coverage_weight=args.coverage_weight,
        vehicle_weight=args.vehicle_weight,
        route_score_scale=args.route_score_scale,
    )
    if getattr(args, "progress", False):
        print(
            f"Sampling route-pool QUBO: vars={len(route_pool)} terms={len(Q)} "
            f"reads={args.num_reads} sweeps={args.num_sweeps}",
            flush=True,
        )
    sampleset = SimulatedAnnealingSampler().sample_qubo(
        Q,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        seed=args.seed,
    )
    if getattr(args, "progress", False):
        print("Decoding route-pool QUBO samples", flush=True)

    best = None
    valid_samples = 0
    invalid_route_count_samples = 0
    invalid_coverage_samples = 0
    invalid_both_samples = 0
    total_route_count_error = 0
    total_coverage_error = 0
    for row in progress_iter(
        sampleset.data(["sample", "energy"]),
        enabled=getattr(args, "progress", False),
        desc="decode QUBO samples",
        total=len(sampleset),
        unit="sample",
    ):
        selected = decode_selected(row.sample, y)
        diagnostics = selection_diagnostics(inst, route_pool, selected)
        route_count_bad = diagnostics["route_count_error"] > 0
        coverage_bad = diagnostics["coverage_error"] > 0
        if route_count_bad or coverage_bad:
            if route_count_bad and coverage_bad:
                invalid_both_samples += 1
            elif route_count_bad:
                invalid_route_count_samples += 1
            else:
                invalid_coverage_samples += 1
            total_route_count_error += diagnostics["route_count_error"]
            total_coverage_error += diagnostics["coverage_error"]
            continue
        valid_samples += 1
        routes = [route_pool[idx]["route"] for idx in selected]
        score = routes_score(inst, routes, args.tw_weight)
        if best is None or score < best["score"]:
            best = {
                "routes": routes,
                "score": score,
                "energy": float(row.energy),
                "selected": selected,
            }

    return best, {
        "vars": len(route_pool),
        "valid_samples": valid_samples,
        "qubo_terms": len(Q),
        "invalid_route_count_samples": invalid_route_count_samples,
        "invalid_coverage_samples": invalid_coverage_samples,
        "invalid_both_samples": invalid_both_samples,
        "mean_route_count_error": total_route_count_error / max(len(sampleset), 1),
        "mean_coverage_error": total_coverage_error / max(len(sampleset), 1),
    }


def route_masks(inst, route_pool):
    index = {cid: bit for bit, cid in enumerate(inst["kept_ids"])}
    masks = []
    for row in route_pool:
        mask = 0
        for cid in route_key(row["route"]):
            mask |= 1 << index[cid]
        masks.append(mask)
    return masks


def solve_route_pool_exact(inst, route_pool, tw_weight):
    masks = route_masks(inst, route_pool)
    target = (1 << len(inst["kept_ids"])) - 1
    containing = {cid: [] for cid in inst["kept_ids"]}
    cid_to_bit = {cid: bit for bit, cid in enumerate(inst["kept_ids"])}
    for idx, mask in enumerate(masks):
        for cid, bit in cid_to_bit.items():
            if mask & (1 << bit):
                containing[cid].append(idx)
    for cid in containing:
        containing[cid].sort(key=lambda idx: route_pool[idx]["score"])

    K = inst["n_vehicles"]
    best = {"score": float("inf"), "selected": None}

    def search(selected, covered, score_so_far):
        if score_so_far >= best["score"]:
            return
        if len(selected) == K:
            if covered == target:
                best["score"] = score_so_far
                best["selected"] = list(selected)
            return
        if covered == target:
            return
        uncovered_bits = [bit for bit in range(len(inst["kept_ids"])) if not (covered & (1 << bit))]
        if not uncovered_bits:
            return
        next_bit = uncovered_bits[0]
        next_cid = inst["kept_ids"][next_bit]
        for idx in containing[next_cid]:
            mask = masks[idx]
            if mask & covered:
                continue
            remaining_slots = K - len(selected) - 1
            remaining_uncovered = target & ~(covered | mask)
            if remaining_slots == 0 and remaining_uncovered:
                continue
            search(selected + [idx], covered | mask, score_so_far + route_pool[idx]["score"])

    search([], 0, 0.0)
    if best["selected"] is None:
        return None
    routes = [route_pool[idx]["route"] for idx in best["selected"]]
    return {
        "routes": routes,
        "score": routes_score(inst, routes, tw_weight),
        "selected": best["selected"],
    }


def routes_text(routes):
    return " ; ".join("-".join(str(cid) for cid in route) for route in routes)


def add_result_row(rows, case, method, routes, elapsed, args, extra=None):
    extra = extra or {}
    metrics = evaluate_routes(args.inst, routes)
    rows.append(
        {
            "instance": case[0],
            "keep": case[1],
            "method": method,
            "distance": round(metrics["distance"], 6),
            "tw_violations": metrics["tw_violations"],
            "tw_violation_amount": round(metrics["tw_violation_amount"], 6),
            "score": round(metrics["distance"] + args.tw_weight * metrics["tw_violation_amount"], 6),
            "elapsed_sec": round(elapsed, 6),
            "route_pool_size": extra.get("route_pool_size", ""),
            "valid_samples": extra.get("valid_samples", ""),
            "qubo_terms": extra.get("qubo_terms", ""),
            "invalid_route_count_samples": extra.get("invalid_route_count_samples", ""),
            "invalid_coverage_samples": extra.get("invalid_coverage_samples", ""),
            "invalid_both_samples": extra.get("invalid_both_samples", ""),
            "mean_route_count_error": extra.get("mean_route_count_error", ""),
            "mean_coverage_error": extra.get("mean_coverage_error", ""),
            "selected": extra.get("selected", ""),
            "route_sizes": "|".join(str(max(0, len(route) - 2)) for route in routes),
            "routes": routes_text(routes),
        }
    )


def write_outputs(rows, output):
    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.splitext(output)[0] + "_summary.md"
    ortools = next((row for row in rows if row["method"] == "ortools_same_k_soft_tw"), None)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Route-Pool QUBO VRPTW Summary\n\n")
        f.write(f"Source CSV: `{output}`\n\n")
        f.write("| method | score | distance | TW violation | gap to OR-Tools |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            gap = ""
            if ortools and row["score"] != "":
                gap = round(float(row["score"]) - float(ortools["score"]), 6)
            f.write(
                f"| {row['method']} | {row['score']} | {row['distance']} | "
                f"{row['tw_violation_amount']} | {gap} |\n"
            )
    return summary_path


def run_case(case, args):
    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{case[0]}.txt"))
    inst = reduce_instance(sol, keep=case[1], slack_vehicles=args.slack_vehicles)
    args.inst = inst
    rows = []

    started = time.perf_counter()
    ort = solve_case(
        inst,
        vehicles=inst["n_vehicles"],
        hard_time_windows=False,
        time_limit_sec=args.ortools_time_limit,
        soft_lateness_penalty=args.tw_weight,
        scale=args.ortools_scale,
    )
    if ort["solved"]:
        add_result_row(rows, case, "ortools_same_k_soft_tw", ort["routes"], time.perf_counter() - started, args)
        refine_started = time.perf_counter()
        ort_refined, _ = refine_routes_sliding_qubo(
            inst,
            ort["routes"],
            tw_weight=args.tw_weight,
            qubo_tw_weight=args.refine_qubo_tw_weight,
            window_size=args.refine_window_size,
            passes=args.refine_passes,
            constraint_weight=args.refine_constraint_weight,
            distance_weight=args.refine_distance_weight,
            due_order_weight=args.refine_due_order_weight,
            num_reads=args.refine_reads,
            num_sweeps=args.refine_sweeps,
            seed=args.seed + args.refine_seed_offset + 50000,
        )
        add_result_row(
            rows,
            case,
            "ortools_seed_sliding_qubo_refined",
            ort_refined,
            time.perf_counter() - refine_started,
            args,
        )

    started = time.perf_counter()
    route_pool, partitions = build_route_pool(
        inst,
        tw_weight=args.tw_weight,
        random_partitions=args.random_partitions,
        max_routes=args.max_routes,
        seed=args.seed,
    )
    pool_elapsed = time.perf_counter() - started
    if partitions:
        best_partition = partitions[0]["routes"]
        add_result_row(
            rows,
            case,
            "best_heuristic_partition",
            best_partition,
            pool_elapsed,
            args,
            {"route_pool_size": len(route_pool)},
        )

    exact_started = time.perf_counter()
    exact = solve_route_pool_exact(inst, route_pool, args.tw_weight)
    if exact is not None:
        add_result_row(
            rows,
            case,
            "route_pool_exact_best",
            exact["routes"],
            time.perf_counter() - exact_started,
            args,
            {
                "route_pool_size": len(route_pool),
                "selected": "|".join(str(idx) for idx in exact["selected"]),
            },
        )

    started = time.perf_counter()
    best, info = solve_route_pool_qubo(inst, route_pool, args)
    solve_elapsed = time.perf_counter() - started
    if best is not None:
        add_result_row(
            rows,
            case,
            "route_pool_set_partition_qubo",
            best["routes"],
            solve_elapsed,
            args,
            {
                "route_pool_size": len(route_pool),
                "valid_samples": info["valid_samples"],
                "qubo_terms": info["qubo_terms"],
                "invalid_route_count_samples": info.get("invalid_route_count_samples", ""),
                "invalid_coverage_samples": info.get("invalid_coverage_samples", ""),
                "invalid_both_samples": info.get("invalid_both_samples", ""),
                "mean_route_count_error": info.get("mean_route_count_error", ""),
                "mean_coverage_error": info.get("mean_coverage_error", ""),
                "selected": "|".join(str(idx) for idx in best["selected"]),
            },
        )

        refined_started = time.perf_counter()
        refined, refine_infos = refine_routes_sliding_qubo(
            inst,
            best["routes"],
            tw_weight=args.tw_weight,
            qubo_tw_weight=args.refine_qubo_tw_weight,
            window_size=args.refine_window_size,
            passes=args.refine_passes,
            constraint_weight=args.refine_constraint_weight,
            distance_weight=args.refine_distance_weight,
            due_order_weight=args.refine_due_order_weight,
            num_reads=args.refine_reads,
            num_sweeps=args.refine_sweeps,
            seed=args.seed + args.refine_seed_offset,
        )
        add_result_row(
            rows,
            case,
            "route_pool_qubo_sliding_refined",
            refined,
            time.perf_counter() - refined_started,
            args,
            {
                "route_pool_size": len(route_pool),
                "valid_samples": info["valid_samples"],
                "qubo_terms": info["qubo_terms"],
                "invalid_route_count_samples": info.get("invalid_route_count_samples", ""),
                "invalid_coverage_samples": info.get("invalid_coverage_samples", ""),
                "invalid_both_samples": info.get("invalid_both_samples", ""),
                "mean_route_count_error": info.get("mean_route_count_error", ""),
                "mean_coverage_error": info.get("mean_coverage_error", ""),
                "selected": "|".join(str(idx) for idx in best["selected"]),
            },
        )

    return rows


def run(args):
    all_rows = []
    for case in parse_cases(args.cases):
        print(f"Route-pool QUBO {case[0]} keep={case[1]}")
        all_rows.extend(run_case(case, args))
    if all_rows:
        summary = write_outputs(all_rows, args.output)
        print(f"Saved rows: {args.output}")
        print(f"Saved summary: {summary}")


def main():
    parser = argparse.ArgumentParser(description="Run route-pool set-partitioning QUBO for VRPTW.")
    parser.add_argument("--cases", default="r105:30")
    parser.add_argument("--output", default="results/route_pool_qubo_r105.csv")
    parser.add_argument("--sol-dir", default="data/solomon-100")
    parser.add_argument("--slack-vehicles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--tw-weight", type=float, default=6.0)
    parser.add_argument("--ortools-time-limit", type=int, default=5)
    parser.add_argument("--ortools-scale", type=int, default=1000)
    parser.add_argument("--random-partitions", type=int, default=80)
    parser.add_argument("--max-routes", type=int, default=350)
    parser.add_argument("--coverage-weight", type=float, default=60.0)
    parser.add_argument("--vehicle-weight", type=float, default=30.0)
    parser.add_argument("--route-score-scale", type=float, default=1000.0)
    parser.add_argument("--num-reads", type=int, default=2000)
    parser.add_argument("--num-sweeps", type=int, default=1500)
    parser.add_argument("--refine-window-size", type=int, default=7)
    parser.add_argument("--refine-passes", type=int, default=4)
    parser.add_argument("--refine-constraint-weight", type=float, default=140.0)
    parser.add_argument("--refine-distance-weight", type=float, default=1.0)
    parser.add_argument("--refine-qubo-tw-weight", type=float, default=12.0)
    parser.add_argument("--refine-due-order-weight", type=float, default=0.2)
    parser.add_argument("--refine-reads", type=int, default=400)
    parser.add_argument("--refine-sweeps", type=int, default=700)
    parser.add_argument("--refine-seed-offset", type=int, default=9000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
