"""Probe whether stronger route candidates can close the R-series gap.

This experiment reads saved standalone ML-QUBO routes, generates additional
OR-Tools-free candidate routes with destroy/repair and insertion heuristics,
then solves the route-pool selection exactly before trying any QUBO selector.

The purpose is diagnostic: if exact selection over the generated route pool
does not improve the current standalone result, another route-pool QUBO will
not improve it either.
"""

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

from compare_with_ortools import solve_case
from decomposed_vrptw_qubo import evaluate_routes, route_metrics
from new_qubo_objectives import load_solomon_txt, reduce_instance
from route_pool_qubo_vrptw import (
    build_route_pool,
    progress_iter,
    route_key,
    route_score,
    routes_score,
    solve_route_pool_exact,
    solve_route_pool_qubo,
)
from route_repair import local_search_route_set, standalone_improve_routes


def parse_csv_list(value, cast=str):
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def parse_routes(value):
    routes = []
    for part in str(value or "").split(";"):
        part = part.strip()
        if part:
            routes.append([int(token) for token in part.split("-") if token.strip()])
    return routes


def routes_text(routes):
    return " ; ".join("-".join(str(cid) for cid in route) for route in routes)


def route_load(inst, route):
    return sum(float(inst["customers"][cid]["demand"]) for cid in route if cid != 0)


def route_set_valid(inst, routes):
    counts = {cid: 0 for cid in inst["kept_ids"]}
    for route in routes:
        if route_load(inst, route) > float(inst["capacity"]) + 1e-9:
            return False
        for cid in route:
            if cid != 0:
                counts[cid] = counts.get(cid, 0) + 1
    return all(counts[cid] == 1 for cid in inst["kept_ids"])


def compact_routes(routes):
    compacted = []
    for route in routes:
        customers = [cid for cid in route if cid != 0]
        if customers:
            compacted.append([0] + customers + [0])
    return compacted


def route_signature(route):
    return tuple(sorted(cid for cid in route if cid != 0))


def add_route(route_map, inst, route, source, tw_weight):
    customers = route_signature(route)
    if not customers:
        return
    if route_load(inst, route) > float(inst["capacity"]) + 1e-9:
        return
    metrics = route_metrics(inst, route)
    score = float(metrics["distance"] + tw_weight * metrics["tw_violation_amount"])
    old = route_map.get(customers)
    if old is None or score + 1e-9 < old["score"]:
        route_map[customers] = {
            "route": list(route),
            "score": score,
            "distance": float(metrics["distance"]),
            "tw_violation_amount": float(metrics["tw_violation_amount"]),
            "source": source,
        }


def add_route_set(route_map, inst, routes, source, tw_weight):
    for route in compact_routes(routes):
        add_route(route_map, inst, route, source, tw_weight)


def customers_with_lateness(inst, routes):
    out = []
    for ridx, route in enumerate(routes):
        time_now = 0.0
        for pos in range(1, len(route) - 1):
            prev = route[pos - 1]
            cid = route[pos]
            metrics = inst["customers"][cid]
            x0, y0 = inst["coords"][prev]
            x1, y1 = inst["coords"][cid]
            travel = math.hypot(x0 - x1, y0 - y1)
            arrival = time_now + travel
            start = max(arrival, float(metrics["ready"]))
            late = max(0.0, start - float(metrics["due"]))
            out.append(
                {
                    "cid": cid,
                    "route_index": ridx,
                    "pos": pos,
                    "lateness": late,
                    "start": start,
                    "ready": float(metrics["ready"]),
                    "due": float(metrics["due"]),
                }
            )
            time_now = start + float(metrics["service"])
    return out


def remove_customers(routes, removed):
    removed = set(removed)
    new_routes = []
    for route in routes:
        kept = [cid for cid in route if cid == 0 or cid not in removed]
        customers = [cid for cid in kept if cid != 0]
        if customers:
            new_routes.append([0] + customers + [0])
        else:
            new_routes.append([0, 0])
    return new_routes


def choose_removed(inst, routes, rng, remove_count, strategy):
    customers = [cid for route in routes for cid in route if cid != 0]
    if remove_count >= len(customers):
        return customers
    if strategy == "random":
        return rng.sample(customers, remove_count)

    late_rows = customers_with_lateness(inst, routes)
    if strategy == "late":
        ranked = sorted(late_rows, key=lambda row: row["lateness"], reverse=True)
        head = [row["cid"] for row in ranked[: max(remove_count, min(len(ranked), remove_count * 2))]]
        if len(head) >= remove_count:
            return rng.sample(head, remove_count)
        return head + rng.sample([cid for cid in customers if cid not in head], remove_count - len(head))

    if strategy == "late_block":
        ranked = sorted(late_rows, key=lambda row: row["lateness"], reverse=True)
        if ranked:
            anchor = rng.choice(ranked[: max(1, min(6, len(ranked)))])
            route = routes[anchor["route_index"]]
            removed = []
            for radius in range(0, max(3, len(route))):
                for pos in (anchor["pos"] - radius, anchor["pos"] + radius):
                    if 0 < pos < len(route) - 1:
                        cid = route[pos]
                        if cid != 0 and cid not in removed:
                            removed.append(cid)
                    if len(removed) >= remove_count:
                        return removed

            anchor_due = anchor["due"]
            due_neighbors = sorted(
                [row for row in late_rows if row["cid"] not in removed],
                key=lambda row: (abs(row["due"] - anchor_due), -row["lateness"]),
            )
            for row in due_neighbors:
                removed.append(row["cid"])
                if len(removed) >= remove_count:
                    return removed

    # Mixed strategy: pick one late customer and nearby route-neighborhood noise.
    ranked = sorted(late_rows, key=lambda row: row["lateness"], reverse=True)
    removed = []
    if ranked:
        anchor = rng.choice(ranked[: max(1, min(8, len(ranked)))])
        removed.append(anchor["cid"])
        route = routes[anchor["route_index"]]
        for pos in range(max(1, anchor["pos"] - 2), min(len(route) - 1, anchor["pos"] + 3)):
            cid = route[pos]
            if cid != 0 and cid not in removed:
                removed.append(cid)
            if len(removed) >= remove_count:
                return removed
    remaining = [cid for cid in customers if cid not in removed]
    if remaining:
        removed.extend(rng.sample(remaining, min(remove_count - len(removed), len(remaining))))
    return removed


def route_due_order_penalty(inst, route):
    customers = [cid for cid in route if cid != 0]
    if len(customers) < 2:
        return 0.0
    horizon = max(float(inst["customers"][cid]["due"]) for cid in inst["kept_ids"]) or 1.0
    penalty = 0.0
    prev_due = float(inst["customers"][customers[0]]["due"])
    for cid in customers[1:]:
        due = float(inst["customers"][cid]["due"])
        penalty += max(0.0, prev_due - due) / horizon
        prev_due = due
    return penalty


def route_repair_score(inst, route, tw_weight, due_order_weight):
    if len(route) <= 2:
        return {"score": 0.0, "distance": 0.0, "lateness": 0.0}
    metrics = route_metrics(inst, route)
    score = (
        float(metrics["distance"])
        + tw_weight * float(metrics["tw_violation_amount"])
        + due_order_weight * route_due_order_penalty(inst, route)
    )
    return {
        "score": score,
        "distance": float(metrics["distance"]),
        "lateness": float(metrics["tw_violation_amount"]),
    }


def insertion_options(inst, routes, loads, cid, tw_weight, rng, noise, due_order_weight):
    demand = float(inst["customers"][cid]["demand"])
    options = []
    for ridx, route in enumerate(routes):
        if loads[ridx] + demand > float(inst["capacity"]) + 1e-9:
            continue
        old = route_repair_score(inst, route, tw_weight, due_order_weight)
        for pos in range(1, len(route)):
            candidate = route[:pos] + [cid] + route[pos:]
            new = route_repair_score(inst, candidate, tw_weight, due_order_weight)
            delta = new["score"] - old["score"]
            if noise:
                delta += rng.uniform(-noise, noise)
            options.append(
                {
                    "delta": delta,
                    "lateness_delta": new["lateness"] - old["lateness"],
                    "distance_delta": new["distance"] - old["distance"],
                    "route_index": ridx,
                    "route": candidate,
                }
            )
    options.sort(key=lambda item: item["delta"])
    return options


def regret_repair(inst, partial_routes, missing, tw_weight, rng, mode, noise, due_order_weight):
    routes = [list(route) for route in partial_routes]
    loads = [route_load(inst, route) for route in routes]
    remaining = list(missing)
    while remaining:
        choices = []
        for cid in remaining:
            options = insertion_options(inst, routes, loads, cid, tw_weight, rng, noise, due_order_weight)
            if not options:
                continue
            best = options[0]
            regret = (options[1]["delta"] - best["delta"]) if len(options) > 1 else 1e6
            lateness_best = min(options, key=lambda item: (item["lateness_delta"], item["delta"], item["distance_delta"]))
            choices.append({"cid": cid, "best": best, "lateness_best": lateness_best, "regret": regret})
        if not choices:
            return None
        if mode == "best":
            choice = min(choices, key=lambda item: item["best"]["delta"])
        elif mode == "random_best":
            top = sorted(choices, key=lambda item: item["best"]["delta"])[: max(1, min(5, len(choices)))]
            choice = rng.choice(top)
        elif mode == "lateness_best":
            choice = min(choices, key=lambda item: (item["lateness_best"]["lateness_delta"], item["lateness_best"]["delta"]))
            choice["best"] = choice["lateness_best"]
        elif mode == "lateness_regret":
            choice = max(
                choices,
                key=lambda item: (
                    -item["lateness_best"]["lateness_delta"],
                    item["regret"],
                    -item["lateness_best"]["delta"],
                ),
            )
            choice["best"] = choice["lateness_best"]
        else:
            choice = max(choices, key=lambda item: (item["regret"], -item["best"]["delta"]))

        ridx = choice["best"]["route_index"]
        routes[ridx] = choice["best"]["route"]
        loads[ridx] += float(inst["customers"][choice["cid"]]["demand"])
        remaining.remove(choice["cid"])

    return compact_routes(routes)


def lns_route_sets(inst, starts, args):
    rng = random.Random(args.seed)
    best_sets = []
    strategies = parse_csv_list(args.destroy_strategies)
    modes = parse_csv_list(args.repair_modes)
    repair_weights = parse_csv_list(args.repair_weights, float)
    remove_counts = parse_csv_list(args.remove_counts, int)

    total_iterations = len(starts) * args.lns_iterations
    iterator = (
        (start_idx, start_routes)
        for start_idx, start_routes in enumerate(starts)
        for _ in range(args.lns_iterations)
    )
    for start_idx, start_routes in progress_iter(
        iterator,
        enabled=args.progress,
        desc="LNS destroy/repair",
        total=total_iterations,
        unit="iter",
    ):
        remove_count = rng.choice(remove_counts)
        strategy = rng.choice(strategies)
        repair_weight = rng.choice(repair_weights)
        mode = rng.choice(modes)
        removed = choose_removed(inst, start_routes, rng, remove_count, strategy)
        partial = remove_customers(start_routes, removed)
        repaired = regret_repair(
            inst,
            partial,
            removed,
            tw_weight=repair_weight,
            rng=rng,
            mode=mode,
            noise=args.insertion_noise,
            due_order_weight=args.due_order_weight,
        )
        if repaired is None:
            continue
        repaired = local_search_route_set(inst, repaired, repair_weight, max_passes=args.local_search_passes)
        if route_set_valid(inst, repaired):
            score = routes_score(inst, repaired, args.tw_weight)
            best_sets.append(
                {
                    "routes": repaired,
                    "score": score,
                    "source": f"lns_start{start_idx}_{strategy}_{mode}_rm{remove_count}_tw{repair_weight:g}",
                }
            )

    best_sets.sort(key=lambda row: row["score"])
    return best_sets[: args.keep_lns_sets]


def load_source_rows(args):
    rows = []
    if not Path(args.source_csv).exists():
        return rows
    with open(args.source_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("instance", "").lower() != args.instance.lower():
                continue
            if int(row.get("keep", 0)) != int(args.keep):
                continue
            if int(row.get("seed", 0)) not in set(parse_csv_list(args.seeds, int)):
                continue
            if row.get("method") not in set(parse_csv_list(args.source_methods)):
                continue
            rows.append(row)
    return rows


def get_benchmark_ortools(inst, args):
    if Path(args.ortools_csv).exists():
        with open(args.ortools_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("mode") == "ortools_same_k_soft_tw":
                    routes = parse_routes(row["routes"])
                    score = float(row["distance"]) + args.tw_weight * float(row["tw_violation_amount"])
                    return {"routes": routes, "score": score, "source": "best_known_ortools_csv"}
    started = time.perf_counter()
    if args.progress:
        print(
            f"Running OR-Tools benchmark: instance={args.instance} keep={args.keep} "
            f"time_limit={args.ortools_time_limit}s",
            flush=True,
        )
    result = solve_case(
        inst,
        vehicles=inst["n_vehicles"],
        hard_time_windows=False,
        time_limit_sec=args.ortools_time_limit,
        soft_lateness_penalty=args.tw_weight,
        scale=args.ortools_scale,
    )
    if not result["solved"]:
        return None
    return {"routes": result["routes"], "score": routes_score(inst, result["routes"], args.tw_weight), "source": f"ortools_{time.perf_counter() - started:.1f}s"}


def result_row(inst, method, routes, elapsed, args, extra=None):
    extra = extra or {}
    metrics = evaluate_routes(inst, routes)
    return {
        "instance": args.instance,
        "keep": args.keep,
        "method": method,
        "distance": round(float(metrics["distance"]), 6),
        "tw_violations": metrics["tw_violations"],
        "tw_violation_amount": round(float(metrics["tw_violation_amount"]), 6),
        "score": round(float(metrics["distance"] + args.tw_weight * metrics["tw_violation_amount"]), 6),
        "elapsed_sec": round(float(elapsed), 6),
        "route_pool_size": extra.get("route_pool_size", ""),
        "selected": extra.get("selected", ""),
        "selected_sources": extra.get("selected_sources", ""),
        "valid_samples": extra.get("valid_samples", ""),
        "qubo_terms": extra.get("qubo_terms", ""),
        "invalid_route_count_samples": extra.get("invalid_route_count_samples", ""),
        "invalid_coverage_samples": extra.get("invalid_coverage_samples", ""),
        "invalid_both_samples": extra.get("invalid_both_samples", ""),
        "mean_route_count_error": extra.get("mean_route_count_error", ""),
        "mean_coverage_error": extra.get("mean_coverage_error", ""),
        "source": extra.get("source", ""),
        "route_sizes": "|".join(str(max(0, len(route) - 2)) for route in routes),
        "routes": routes_text(routes),
    }


def write_outputs(rows, args, pool_rows, lns_sets):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pool_path = output.with_name(output.stem + "_route_pool.csv")
    with pool_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["route_index", "source", "score", "distance", "tw_violation_amount", "route_size", "route"],
        )
        writer.writeheader()
        for idx, row in enumerate(pool_rows):
            writer.writerow(
                {
                    "route_index": idx,
                    "source": row.get("source", ""),
                    "score": round(float(row["score"]), 6),
                    "distance": round(float(row["distance"]), 6),
                    "tw_violation_amount": round(float(row["tw_violation_amount"]), 6),
                    "route_size": max(0, len(row["route"]) - 2),
                    "route": "-".join(str(cid) for cid in row["route"]),
                }
            )

    lns_path = output.with_name(output.stem + "_lns_sets.csv")
    with lns_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "source", "score", "routes"])
        writer.writeheader()
        for rank, row in enumerate(lns_sets[:50], start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "source": row["source"],
                    "score": round(float(row["score"]), 6),
                    "routes": routes_text(row["routes"]),
                }
            )

    ortools = next((row for row in rows if row["method"] == "best_known_ortools"), None)
    summary_path = output.with_name(output.stem + "_summary.md")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("# Strong Route-Pool Repair Probe\n\n")
        f.write("OR-Tools is used only as the benchmark row. No OR-Tools routes are added to the candidate pool.\n\n")
        f.write(f"- Instance: `{args.instance}`\n")
        f.write(f"- Keep: `{args.keep}`\n")
        f.write(f"- Seeds: `{args.seeds}`\n")
        f.write(f"- Route pool size: `{len(pool_rows)}`\n")
        f.write(f"- LNS sets kept: `{len(lns_sets)}`\n\n")
        f.write("| method | score | distance | lateness | gap_to_ortools |\n")
        f.write("| --- | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            gap = ""
            if ortools is not None:
                gap = f"{float(row['score']) - float(ortools['score']):.6f}"
            f.write(
                f"| {row['method']} | {float(row['score']):.6f} | {float(row['distance']):.6f} | "
                f"{float(row['tw_violation_amount']):.6f} | {gap} |\n"
            )
        f.write("\n## Interpretation Guardrail\n\n")
        f.write(
            "If `strong_route_pool_exact_best` does not improve on `best_current_standalone`, "
            "the generated route pool is still the bottleneck and a QUBO selector should not be expected to help.\n"
        )
    return output, pool_path, lns_path, summary_path


def run(args):
    if args.progress:
        print(f"Starting strong route-pool repair: instance={args.instance} keep={args.keep}", flush=True)
    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{args.instance.lower()}.txt"))
    inst = reduce_instance(sol, keep=args.keep, slack_vehicles=args.slack_vehicles)
    source_rows = load_source_rows(args)
    if not source_rows and not args.allow_empty_source:
        raise SystemExit(f"No source rows found in {args.source_csv}")

    route_map = {}
    start_sets = []
    rows = []
    start = time.perf_counter()

    for row in progress_iter(source_rows, enabled=args.progress, desc="source route sets", unit="row"):
        routes = parse_routes(row["routes"])
        if not route_set_valid(inst, routes):
            continue
        seed_label = row.get("seed", "")
        source_label = f"source_{row['method']}_seed{seed_label}" if seed_label != "" else f"source_{row['method']}"
        add_route_set(route_map, inst, routes, source_label, args.tw_weight)
        start_sets.append(routes)
        rows.append(
            result_row(
                inst,
                source_label,
                routes,
                0.0,
                args,
                {"source": row["method"]},
            )
        )

    if rows:
        current_best_candidates = [row for row in rows if "self_supervised_ml_qubo_local_refined" in row["method"]]
        if not current_best_candidates:
            current_best_candidates = rows
        current_best = min(current_best_candidates, key=lambda row: float(row["score"]))
        rows.append(
            {
                **current_best,
                "method": "best_current_standalone",
                "elapsed_sec": 0.0,
                "source": "best source self_supervised_ml_qubo_local_refined",
            }
        )

    if args.progress:
        print(
            f"Building heuristic route pool: random_partitions={args.random_partitions} "
            f"heuristic_pool_routes={args.heuristic_pool_routes}",
            flush=True,
        )
    heuristic_pool, heuristic_partitions = build_route_pool(
        inst,
        tw_weight=args.tw_weight,
        random_partitions=args.random_partitions,
        max_routes=args.heuristic_pool_routes,
        seed=args.seed,
        progress=args.progress,
    )
    for row in progress_iter(heuristic_pool, enabled=args.progress, desc="add heuristic routes", unit="route"):
        add_route(route_map, inst, row["route"], f"heuristic_{row.get('source', '')}", args.tw_weight)
    for partition in progress_iter(
        heuristic_partitions[: args.start_partitions],
        enabled=args.progress,
        desc="seed start partitions",
        unit="set",
    ):
        routes = partition["routes"]
        if route_set_valid(inst, routes):
            start_sets.append(routes)
            add_route_set(route_map, inst, routes, f"heuristic_partition_{partition['source']}", args.tw_weight)

    valid_heuristic_partitions = [
        partition for partition in heuristic_partitions if route_set_valid(inst, partition["routes"])
    ]
    if valid_heuristic_partitions:
        best_heuristic = min(valid_heuristic_partitions, key=lambda row: float(row["score"]))
        rows.append(
            result_row(
                inst,
                "best_heuristic_partition",
                best_heuristic["routes"],
                0.0,
                args,
                {"source": best_heuristic["source"]},
            )
        )

    if not start_sets:
        raise SystemExit("No valid source or heuristic start route sets were generated.")

    # --- DEDUPLICATION + HYBRID SELECTION (manu_sir_additional_experimentation) ---
    import random as _random

    def _sol_edges(sol):
        """Union of all consecutive (u,v) edge pairs across all routes in a solution."""
        edges = set()
        for route in sol:
            edges.update(zip(route[:-1], route[1:]))
        return edges

    def _jaccard(e1, e2):
        inter = len(e1 & e2)
        union = len(e1 | e2)
        return inter / union if union > 0 else 0.0

    # ── Step 0: Baseline overlap on the raw pool ──────────────────────────────
    _raw_n = len(start_sets)
    _raw_sims = []
    if _raw_n > 1:
        _raw_edge_cache = [_sol_edges(s) for s in start_sets]
        for _i in range(_raw_n):
            for _j in range(_i + 1, _raw_n):
                _raw_sims.append(_jaccard(_raw_edge_cache[_i], _raw_edge_cache[_j]))
        _raw_avg = sum(_raw_sims) / len(_raw_sims)
        _raw_max = max(_raw_sims)
        _raw_min = min(_raw_sims)
    else:
        _raw_avg = _raw_max = _raw_min = 0.0
        _raw_sims = []

    # ── Step 1: Deduplication — discard 100% Jaccard duplicates ──────────────
    _deduped = []           # list[list[list[int]]]  — the solutions
    _deduped_edges = []     # cached edge sets
    _deduped_scores = []    # routes_score for each

    for _sol in start_sets:
        _e = _sol_edges(_sol)
        if any(_e == _ex for _ex in _deduped_edges):
            continue        # exact duplicate → drop
        _deduped.append(_sol)
        _deduped_edges.append(_e)
        _deduped_scores.append(routes_score(inst, _sol, args.tw_weight))

    _n_removed = _raw_n - len(_deduped)

    # ── Step 2: Post-dedup overlap report ────────────────────────────────────
    _ded_n = len(_deduped)
    _ded_sims = []
    _ded_pair_rows = []
    if _ded_n > 1:
        for _i in range(_ded_n):
            for _j in range(_i + 1, _ded_n):
                _sim = _jaccard(_deduped_edges[_i], _deduped_edges[_j])
                _ded_sims.append(_sim)
                _ded_pair_rows.append({"sol_i": _i, "sol_j": _j, "jaccard_similarity": round(_sim, 6)})
        _ded_avg = sum(_ded_sims) / len(_ded_sims)
        _ded_max = max(_ded_sims)
        _ded_min = min(_ded_sims)
    else:
        _ded_avg = _ded_max = _ded_min = 0.0
        _ded_pair_rows = []

    # --- ADAPTIVE SLIDING SCALE HYBRID SELECTION ---
    _target_n = min(args.start_partitions, _ded_n)
    if _raw_avg > 0.60:
        _elite_ratio = 0.30
    elif _raw_avg < 0.30:
        _elite_ratio = 0.70
    else:
        _elite_ratio = 0.50

    _elite_n = int(_target_n * _elite_ratio)
    _explore_n = _target_n - _elite_n

    print(f"=== ADAPTIVE HYBRID SELECTION ===")
    print(f"Pre-Dedup Avg Overlap: {_raw_avg:.2%}")
    print(f"Selected Elite Ratio: {_elite_ratio * 100:.0f}% (Elite: {_elite_n}, Explore: {_explore_n})")
    print("==================================")
    # -----------------------------------------------

    _sorted_idx = sorted(range(_ded_n), key=lambda _k: _deduped_scores[_k])
    _elite_pool  = [_deduped[_k] for _k in _sorted_idx[:_elite_n]]
    _remain_pool = [_deduped[_k] for _k in _sorted_idx[_elite_n:]]
    _rng_hybrid = _random.Random(args.seed)
    _explore_pool = _rng_hybrid.sample(_remain_pool, min(_explore_n, len(_remain_pool)))

    start_sets = _elite_pool + _explore_pool   # ← rebuilt start_sets fed into LNS

    # ── Step 4: Save full CSV report ─────────────────────────────────────────
    _report_dir = Path("manu_sir_additional_experimentation")
    _report_dir.mkdir(parents=True, exist_ok=True)
    _report_fname = (
        f"hybrid_selection_overlap_report"
        f"_{args.instance.lower()}"
        f"_keep{args.keep}"
        f"_seeds{args.seeds.replace(',', '-')}"
        f"_raw{_raw_n}_dedup{_ded_n}_final{len(start_sets)}.csv"
    )
    _report_path = _report_dir / _report_fname

    with _report_path.open("w", newline="", encoding="utf-8") as _f:
        _writer = csv.DictWriter(_f, fieldnames=[
            "section", "experiment", "instance", "keep", "seeds",
            "raw_solutions", "dedup_solutions", "final_solutions",
            "avg_jaccard_before", "avg_jaccard_after",
            "max_jaccard_before", "max_jaccard_after",
            "min_jaccard_before", "min_jaccard_after",
            "elite_count", "explore_count",
            "sol_i", "sol_j", "jaccard_similarity",
        ])
        _writer.writeheader()
        # Summary row
        _writer.writerow({
            "section": "summary",
            "experiment": "hybrid_selection_overlap_analysis",
            "instance": args.instance, "keep": args.keep, "seeds": args.seeds,
            "raw_solutions": _raw_n,
            "dedup_solutions": _ded_n,
            "final_solutions": len(start_sets),
            "avg_jaccard_before": round(_raw_avg, 6),
            "avg_jaccard_after":  round(_ded_avg, 6),
            "max_jaccard_before": round(_raw_max, 6),
            "max_jaccard_after":  round(_ded_max, 6),
            "min_jaccard_before": round(_raw_min, 6),
            "min_jaccard_after":  round(_ded_min, 6),
            "elite_count": len(_elite_pool),
            "explore_count": len(_explore_pool),
            "sol_i": "", "sol_j": "", "jaccard_similarity": "",
        })
        # Post-dedup pairwise rows
        for _pr in _ded_pair_rows:
            _writer.writerow({
                "section": "pairwise_dedup",
                "experiment": "hybrid_selection_overlap_analysis",
                "instance": args.instance, "keep": args.keep, "seeds": args.seeds,
                "raw_solutions": _raw_n,
                "dedup_solutions": _ded_n,
                "final_solutions": len(start_sets),
                "avg_jaccard_before": "", "avg_jaccard_after": "",
                "max_jaccard_before": "", "max_jaccard_after": "",
                "min_jaccard_before": "", "min_jaccard_after": "",
                "elite_count": "", "explore_count": "",
                **_pr,
            })

    # ── END HYBRID SELECTION ──────────────────────────────────────────────────

    if args.progress:
        print(f"Running LNS route repair from {len(start_sets)} start sets", flush=True)
    lns_sets = lns_route_sets(inst, start_sets, args)
    for row in progress_iter(lns_sets, enabled=args.progress, desc="add LNS route sets", unit="set"):
        add_route_set(route_map, inst, row["routes"], row["source"], args.tw_weight)

    # Try the expensive guarded repair only for the top few route sets.
    guarded_args = SimpleNamespace(
        tw_weight=args.tw_weight,
        repair_tw_weight=args.guarded_repair_tw_weight,
        local_search_passes=args.guarded_local_passes,
        inter_route_passes=args.guarded_inter_route_passes,
    )
    guarded_rows = list(enumerate(lns_sets[: args.guarded_repair_sets]))
    for idx, row in progress_iter(guarded_rows, enabled=args.progress, desc="guarded repair", unit="set"):
        repaired = standalone_improve_routes(inst, row["routes"], guarded_args)
        if route_set_valid(inst, repaired):
            score = routes_score(inst, repaired, args.tw_weight)
            lns_sets.append({"routes": repaired, "score": score, "source": f"guarded_repair_{idx}_{row['source']}"})
            add_route_set(route_map, inst, repaired, f"guarded_repair_{idx}_{row['source']}", args.tw_weight)

    route_pool = sorted(route_map.values(), key=lambda row: row["score"])[: args.max_routes]
    if args.progress:
        print(f"Final route pool size: {len(route_pool)}", flush=True)
    best_lns = min(lns_sets, key=lambda row: row["score"]) if lns_sets else None
    if best_lns is not None:
        rows.append(result_row(inst, "best_lns_route_set", best_lns["routes"], time.perf_counter() - start, args, {"source": best_lns["source"]}))

    if not args.skip_exact:
        if args.progress:
            print("Solving exact route-pool cover", flush=True)
        exact_started = time.perf_counter()
        exact = solve_route_pool_exact(inst, route_pool, args.tw_weight)
        if exact is not None:
            selected_sources = "|".join(route_pool[idx].get("source", "") for idx in exact["selected"])
            rows.append(
                result_row(
                    inst,
                    "strong_route_pool_exact_best",
                    exact["routes"],
                    time.perf_counter() - exact_started,
                    args,
                    {
                        "route_pool_size": len(route_pool),
                        "selected": "|".join(str(idx) for idx in exact["selected"]),
                        "selected_sources": selected_sources,
                    },
                )
            )

    if args.run_qubo:
        if args.progress:
            print("Solving QUBO route-pool cover", flush=True)
        qubo_started = time.perf_counter()
        best, info = solve_route_pool_qubo(inst, route_pool, args)
        if best is not None:
            selected_sources = "|".join(route_pool[idx].get("source", "") for idx in best["selected"])
            rows.append(
                result_row(
                    inst,
                    "strong_route_pool_set_partition_qubo",
                    best["routes"],
                    time.perf_counter() - qubo_started,
                    args,
                    {
                        "route_pool_size": len(route_pool),
                        "selected": "|".join(str(idx) for idx in best["selected"]),
                        "selected_sources": selected_sources,
                        "valid_samples": info["valid_samples"],
                        "qubo_terms": info["qubo_terms"],
                        "invalid_route_count_samples": info.get("invalid_route_count_samples", ""),
                        "invalid_coverage_samples": info.get("invalid_coverage_samples", ""),
                        "invalid_both_samples": info.get("invalid_both_samples", ""),
                        "mean_route_count_error": info.get("mean_route_count_error", ""),
                        "mean_coverage_error": info.get("mean_coverage_error", ""),
                    },
                )
            )

    # benchmark = get_benchmark_ortools(inst, args)
    # if benchmark is not None:
    #     rows.append(result_row(inst, "best_known_ortools", benchmark["routes"], 0.0, args, {"source": benchmark["source"]}))

    rows.sort(key=lambda row: float(row["score"]))
    if args.progress:
        print("Writing output CSVs and summary", flush=True)
    return write_outputs(rows, args, route_pool, sorted(lns_sets, key=lambda row: row["score"]))


def main():
    parser = argparse.ArgumentParser(description="Probe strong route-pool repair on saved standalone routes.")
    parser.add_argument("--sol-dir", default="data/solomon-100")
    parser.add_argument("--instance", default="r102")
    parser.add_argument("--keep", type=int, default=30)
    parser.add_argument("--seeds", default="100,200,300")
    parser.add_argument("--source-csv", default="results/self_supervised_ml_qubo_rrepair_tw10_r102_3seed_vs_ortools.csv")
    parser.add_argument("--source-methods", default="greedy_sweep_nn,normal_qubo_nn_angle,self_supervised_ml_qubo,self_supervised_ml_qubo_refined,self_supervised_ml_qubo_local_refined")
    parser.add_argument("--ortools-csv", default="results/ortools_r102_keep30_time30_check.csv")
    parser.add_argument("--output", default="results/strong_route_pool_repair_r102_keep30.csv")
    parser.add_argument("--tw-weight", type=float, default=6.0)
    parser.add_argument("--repair-weights", default="6,8,10,12,15")
    parser.add_argument("--remove-counts", default="3,4,5,6,8")
    parser.add_argument("--destroy-strategies", default="random,late,mixed")
    parser.add_argument("--repair-modes", default="best,regret,random_best")
    parser.add_argument("--due-order-weight", type=float, default=0.0)
    parser.add_argument("--random-partitions", type=int, default=160)
    parser.add_argument("--heuristic-pool-routes", type=int, default=700)
    parser.add_argument("--start-partitions", type=int, default=60)
    parser.add_argument("--lns-iterations", type=int, default=140)
    parser.add_argument("--keep-lns-sets", type=int, default=120)
    parser.add_argument("--max-routes", type=int, default=900)
    parser.add_argument("--local-search-passes", type=int, default=1)
    parser.add_argument("--insertion-noise", type=float, default=0.0)
    parser.add_argument("--guarded-repair-sets", type=int, default=8)
    parser.add_argument("--guarded-repair-tw-weight", type=float, default=10.0)
    parser.add_argument("--guarded-local-passes", type=int, default=2)
    parser.add_argument("--guarded-inter-route-passes", type=int, default=1)
    parser.add_argument("--skip-exact", action="store_true")
    parser.add_argument("--allow-empty-source", action="store_true")
    parser.add_argument("--run-qubo", action="store_true")
    parser.add_argument("--coverage-weight", type=float, default=120.0)
    parser.add_argument("--vehicle-weight", type=float, default=120.0)
    parser.add_argument("--route-score-scale", type=float, default=1000.0)
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--slack-vehicles", type=int, default=0)
    parser.add_argument("--ortools-time-limit", type=int, default=30)
    parser.add_argument("--ortools-scale", type=int, default=1000)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars and stage messages.")
    args = parser.parse_args()
    args.progress = not args.no_progress

    output, pool_path, lns_path, summary_path = run(args)
    print(f"Wrote {output}")
    print(f"Wrote {pool_path}")
    print(f"Wrote {lns_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
