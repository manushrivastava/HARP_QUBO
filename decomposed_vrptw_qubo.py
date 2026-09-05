"""Prototype decomposed QUBO pipeline for Solomon VRPTW.

The experiment is intentionally modular:
- assignment QUBO clusters customers into vehicles under capacity
- each vehicle cluster is routed by a smaller TSP/time-window QUBO
- all final routes are evaluated with the same VRPTW simulator

This is meant as a research prototype next to new_qubo_objectives.py.
"""

import argparse
import math
import os

from neal import SimulatedAnnealingSampler

from new_qubo_objectives import (
    build_qubo_nn_angle_slack,
    build_qubo_nearest_neighbor_slack,
    build_qubo_sweep_angle_slack,
    greedy_assignment_sweep,
    load_solomon_txt,
    pairwise_dist,
    reduce_instance,
    run_qubo,
    tsp_nn_route,
)


def add_qubo(Q, a, b, value):
    if abs(value) < 1e-12:
        return
    if a > b:
        a, b = b, a
    Q[(a, b)] = Q.get((a, b), 0.0) + float(value)


def route_metrics(inst, route):
    """Evaluate one route [0, ..., 0] with Solomon time windows."""
    D = pairwise_dist(inst)
    cust = inst["customers"]
    time = 0.0
    distance = 0.0
    violations = 0
    violation_amount = 0.0

    for prev, cid in zip(route[:-1], route[1:]):
        travel = D[(prev, cid)]
        distance += travel
        if cid == 0:
            time += travel
            continue
        arrival = time + travel
        start = max(arrival, cust[cid]["ready"])
        late = max(0.0, start - cust[cid]["due"])
        if late > 1e-9:
            violations += 1
            violation_amount += late
        time = start + cust[cid]["service"]

    return {
        "distance": float(distance),
        "time": float(time),
        "tw_violations": int(violations),
        "tw_violation_amount": float(violation_amount),
    }


def evaluate_routes(inst, routes):
    metrics = [route_metrics(inst, route) for route in routes]
    return {
        "distance": sum(m["distance"] for m in metrics),
        "tw_violations": sum(m["tw_violations"] for m in metrics),
        "tw_violation_amount": sum(m["tw_violation_amount"] for m in metrics),
        "route_metrics": metrics,
    }


def transition_tw_penalty(inst, i, j):
    """A local ordered-pair proxy for time-window compatibility."""
    cust = inst["customers"]
    D = pairwise_dist(inst)
    start_i = max(D[(0, i)], cust[i]["ready"])
    arrival_j = start_i + cust[i]["service"] + D[(i, j)]
    start_j = max(arrival_j, cust[j]["ready"])
    return max(0.0, start_j - cust[j]["due"])


def solve_route_qubo(
    inst,
    cluster,
    A=80.0,
    distance_weight=1.0,
    tw_weight=6.0,
    order_preferences=None,
    order_weight=0.0,
    num_reads=200,
    num_sweeps=2000,
    seed=0,
):
    """Solve a small TSP + local time-window ordering QUBO for one cluster."""
    cluster = list(cluster)
    m = len(cluster)
    if m <= 1:
        return [0] + cluster + [0], {"used_fallback": False, "vars": m * m}

    D = pairwise_dist(inst)
    cust = inst["customers"]
    Q = {}
    y = {(cid, t): f"y_{cid}_{t}" for cid in cluster for t in range(m)}

    # Every customer appears once.
    for cid in cluster:
        for t in range(m):
            add_qubo(Q, y[(cid, t)], y[(cid, t)], -A)
        for t1 in range(m):
            for t2 in range(t1 + 1, m):
                add_qubo(Q, y[(cid, t1)], y[(cid, t2)], 2.0 * A)

    # Every position has one customer.
    for t in range(m):
        for cid in cluster:
            add_qubo(Q, y[(cid, t)], y[(cid, t)], -A)
        for a in range(m):
            for b in range(a + 1, m):
                add_qubo(Q, y[(cluster[a], t)], y[(cluster[b], t)], 2.0 * A)

    # Depot-to-first and last-to-depot terms.
    for cid in cluster:
        start_late = max(0.0, max(D[(0, cid)], cust[cid]["ready"]) - cust[cid]["due"])
        add_qubo(Q, y[(cid, 0)], y[(cid, 0)], distance_weight * D[(0, cid)] + tw_weight * start_late)
        add_qubo(Q, y[(cid, m - 1)], y[(cid, m - 1)], distance_weight * D[(cid, 0)])

    # Ordered transition costs between adjacent route positions.
    for t in range(m - 1):
        for i in cluster:
            for j in cluster:
                if i == j:
                    continue
                tw_late = transition_tw_penalty(inst, i, j)
                cost = distance_weight * D[(i, j)] + tw_weight * tw_late
                add_qubo(Q, y[(i, t)], y[(j, t + 1)], cost)

    # Soft precedence constraints from directional time-window compatibility.
    # If before should precede after, penalize samples where before appears later.
    if order_preferences and order_weight > 0:
        cluster_set = set(cluster)
        seen_preferences = set()
        for before, after in order_preferences:
            if before == after or before not in cluster_set or after not in cluster_set:
                continue
            if (before, after) in seen_preferences:
                continue
            seen_preferences.add((before, after))
            for before_pos in range(1, m):
                for after_pos in range(before_pos):
                    add_qubo(
                        Q,
                        y[(before, before_pos)],
                        y[(after, after_pos)],
                        order_weight,
                    )

    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(Q, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed)

    best_route = None
    best_score = float("inf")
    for row in sampleset.data(["sample", "energy"]):
        sample = row.sample
        route_customers = []
        used = set()
        valid = True
        for t in range(m):
            chosen = [cid for cid in cluster if int(sample.get(y[(cid, t)], 0)) == 1]
            if len(chosen) != 1 or chosen[0] in used:
                valid = False
                break
            route_customers.append(chosen[0])
            used.add(chosen[0])
        if not valid or len(used) != m:
            continue
        route = [0] + route_customers + [0]
        metrics = route_metrics(inst, route)
        score = metrics["distance"] + tw_weight * metrics["tw_violation_amount"]
        if score < best_score:
            best_score = score
            best_route = route

    if best_route is None:
        return tsp_nn_route(inst, cluster), {"used_fallback": True, "vars": m * m}
    return best_route, {"used_fallback": False, "vars": m * m}


def routes_from_nn(inst, sets):
    return [tsp_nn_route(inst, cluster) for cluster in sets]


def solve_decomposed(
    inst,
    assignment_builder,
    seed=100,
    route_seed=500,
    num_reads=200,
    num_sweeps=2000,
    route_reads=200,
    route_sweeps=2000,
    tw_weight=6.0,
):
    sets, loads, info, _, _, _, _, _, _ = run_qubo(
        inst,
        build_qubo_fn=assignment_builder,
        seed=seed,
        A=200.0,
        B=30.0,
        gamma=0.01,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        build_qmat=False,
    )

    routes = []
    route_infos = []
    for k, cluster in enumerate(sets):
        route, route_info = solve_route_qubo(
            inst,
            cluster,
            tw_weight=tw_weight,
            num_reads=route_reads,
            num_sweeps=route_sweeps,
            seed=route_seed + k,
        )
        routes.append(route)
        route_infos.append(route_info)

    return {
        "sets": sets,
        "loads": loads,
        "assignment_info": info,
        "routes": routes,
        "route_infos": route_infos,
        "metrics": evaluate_routes(inst, routes),
    }


def print_result(name, inst, routes, loads=None, route_infos=None):
    metrics = evaluate_routes(inst, routes)
    print(f"\n{name}")
    print("-" * len(name))
    print("distance:", round(metrics["distance"], 3))
    print("time-window violations:", metrics["tw_violations"])
    print("time-window violation amount:", round(metrics["tw_violation_amount"], 3))
    if loads is not None:
        print("loads:", [round(x, 3) for x in loads])
    if route_infos is not None:
        print("route QUBO vars:", sum(info["vars"] for info in route_infos))
        print("route fallbacks:", sum(1 for info in route_infos if info["used_fallback"]))
    for idx, route in enumerate(routes, start=1):
        print(f"V{idx}: {route}")


def main():
    parser = argparse.ArgumentParser(description="Run decomposed assignment + route/time-window QUBO")
    parser.add_argument("--instance", default="c101")
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--sol-dir", default="data/solomon-100")
    parser.add_argument("--assignment-method", choices=["sweep_angle", "nearest_neighbor", "nn_angle"], default="nn_angle")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--route-seed", type=int, default=500)
    parser.add_argument("--num-reads", type=int, default=200)
    parser.add_argument("--num-sweeps", type=int, default=2000)
    parser.add_argument("--route-reads", type=int, default=200)
    parser.add_argument("--route-sweeps", type=int, default=2000)
    parser.add_argument("--tw-weight", type=float, default=6.0)
    args = parser.parse_args()

    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{args.instance}.txt"))
    inst = reduce_instance(sol, keep=args.keep)

    builders = {
        "sweep_angle": build_qubo_sweep_angle_slack,
        "nearest_neighbor": build_qubo_nearest_neighbor_slack,
        "nn_angle": build_qubo_nn_angle_slack,
    }

    greedy_sets, greedy_loads, _, _ = greedy_assignment_sweep(inst)
    greedy_routes = routes_from_nn(inst, greedy_sets)
    print(f"Instance={args.instance}, keep={args.keep}, vehicles={inst['n_vehicles']}")
    print_result("Greedy sweep + NN route", inst, greedy_routes, greedy_loads)

    assignment_result = solve_decomposed(
        inst,
        assignment_builder=builders[args.assignment_method],
        seed=args.seed,
        route_seed=args.route_seed,
        num_reads=args.num_reads,
        num_sweeps=args.num_sweeps,
        route_reads=args.route_reads,
        route_sweeps=args.route_sweeps,
        tw_weight=args.tw_weight,
    )

    assignment_nn_routes = routes_from_nn(inst, assignment_result["sets"])
    print_result(
        f"Assignment QUBO ({args.assignment_method}) + NN route",
        inst,
        assignment_nn_routes,
        assignment_result["loads"],
    )
    print_result(
        f"Decomposed QUBO ({args.assignment_method}) + route/time-window QUBO",
        inst,
        assignment_result["routes"],
        assignment_result["loads"],
        assignment_result["route_infos"],
    )


if __name__ == "__main__":
    main()
