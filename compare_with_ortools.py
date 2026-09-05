"""Compare existing QUBO VRPTW experiments against Google OR-Tools baselines.

The script uses the same Solomon parser and route evaluator as the QUBO
experiments, then solves comparable reduced instances with OR-Tools Routing.

It writes rows for:
- ortools_same_k_hard_tw: hard time windows with the same capacity-derived K
- ortools_same_k_soft_tw: same K, soft lateness penalties
- ortools_min_k_hard_tw: first vehicle count from K..max_vehicles with a hard-TW solution
"""

import argparse
import csv
import os
import time

from decomposed_vrptw_qubo import evaluate_routes
from new_qubo_objectives import load_solomon_txt, pairwise_dist, reduce_instance


def parse_csv_list(value, cast=str):
    return [cast(part.strip()) for part in str(value).split(",") if part.strip()]


def require_ortools():
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as exc:
        raise SystemExit(
            "OR-Tools is not installed. Install it with: python -m pip install ortools"
        ) from exc
    return pywrapcp, routing_enums_pb2


def scaled(value, scale):
    return int(round(float(value) * scale))


def routing_status_name(status):
    pywrapcp, _ = require_ortools()
    for name in [
        "ROUTING_NOT_SOLVED",
        "ROUTING_SUCCESS",
        "ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED",
        "ROUTING_FAIL",
        "ROUTING_FAIL_TIMEOUT",
        "ROUTING_INVALID",
        "ROUTING_INFEASIBLE",
        "ROUTING_OPTIMAL",
    ]:
        if getattr(pywrapcp.RoutingModel, name, None) == status:
            return name
    return str(status)


def make_solver(inst, vehicles, hard_time_windows, time_limit_sec, soft_lateness_penalty, scale):
    pywrapcp, routing_enums_pb2 = require_ortools()
    ids = [0] + list(inst["kept_ids"])
    node_to_cid = {idx: cid for idx, cid in enumerate(ids)}
    cid_to_node = {cid: idx for idx, cid in node_to_cid.items()}
    D = pairwise_dist(inst)
    customers = inst["customers"]

    manager = pywrapcp.RoutingIndexManager(len(ids), vehicles, cid_to_node[0])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_cid = node_to_cid[manager.IndexToNode(from_index)]
        to_cid = node_to_cid[manager.IndexToNode(to_index)]
        return scaled(D[(from_cid, to_cid)], scale)

    distance_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(distance_callback_index)

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        lambda from_index: int(round(customers[node_to_cid[manager.IndexToNode(from_index)]]["demand"]))
    )
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [int(round(inst["capacity"]))] * vehicles,
        True,
        "Capacity",
    )

    def time_callback(from_index, to_index):
        from_cid = node_to_cid[manager.IndexToNode(from_index)]
        to_cid = node_to_cid[manager.IndexToNode(to_index)]
        service = 0.0 if from_cid == 0 else customers[from_cid]["service"]
        return scaled(service + D[(from_cid, to_cid)], scale)

    time_callback_index = routing.RegisterTransitCallback(time_callback)
    horizon = max(customers[cid]["due"] + customers[cid]["service"] for cid in ids) + 10000
    routing.AddDimension(
        time_callback_index,
        scaled(horizon, scale),
        scaled(horizon, scale),
        True,
        "Time",
    )
    time_dimension = routing.GetDimensionOrDie("Time")

    for cid in inst["kept_ids"]:
        index = manager.NodeToIndex(cid_to_node[cid])
        ready = scaled(customers[cid]["ready"], scale)
        due = scaled(customers[cid]["due"], scale)
        if hard_time_windows:
            time_dimension.CumulVar(index).SetRange(ready, due)
        else:
            time_dimension.CumulVar(index).SetRange(ready, scaled(horizon, scale))
            time_dimension.SetCumulVarSoftUpperBound(
                index,
                due,
                int(round(soft_lateness_penalty)),
            )

    for vehicle_id in range(vehicles):
        time_dimension.CumulVar(routing.Start(vehicle_id)).SetRange(0, 0)
        time_dimension.CumulVar(routing.End(vehicle_id)).SetRange(0, scaled(horizon, scale))

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(int(time_limit_sec))
    search_parameters.log_search = False

    return routing, manager, node_to_cid, search_parameters


def extract_routes(routing, manager, node_to_cid, solution, vehicles):
    routes = []
    for vehicle_id in range(vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(node_to_cid[manager.IndexToNode(index)])
            index = solution.Value(routing.NextVar(index))
        route.append(node_to_cid[manager.IndexToNode(index)])
        if len(route) > 2:
            routes.append(route)
    return routes


def route_loads(inst, routes):
    customers = inst["customers"]
    return [
        sum(float(customers[cid]["demand"]) for cid in route if cid != 0)
        for route in routes
    ]


def solve_case(inst, vehicles, hard_time_windows, time_limit_sec, soft_lateness_penalty, scale):
    started = time.perf_counter()
    routing, manager, node_to_cid, search_parameters = make_solver(
        inst,
        vehicles=vehicles,
        hard_time_windows=hard_time_windows,
        time_limit_sec=time_limit_sec,
        soft_lateness_penalty=soft_lateness_penalty,
        scale=scale,
    )
    solution = routing.SolveWithParameters(search_parameters)
    elapsed = time.perf_counter() - started
    if solution is None:
        return {
            "solved": False,
            "status": routing_status_name(routing.status()),
            "elapsed_sec": elapsed,
            "routes": [],
        }

    routes = extract_routes(routing, manager, node_to_cid, solution, vehicles)
    metrics = evaluate_routes(inst, routes)
    loads = route_loads(inst, routes)
    return {
        "solved": True,
        "status": routing_status_name(routing.status()),
        "elapsed_sec": elapsed,
        "routes": routes,
        "distance": metrics["distance"],
        "tw_violations": metrics["tw_violations"],
        "tw_violation_amount": metrics["tw_violation_amount"],
        "loads": loads,
    }


def row_for(instance, keep, mode, inst, vehicles_limit, result):
    routes = result.get("routes", [])
    loads = result.get("loads", [])
    return {
        "instance": instance,
        "keep": keep,
        "capacity_min_vehicles": inst["n_vehicles"],
        "mode": mode,
        "vehicles_limit": vehicles_limit,
        "vehicles_used": len(routes),
        "solved": result["solved"],
        "status": result["status"],
        "distance": round(result.get("distance", 0.0), 6) if result["solved"] else "",
        "tw_violations": result.get("tw_violations", "") if result["solved"] else "",
        "tw_violation_amount": (
            round(result.get("tw_violation_amount", 0.0), 6) if result["solved"] else ""
        ),
        "loads": "|".join(str(round(load, 6)) for load in loads),
        "route_sizes": "|".join(str(max(0, len(route) - 2)) for route in routes),
        "elapsed_sec": round(result["elapsed_sec"], 6),
        "routes": " ; ".join("-".join(str(cid) for cid in route) for route in routes),
    }


def solve_min_k_hard(instance, keep, sol, base_inst, args):
    for vehicles in range(base_inst["n_vehicles"], int(sol["max_vehicles"]) + 1):
        inst = dict(base_inst)
        inst["n_vehicles"] = vehicles
        result = solve_case(
            inst,
            vehicles=vehicles,
            hard_time_windows=True,
            time_limit_sec=args.time_limit,
            soft_lateness_penalty=args.soft_lateness_penalty,
            scale=args.scale,
        )
        if result["solved"]:
            return row_for(instance, keep, "ortools_min_k_hard_tw", inst, vehicles, result)
    result["elapsed_sec"] = result.get("elapsed_sec", 0.0)
    return row_for(instance, keep, "ortools_min_k_hard_tw", base_inst, sol["max_vehicles"], result)


def run(args):
    rows = []
    for instance in parse_csv_list(args.instances, str):
        sol = load_solomon_txt(os.path.join(args.sol_dir, f"{instance}.txt"))
        for keep in parse_csv_list(args.keeps, int):
            base_inst = reduce_instance(sol, keep=keep)
            print(f"Running OR-Tools comparison: {instance} keep={keep} K={base_inst['n_vehicles']}")

            same_hard = solve_case(
                base_inst,
                vehicles=base_inst["n_vehicles"],
                hard_time_windows=True,
                time_limit_sec=args.time_limit,
                soft_lateness_penalty=args.soft_lateness_penalty,
                scale=args.scale,
            )
            rows.append(
                row_for(
                    instance,
                    keep,
                    "ortools_same_k_hard_tw",
                    base_inst,
                    base_inst["n_vehicles"],
                    same_hard,
                )
            )

            same_soft = solve_case(
                base_inst,
                vehicles=base_inst["n_vehicles"],
                hard_time_windows=False,
                time_limit_sec=args.time_limit,
                soft_lateness_penalty=args.soft_lateness_penalty,
                scale=args.scale,
            )
            rows.append(
                row_for(
                    instance,
                    keep,
                    "ortools_same_k_soft_tw",
                    base_inst,
                    base_inst["n_vehicles"],
                    same_soft,
                )
            )

            if args.find_min_hard_vehicles:
                rows.append(solve_min_k_hard(instance, keep, sol, base_inst, args))

    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare QUBO VRPTW results with OR-Tools")
    parser.add_argument("--instances", default="c101,r101,rc101")
    parser.add_argument("--keeps", default="30,40")
    parser.add_argument("--sol-dir", default="data/solomon-100")
    parser.add_argument("--output", default="results/ortools_vrptw_comparison.csv")
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--soft-lateness-penalty", type=float, default=6.0)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--find-min-hard-vehicles", action="store_true")
    args = parser.parse_args()

    rows = run(args)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "instance",
        "keep",
        "capacity_min_vehicles",
        "mode",
        "vehicles_limit",
        "vehicles_used",
        "solved",
        "status",
        "distance",
        "tw_violations",
        "tw_violation_amount",
        "loads",
        "route_sizes",
        "elapsed_sec",
        "routes",
    ]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
