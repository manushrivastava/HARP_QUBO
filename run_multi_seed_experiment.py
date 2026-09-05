"""
run_multi_seed_experiment.py
=====================================================================
Single-instance driver for the "why quantum" storyline's core claims:

  Point 3: does pooling routes from multiple INDEPENDENT LNS runs
           (different seeds) beat the best single LNS run?
  Point 4: does that pooling make the route-selection problem too
           large for the classical EXACT solver, while classical
           heuristics (neal bit-flip QUBO, swap-move annealer) run
           fast but with a quality/reliability gap?

For one Solomon instance + keep level:
  1. Build a construction-stage batch of complete route-set "starts"
     (build_route_pool's partition_rows output).
  2. Run lns_route_sets() N_SEEDS times with N_SEEDS different seeds
     -- genuinely independent LNS campaigns, not just different
     random draws within one campaign.
  3. Track the single best score any ONE seed's LNS campaign found
     on its own (the "best single LNS run" baseline).
  4. Pool every route surfacing from every seed (deduped, best score
     kept), checkpointing to disk after every seed so partial
     progress survives interruption.
  5. Solve the resulting pool with:
       - the classical EXACT solver (branch-and-bound) -- skipped if
         the pool exceeds --max-exact-n, since real timing data on
         this project showed multi-hour blowups well before N=1000
         on genuinely competitive pools
       - the QUBO + neal (classical simulated annealing, bit-flip)
       - the swap-move annealer (cardinality-preserving classical fix)
  6. Write one result row (JSON) with everything needed to reproduce
     the paper's tables: instance, keep, seeds used, pool size, best
     single-LNS score, multi-seed pooled score (exact/neal/swap),
     validity rates, and wall-clock timings for every stage.

Usage:
    python run_multi_seed_experiment.py --instance r101 --keep 100 \
        --num-seeds 8 --output-dir results/multi_seed

Run run_all_solomon.py to sweep this over the full Solomon dataset.
=====================================================================
"""
import argparse
import csv
import json
import os
import time
from types import SimpleNamespace

from new_qubo_objectives import load_solomon_txt, reduce_instance
from route_pool_qubo_vrptw import (
    build_route_pool,
    route_key,
    routes_score,
    solve_route_pool_exact,
    solve_route_pool_qubo,
)
from probe_strong_route_pool_repair import lns_route_sets, add_route_set
from route_repair import standalone_improve_routes
from swap_move_annealer import swap_anneal
from compare_with_ortools import solve_case

DEFAULT_SOL_DIR = os.path.join(os.path.dirname(__file__), "data", "solomon-100")


def family_of(instance_name):
    """R / C / RC family from the instance name prefix (rc.. before r.., since
    'rc101' also starts with 'r')."""
    name = instance_name.lower()
    if name.startswith("rc"):
        return "RC"
    if name.startswith("r"):
        return "R"
    if name.startswith("c"):
        return "C"
    return "UNKNOWN"


def build_args(parser=None):
    parser = parser or argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, help="e.g. r101, c101, rc205")
    parser.add_argument("--keep", type=int, default=100)
    parser.add_argument("--sol-dir", default=DEFAULT_SOL_DIR)
    parser.add_argument("--tw-weight", type=float, default=6.0)
    parser.add_argument("--slack-vehicles", type=int, default=0)

    # construction-stage "starts" for LNS -- matches the manuscript's Algorithm 2
    # (240 seeded-random partitions, retain up to 1200, select 90 lowest-score as starts)
    parser.add_argument("--construction-random-partitions", type=int, default=240)
    parser.add_argument("--construction-max-routes", type=int, default=1200)
    parser.add_argument("--construction-seed", type=int, default=84)
    parser.add_argument("--num-starts", type=int, default=90)

    # multi-seed LNS campaign -- 220 destroy/repair trials per start, retain the
    # 180 lowest-score valid sets (manuscript's per-seed recipe); num-seeds is
    # THIS project's own extension (the paper itself uses one seed)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=601)
    parser.add_argument("--lns-iterations", type=int, default=220)
    parser.add_argument("--keep-lns-sets", type=int, default=180)
    parser.add_argument("--destroy-strategies", default="random,late,mixed")
    parser.add_argument("--repair-modes", default="best,regret,random_best")
    parser.add_argument("--repair-weights", default="6,8,10,12,15")
    parser.add_argument("--remove-counts", default="3,4,5,6,8")
    parser.add_argument("--due-order-weight", type=float, default=0.0)
    parser.add_argument("--insertion-noise", type=float, default=0.0)
    parser.add_argument("--local-search-passes", type=int, default=1)

    # guarded repair on the top-N retained LNS sets per seed (manuscript: top 12)
    parser.add_argument("--guarded-repair-sets", type=int, default=12)
    parser.add_argument("--guarded-repair-tw-weight", type=float, default=10.0)
    parser.add_argument("--guarded-local-passes", type=int, default=2)
    parser.add_argument("--guarded-inter-route-passes", type=int, default=1)

    # master-pool cap after multi-seed merging -- manuscript uses a family-
    # dependent cap (1400 for R-family, 600 for C/RC-family); pass
    # --master-pool-cap to override with one fixed number for every instance
    parser.add_argument("--master-pool-cap-r", type=int, default=1400)
    parser.add_argument("--master-pool-cap-c-rc", type=int, default=600)
    parser.add_argument("--master-pool-cap", type=int, default=None,
                         help="override: use this cap for every instance regardless of family")

    # exact solver cutoff
    parser.add_argument("--max-exact-n", type=int, default=700,
                         help="skip exact solving if the merged pool exceeds this size "
                              "(real data on this project showed multi-hour blowups well "
                              "before N=1000 on genuinely competitive pools)")

    # neal (QUBO bit-flip) settings
    parser.add_argument("--coverage-weight", type=float, default=120.0)
    parser.add_argument("--vehicle-weight", type=float, default=120.0)
    parser.add_argument("--route-score-scale", type=float, default=1000.0)
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--neal-seed", type=int, default=100)

    # swap-move annealer settings
    parser.add_argument("--swap-restarts", type=int, default=200)
    parser.add_argument("--swap-steps", type=int, default=5000)

    # OR-Tools benchmark (same convention as the original pipeline)
    parser.add_argument("--run-ortools", dest="run_ortools", action="store_true", default=True)
    parser.add_argument("--no-ortools", dest="run_ortools", action="store_false")
    parser.add_argument("--ortools-time-limit", type=int, default=30)
    parser.add_argument("--ortools-scale", type=int, default=1000)

    parser.add_argument("--output-dir", default="results/multi_seed")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def load_pool_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append({
                "route": [int(t) for t in row["route"].split("-")],
                "score": float(row["score"]),
                "source": row.get("source", ""),
            })
    return rows


def save_pool_csv(path, pool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route", "score", "source"])
        for r in pool:
            w.writerow(["-".join(str(c) for c in r["route"]), r["score"], r.get("source", "")])


def run(args):
    progress = not args.no_progress
    t_start = time.perf_counter()
    inst_name = args.instance.lower()
    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{inst_name}.txt"))
    inst = reduce_instance(sol, keep=args.keep, slack_vehicles=args.slack_vehicles)
    K = inst["n_vehicles"]
    n_customers = len(inst["kept_ids"])

    out_dir = os.path.join(args.output_dir, f"{inst_name}_keep{args.keep}")
    os.makedirs(out_dir, exist_ok=True)
    pool_csv_path = os.path.join(out_dir, "merged_pool.csv")
    result_json_path = os.path.join(out_dir, "result.json")

    print(f"[{inst_name}_keep{args.keep}] K={K} customers={n_customers}", flush=True)

    result = {
        "instance": inst_name, "keep": args.keep, "K": K, "customers": n_customers,
    }

    # --- OR-Tools benchmark (independent of the LNS pool -- solves the instance directly) ---
    if args.run_ortools:
        t0 = time.perf_counter()
        ort = solve_case(
            inst, vehicles=K, hard_time_windows=False,
            time_limit_sec=args.ortools_time_limit,
            soft_lateness_penalty=args.tw_weight, scale=args.ortools_scale,
        )
        t_ortools = time.perf_counter() - t0
        ort_score = routes_score(inst, ort["routes"], args.tw_weight) if ort["solved"] else None
        result["ortools"] = {
            "seconds": t_ortools, "solved": ort["solved"], "status": ort["status"], "score": ort_score,
            "time_limit_sec": args.ortools_time_limit,
        }
        print(f"  ortools: {t_ortools:.1f}s solved={ort['solved']} score={ort_score}", flush=True)
    else:
        result["ortools"] = {"skipped": "--no-ortools"}

    # --- Step 1: construction-stage starts ---
    t0 = time.perf_counter()
    _, partition_rows = build_route_pool(
        inst, tw_weight=args.tw_weight,
        random_partitions=args.construction_random_partitions,
        max_routes=args.construction_max_routes,
        seed=args.construction_seed, progress=progress,
    )
    t_construction = time.perf_counter() - t0
    starts = [p["routes"] for p in partition_rows[: args.num_starts]]
    print(f"  construction: {len(partition_rows)} partitions built ({t_construction:.1f}s), "
          f"using best {len(starts)} as LNS starts", flush=True)

    # --- Step 2+3: multi-seed LNS campaign, tracking best SINGLE-seed score ---
    route_map = {}
    best_single_seed_score = None
    seed_details = []
    seeds = [args.seed_base + i for i in range(args.num_seeds)]
    for seed in seeds:
        lns_args = SimpleNamespace(
            tw_weight=args.tw_weight,
            destroy_strategies=args.destroy_strategies,
            repair_modes=args.repair_modes,
            repair_weights=args.repair_weights,
            remove_counts=args.remove_counts,
            due_order_weight=args.due_order_weight,
            insertion_noise=args.insertion_noise,
            local_search_passes=args.local_search_passes,
            lns_iterations=args.lns_iterations,
            keep_lns_sets=args.keep_lns_sets,
            progress=progress,
            seed=seed,
        )
        t0 = time.perf_counter()
        lns_sets = lns_route_sets(inst, starts, lns_args)
        elapsed = time.perf_counter() - t0
        seed_best = lns_sets[0]["score"] if lns_sets else None
        if seed_best is not None and (best_single_seed_score is None or seed_best < best_single_seed_score):
            best_single_seed_score = seed_best
        seed_details.append({"seed": seed, "n_sets": len(lns_sets), "best_score": seed_best, "seconds": elapsed})
        print(f"  seed={seed}: {len(lns_sets)} LNS sets, best={seed_best} ({elapsed:.1f}s)", flush=True)
        for row in lns_sets:
            add_route_set(route_map, inst, row["routes"], f"seed{seed}_{row['source']}", args.tw_weight)

        # guarded repair on this seed's top --guarded-repair-sets LNS sets
        # (manuscript Algorithm 2, lines 17-21: intra-route + relocate + swap
        # improvement applied only to the strongest few sets, then their
        # constituent routes are added to the shared candidate pool too)
        guarded_args = SimpleNamespace(
            tw_weight=args.tw_weight, repair_tw_weight=args.guarded_repair_tw_weight,
            local_search_passes=args.guarded_local_passes,
            inter_route_passes=args.guarded_inter_route_passes,
        )
        n_guarded_accepted = 0
        for row in lns_sets[: args.guarded_repair_sets]:
            repaired = standalone_improve_routes(inst, row["routes"], guarded_args)
            covered = set()
            ok = True
            for route in repaired:
                cs = set(route_key(route))
                if covered & cs:
                    ok = False
                    break
                covered |= cs
            if ok and covered == set(inst["kept_ids"]):
                n_guarded_accepted += 1
                add_route_set(route_map, inst, repaired, f"seed{seed}_{row['source']}_guarded", args.tw_weight)
        print(f"  seed={seed}: guarded repair accepted {n_guarded_accepted}/"
              f"{min(args.guarded_repair_sets, len(lns_sets))} sets", flush=True)

        merged_pool = sorted(route_map.values(), key=lambda r: r["score"])
        save_pool_csv(pool_csv_path, merged_pool)  # checkpoint after every seed

    merged_pool = sorted(route_map.values(), key=lambda r: r["score"])
    n_pool_before_cap = len(merged_pool)

    # master-pool cap: family-dependent (manuscript: 1400 for R, 600 for C/RC)
    # unless --master-pool-cap overrides with one fixed number for everything
    if args.master_pool_cap is not None:
        cap = args.master_pool_cap
    else:
        family = family_of(inst_name)
        cap = args.master_pool_cap_r if family == "R" else args.master_pool_cap_c_rc
    if len(merged_pool) > cap:
        merged_pool = merged_pool[:cap]
    save_pool_csv(pool_csv_path, merged_pool)
    n_pool = len(merged_pool)
    print(f"  multi-seed pool: {n_pool_before_cap} unique routes before cap, "
          f"{n_pool} after master-pool cap ({cap}), "
          f"best single-seed score={best_single_seed_score}", flush=True)

    # --- verification: no duplicates, and routes genuinely trace back to
    # multiple different seeds (not just one seed dominating) ---
    customer_sets = [frozenset(route_key(r["route"])) for r in merged_pool]
    n_duplicate_customer_sets = len(customer_sets) - len(set(customer_sets))
    seed_origin_counts = {}
    for r in merged_pool:
        src = r.get("source", "")
        seed_tag = src.split("_", 1)[0] if src.startswith("seed") else "unknown"
        seed_origin_counts[seed_tag] = seed_origin_counts.get(seed_tag, 0) + 1
    n_seeds_represented = len([k for k in seed_origin_counts if k != "unknown"])
    print(f"  verification: {n_duplicate_customer_sets} duplicate customer-sets in pool "
          f"(should be 0); routes trace back to {n_seeds_represented}/{args.num_seeds} "
          f"distinct seeds: {seed_origin_counts}", flush=True)

    result.update({
        "num_seeds": args.num_seeds, "seeds": seeds, "seed_details": seed_details,
        "construction_seconds": t_construction,
        "family": family_of(inst_name),
        "pool_size_before_cap": n_pool_before_cap,
        "master_pool_cap": cap,
        "pool_size": n_pool,
        "best_single_lns_score": best_single_seed_score,
        "verification": {
            "n_duplicate_customer_sets": n_duplicate_customer_sets,
            "n_seeds_represented": n_seeds_represented,
            "seed_origin_counts": seed_origin_counts,
        },
    })

    # --- Step 5a: exact solver (bounded) ---
    if n_pool <= args.max_exact_n:
        covered = set()
        for r in merged_pool:
            covered |= set(route_key(r["route"]))
        if set(inst["kept_ids"]).issubset(covered):
            t0 = time.perf_counter()
            exact_result = solve_route_pool_exact(inst, merged_pool, tw_weight=args.tw_weight)
            t_exact = time.perf_counter() - t0
            result["exact"] = {
                "seconds": t_exact,
                "feasible": exact_result is not None,
                "score": exact_result["score"] if exact_result else None,
            }
            print(f"  exact: {t_exact:.1f}s feasible={exact_result is not None} "
                  f"score={exact_result['score'] if exact_result else None}", flush=True)
        else:
            result["exact"] = {"skipped": "pool does not cover all customers"}
    else:
        result["exact"] = {"skipped": f"pool size {n_pool} exceeds max-exact-n {args.max_exact_n}"}
        print(f"  exact: skipped (pool size {n_pool} > max-exact-n {args.max_exact_n})", flush=True)

    # --- Step 5b: neal (QUBO bit-flip) ---
    neal_args = SimpleNamespace(
        tw_weight=args.tw_weight, coverage_weight=args.coverage_weight,
        vehicle_weight=args.vehicle_weight, route_score_scale=args.route_score_scale,
        num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.neal_seed,
        progress=progress,
    )
    t0 = time.perf_counter()
    neal_best, neal_info = solve_route_pool_qubo(inst, merged_pool, neal_args)
    t_neal = time.perf_counter() - t0
    result["neal"] = {
        "seconds": t_neal, "valid_samples": neal_info["valid_samples"],
        "num_reads": args.num_reads, "qubo_terms": neal_info["qubo_terms"],
        "best_score": neal_best["score"] if neal_best else None,
    }
    print(f"  neal: {t_neal:.1f}s valid={neal_info['valid_samples']}/{args.num_reads} "
          f"best={neal_best['score'] if neal_best else None}", flush=True)

    # --- Step 5c: swap-move annealer ---
    t0 = time.perf_counter()
    swap_result = swap_anneal(inst, merged_pool, n_restarts=args.swap_restarts,
                               n_steps=args.swap_steps, seed=args.neal_seed)
    t_swap = time.perf_counter() - t0
    result["swap_annealer"] = {
        "seconds": t_swap, "valid": swap_result["n_valid"], "restarts": args.swap_restarts,
        "best_score": swap_result["best_score"],
    }
    print(f"  swap-annealer: {t_swap:.1f}s valid={swap_result['n_valid']}/{args.swap_restarts} "
          f"best={swap_result['best_score']}", flush=True)

    result["total_seconds"] = time.perf_counter() - t_start
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[{inst_name}_keep{args.keep}] done in {result['total_seconds']:.1f}s -> {result_json_path}", flush=True)
    return result


def main():
    args = build_args().parse_args()
    run(args)


if __name__ == "__main__":
    main()
