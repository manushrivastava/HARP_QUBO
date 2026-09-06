"""
explore_losses.py
=====================================================================
Standalone follow-up experiment for the losing cases in the 90-case
multi-seed sweep (results/multi_seed_full). Does NOT modify or import
anything that changes run_multi_seed_experiment.py's behavior -- it
is a fully separate script so the original pipeline stays exactly as
it was and can still be re-run unchanged.

Hypothesis being tested: for a case where the exact solver's pool
score lost to OR-Tools, does substantially MORE LNS exploration
(bigger destroy sizes + nonzero insertion noise) -- continued from
the same verified real baseline pool, up to a much larger pool size
(default 1000 routes) -- close the gap? Then: which of exact / neal
(QUBO bit-flip) / swap-move-annealer actually finds the improvement,
if one exists, on that same final pool?

This is the same methodology already validated by hand on rc101_keep70
and rc103_keep70 (see docs/why_quantum_storyline.tex, "Full-dataset
validation" section), generalized into a reusable, resumable script:

  1. Load the case's VERIFIED real baseline pool from
     results/multi_seed_full/<instance>_keep<keep>/merged_pool.csv
     (sanity-checked against that case's own result.json pool_size --
     refuses to continue if the file looks stale/corrupted).
  2. Rebuild the SAME construction-stage starts used by the original
     pipeline (random_partitions=240, max_routes=1200, seed=84,
     best 90 kept) -- this is a genuine continuation, not a new run.
  3. Run NEW LNS seeds (default starting at 701, so they never collide
     with the original pipeline's seed range 601-608) with more
     aggressive exploration parameters (default: insertion_noise=0.2,
     remove_counts=10,15,20,25, vs. the original 0.0 / 3,4,5,6,8),
     adding every route found (plus guarded-repair output) to the pool,
     until the pool reaches --target-pool-size (default 1000) or
     --max-new-seeds is hit (safety cap).
  4. Once done, solve the FINAL pool with: the exact branch-and-bound
     solver (wrapped in a subprocess with --exact-timeout-seconds, since
     branch-and-bound has no built-in time budget and a 1000-route pool
     can in principle blow up), neal, and the swap-move annealer -- then
     compare all three against the case's already-known OR-Tools score
     (read from the baseline result.json; OR-Tools is independent of
     the pool, so there's no need to re-solve it).
  5. Write results/explore_losses/<instance>_keep<keep>/result.json
     and merged_pool.csv (atomic writes throughout -- a .tmp file then
     os.replace(), never a bare open(path, "w") on the shared pool file,
     to avoid the interleaved-write corruption seen earlier when two
     processes wrote the same CSV concurrently).

Usage (single case):
    python explore_losses.py --instance rc101 --keep 70 \
        --target-pool-size 1000

Run run_losing_cases_sweep.py to do this for all known losing cases.
=====================================================================
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time
from types import SimpleNamespace

from new_qubo_objectives import load_solomon_txt, reduce_instance
from route_pool_qubo_vrptw import (
    build_route_pool,
    route_key,
    solve_route_pool_exact,
    solve_route_pool_qubo,
)
from probe_strong_route_pool_repair import lns_route_sets, add_route_set, route_signature
from route_repair import standalone_improve_routes
from swap_move_annealer import swap_anneal

DEFAULT_SOL_DIR = os.path.join(os.path.dirname(__file__), "data", "solomon-100")


def family_of(instance_name):
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
    parser.add_argument("--instance", required=True, help="e.g. rc101, r112, c203")
    parser.add_argument("--keep", type=int, default=70)
    parser.add_argument("--sol-dir", default=DEFAULT_SOL_DIR)
    parser.add_argument("--tw-weight", type=float, default=6.0)
    parser.add_argument("--slack-vehicles", type=int, default=0)

    # Where to find the verified real baseline (the original 90-case sweep).
    parser.add_argument("--baseline-dir", default="results/multi_seed_full")

    # Construction stage -- MUST match the original pipeline's defaults so
    # this is a genuine continuation of the same experiment.
    parser.add_argument("--construction-random-partitions", type=int, default=240)
    parser.add_argument("--construction-max-routes", type=int, default=1200)
    parser.add_argument("--construction-seed", type=int, default=84)
    parser.add_argument("--num-starts", type=int, default=90)

    # New-seed LNS exploration parameters (deliberately more aggressive
    # than the original pipeline's defaults: insertion-noise=0.0,
    # remove-counts=3,4,5,6,8).
    parser.add_argument("--seed-start", type=int, default=701,
                         help="first new seed id; must not overlap the original "
                              "pipeline's seed range (601..600+num_seeds)")
    parser.add_argument("--max-new-seeds", type=int, default=30,
                         help="safety cap on how many new seeds to try even if "
                              "--target-pool-size is never reached")
    parser.add_argument("--target-pool-size", type=int, default=1000)
    parser.add_argument("--lns-iterations", type=int, default=220)
    parser.add_argument("--keep-lns-sets", type=int, default=180)
    parser.add_argument("--destroy-strategies", default="random,late,mixed")
    parser.add_argument("--repair-modes", default="best,regret,random_best")
    parser.add_argument("--repair-weights", default="6,8,10,12,15")
    parser.add_argument("--remove-counts", default="10,15,20,25",
                         help="original pipeline default is 3,4,5,6,8 -- this "
                              "script defaults to larger, more disruptive removals")
    parser.add_argument("--due-order-weight", type=float, default=0.0)
    parser.add_argument("--insertion-noise", type=float, default=0.2,
                         help="original pipeline default is 0.0")
    parser.add_argument("--local-search-passes", type=int, default=1)

    parser.add_argument("--guarded-repair-sets", type=int, default=12)
    parser.add_argument("--guarded-repair-tw-weight", type=float, default=10.0)
    parser.add_argument("--guarded-local-passes", type=int, default=2)
    parser.add_argument("--guarded-inter-route-passes", type=int, default=1)

    # No master-pool-cap here on purpose: the whole point is to test the
    # FULL grown pool (up to --target-pool-size), not a capped-down one.
    parser.add_argument("--max-exact-n", type=int, default=2000,
                         help="exact solver is skipped above this pool size "
                              "regardless of --exact-timeout-seconds")
    parser.add_argument("--exact-timeout-seconds", type=int, default=3600,
                         help="exact solve runs in a subprocess; killed and "
                              "marked timed-out if it exceeds this")
    parser.add_argument("--exact-check-every-seed", action="store_true",
                         help="also run a bounded exact check after every new "
                              "seed (only while pool size <= --max-exact-n), to "
                              "see when/if a win first appears -- off by default "
                              "since it can be a lot of extra compute")

    parser.add_argument("--coverage-weight", type=float, default=120.0)
    parser.add_argument("--vehicle-weight", type=float, default=120.0)
    parser.add_argument("--route-score-scale", type=float, default=1000.0)
    parser.add_argument("--num-reads", type=int, default=1000)
    parser.add_argument("--num-sweeps", type=int, default=1000)
    parser.add_argument("--neal-seed", type=int, default=100)

    parser.add_argument("--swap-restarts", type=int, default=200)
    parser.add_argument("--swap-steps", type=int, default=5000)

    parser.add_argument("--output-dir", default="results/explore_losses")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def save_pool_csv_atomic(path, pool):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["route", "score", "source"])
        for r in pool:
            w.writerow(["-".join(str(c) for c in r["route"]), r["score"], r.get("source", "")])
    os.replace(tmp, path)  # atomic rename -- avoids interleaved-write corruption


def load_baseline_pool(baseline_dir, inst_name, keep):
    case_dir = os.path.join(baseline_dir, f"{inst_name}_keep{keep}")
    pool_csv = os.path.join(case_dir, "merged_pool.csv")
    result_json = os.path.join(case_dir, "result.json")
    if not os.path.exists(pool_csv) or not os.path.exists(result_json):
        raise FileNotFoundError(
            f"no baseline found for {inst_name}_keep{keep} under {baseline_dir} "
            f"(expected merged_pool.csv and result.json)"
        )
    with open(result_json, encoding="utf-8") as f:
        baseline_result = json.load(f)

    route_map = {}
    with open(pool_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            route = [int(t) for t in row["route"].split("-")]
            route_map[route_signature(route)] = {
                "route": route,
                "score": float(row["score"]),
                "source": row.get("source", "baseline"),
            }

    expected_n = baseline_result.get("pool_size")
    if expected_n is not None and len(route_map) != expected_n:
        raise AssertionError(
            f"baseline pool for {inst_name}_keep{keep} looks stale/corrupted: "
            f"merged_pool.csv has {len(route_map)} unique routes but result.json "
            f"says pool_size={expected_n}. Refusing to continue -- verify the "
            f"file before rerunning."
        )
    return route_map, baseline_result


def _exact_worker(inst, pool, tw_weight, q):
    try:
        res = solve_route_pool_exact(inst, pool, tw_weight=tw_weight)
        q.put(("ok", res))
    except Exception as exc:  # noqa: BLE001 -- report back to the parent
        q.put(("error", str(exc)))


def solve_exact_with_timeout(inst, pool, tw_weight, timeout_s):
    """Runs the branch-and-bound exact solver in a subprocess so a pathological
    pool can't hang the whole experiment. Returns (result_or_None, timed_out)."""
    ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
    q = ctx.Queue()
    p = ctx.Process(target=_exact_worker, args=(inst, pool, tw_weight, q))
    p.start()
    p.join(timeout_s)
    if p.is_alive():
        p.terminate()
        p.join()
        return None, True
    if not q.empty():
        status, payload = q.get()
        if status == "ok":
            return payload, False
    return None, False


def run(args):
    progress = not args.no_progress
    t_start = time.perf_counter()
    inst_name = args.instance.lower()
    sol = load_solomon_txt(os.path.join(args.sol_dir, f"{inst_name}.txt"))
    inst = reduce_instance(sol, keep=args.keep, slack_vehicles=args.slack_vehicles)
    K = inst["n_vehicles"]
    n_customers = len(inst["kept_ids"])
    family = family_of(inst_name)

    out_dir = os.path.join(args.output_dir, f"{inst_name}_keep{args.keep}")
    os.makedirs(out_dir, exist_ok=True)
    pool_csv_path = os.path.join(out_dir, "merged_pool.csv")
    result_json_path = os.path.join(out_dir, "result.json")

    print(f"[{inst_name}_keep{args.keep}] K={K} customers={n_customers} family={family}", flush=True)

    print("=" * 70)
    print("STEP 1: load verified real baseline pool")
    print("=" * 70)
    route_map, baseline_result = load_baseline_pool(args.baseline_dir, inst_name, args.keep)
    baseline_pool_size = len(route_map)
    ortools_score = baseline_result.get("ortools", {}).get("score")
    print(f"Verified clean: {baseline_pool_size} routes. OR-Tools score to beat: {ortools_score}", flush=True)

    print("=" * 70)
    print("STEP 2: rebuild the same construction-stage starts")
    print("=" * 70)
    t0 = time.perf_counter()
    _, partition_rows = build_route_pool(
        inst, tw_weight=args.tw_weight,
        random_partitions=args.construction_random_partitions,
        max_routes=args.construction_max_routes,
        seed=args.construction_seed, progress=progress,
    )
    t_construction = time.perf_counter() - t0
    starts = [p["routes"] for p in partition_rows[: args.num_starts]]
    print(f"Built {len(partition_rows)} partitions ({t_construction:.1f}s), using best {len(starts)} as LNS starts",
          flush=True)

    print("=" * 70)
    print(f"STEP 3: add new-parameter seeds (from {args.seed_start}) until pool "
          f">= {args.target_pool_size} (max {args.max_new_seeds} new seeds)")
    print(f"  insertion_noise={args.insertion_noise}  remove_counts={args.remove_counts}")
    print("=" * 70)
    seed_details = []
    seed = args.seed_start
    n_new_seeds = 0
    while len(route_map) < args.target_pool_size and n_new_seeds < args.max_new_seeds:
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
        for row in lns_sets:
            add_route_set(route_map, inst, row["routes"], f"seed{seed}_explore_{row['source']}", args.tw_weight)

        guarded_args = SimpleNamespace(
            tw_weight=args.tw_weight, repair_tw_weight=args.guarded_repair_tw_weight,
            local_search_passes=args.guarded_local_passes,
            inter_route_passes=args.guarded_inter_route_passes,
        )
        n_guarded = 0
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
                n_guarded += 1
                add_route_set(route_map, inst, repaired, f"seed{seed}_explore_guarded", args.tw_weight)

        pool = sorted(route_map.values(), key=lambda r: r["score"])
        save_pool_csv_atomic(pool_csv_path, pool)

        line = (f"seed={seed}: {len(lns_sets)} LNS sets, seed_best={seed_best}, "
                f"guarded {n_guarded}/{min(args.guarded_repair_sets, len(lns_sets))} "
                f"({elapsed:.1f}s) -> pool={len(pool)}/{args.target_pool_size}")
        detail = {"seed": seed, "n_sets": len(lns_sets), "best_score": seed_best,
                  "seconds": elapsed, "n_guarded_accepted": n_guarded, "pool_size_after": len(pool)}

        if args.exact_check_every_seed and len(pool) <= args.max_exact_n:
            t0 = time.perf_counter()
            exact_result, timed_out = solve_exact_with_timeout(
                inst, pool, args.tw_weight, args.exact_timeout_seconds)
            exact_elapsed = time.perf_counter() - t0
            exact_score = exact_result["score"] if exact_result else None
            beats = exact_score is not None and ortools_score is not None and exact_score < ortools_score - 1e-6
            detail["intermediate_exact_score"] = exact_score
            detail["intermediate_exact_seconds"] = exact_elapsed
            line += f"  exact_score={exact_score} ({exact_elapsed:.1f}s) beats_ortools={beats}"

        seed_details.append(detail)
        print(line, flush=True)
        seed += 1
        n_new_seeds += 1

    final_pool = sorted(route_map.values(), key=lambda r: r["score"])
    save_pool_csv_atomic(pool_csv_path, final_pool)
    n_pool = len(final_pool)
    target_reached = n_pool >= args.target_pool_size
    print("\n" + "=" * 70)
    print(f"{'TARGET REACHED' if target_reached else 'SAFETY CAP HIT'}: "
          f"{n_pool} routes after {n_new_seeds} new seeds")
    print("=" * 70, flush=True)

    # verification, matching the original pipeline's checks
    customer_sets = [route_signature(r["route"]) for r in final_pool]
    n_duplicate_customer_sets = len(customer_sets) - len(set(customer_sets))
    seed_origin_counts = {}
    for r in final_pool:
        src = r.get("source", "")
        tag = src.split("_", 1)[0] if src.startswith("seed") else "baseline"
        seed_origin_counts[tag] = seed_origin_counts.get(tag, 0) + 1

    result = {
        "instance": inst_name, "keep": args.keep, "K": K, "customers": n_customers, "family": family,
        "baseline_pool_size": baseline_pool_size,
        "target_pool_size": args.target_pool_size,
        "pool_size": n_pool,
        "target_reached": target_reached,
        "n_new_seeds": n_new_seeds,
        "seed_start": args.seed_start,
        "insertion_noise": args.insertion_noise,
        "remove_counts": args.remove_counts,
        "seed_details": seed_details,
        "construction_seconds": t_construction,
        "ortools_score": ortools_score,
        "ortools_source": "carried over from baseline result.json (OR-Tools is "
                           "independent of the route pool, not re-solved here)",
        "verification": {
            "n_duplicate_customer_sets": n_duplicate_customer_sets,
            "seed_origin_counts": seed_origin_counts,
        },
    }

    # --- exact (bounded by both max-exact-n and a wall-clock timeout) ---
    if n_pool <= args.max_exact_n:
        covered = set()
        for r in final_pool:
            covered |= set(route_key(r["route"]))
        if set(inst["kept_ids"]).issubset(covered):
            t0 = time.perf_counter()
            exact_result, timed_out = solve_exact_with_timeout(
                inst, final_pool, args.tw_weight, args.exact_timeout_seconds)
            t_exact = time.perf_counter() - t0
            exact_score = exact_result["score"] if exact_result else None
            beats_ortools = (
                exact_score is not None and ortools_score is not None
                and exact_score < ortools_score - 1e-6
            )
            result["exact"] = {
                "seconds": t_exact, "timed_out": timed_out,
                "feasible": exact_result is not None,
                "score": exact_score, "beats_ortools": beats_ortools,
                "margin_vs_ortools": (ortools_score - exact_score)
                if (exact_score is not None and ortools_score is not None) else None,
            }
            print(f"  exact: {t_exact:.1f}s timed_out={timed_out} score={exact_score} "
                  f"beats_ortools={beats_ortools}", flush=True)
        else:
            result["exact"] = {"skipped": "pool does not cover all customers"}
    else:
        result["exact"] = {"skipped": f"pool size {n_pool} exceeds max-exact-n {args.max_exact_n}"}
        print(f"  exact: skipped (pool size {n_pool} > max-exact-n {args.max_exact_n})", flush=True)

    # --- neal ---
    neal_args = SimpleNamespace(
        tw_weight=args.tw_weight, coverage_weight=args.coverage_weight,
        vehicle_weight=args.vehicle_weight, route_score_scale=args.route_score_scale,
        num_reads=args.num_reads, num_sweeps=args.num_sweeps, seed=args.neal_seed,
        progress=progress,
    )
    t0 = time.perf_counter()
    neal_best, neal_info = solve_route_pool_qubo(inst, final_pool, neal_args)
    t_neal = time.perf_counter() - t0
    neal_score = neal_best["score"] if neal_best else None
    result["neal"] = {
        "seconds": t_neal, "valid_samples": neal_info["valid_samples"],
        "num_reads": args.num_reads, "qubo_terms": neal_info["qubo_terms"],
        "best_score": neal_score,
        "beats_ortools": (neal_score is not None and ortools_score is not None
                           and neal_score < ortools_score - 1e-6),
    }
    print(f"  neal: {t_neal:.1f}s valid={neal_info['valid_samples']}/{args.num_reads} best={neal_score}",
          flush=True)

    # --- swap-move annealer ---
    t0 = time.perf_counter()
    swap_result = swap_anneal(inst, final_pool, n_restarts=args.swap_restarts,
                               n_steps=args.swap_steps, seed=args.neal_seed)
    t_swap = time.perf_counter() - t0
    swap_score = swap_result["best_score"]
    result["swap_annealer"] = {
        "seconds": t_swap, "valid": swap_result["n_valid"], "restarts": args.swap_restarts,
        "best_score": swap_score,
        "beats_ortools": (swap_score is not None and ortools_score is not None
                           and swap_score < ortools_score - 1e-6),
    }
    print(f"  swap-annealer: {t_swap:.1f}s valid={swap_result['n_valid']}/{args.swap_restarts} "
          f"best={swap_score}", flush=True)

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
