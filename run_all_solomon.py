"""
run_all_solomon.py
=====================================================================
Sweep run_multi_seed_experiment.run() over the complete Solomon
dataset (all 45 instances shipped in data/solomon-100/) x a list of
keep-levels x the multi-seed LNS + pool + exact/neal/swap-annealer
pipeline, then aggregate every per-instance result.json into one
combined CSV for the paper.

Runs instances in parallel across CPU cores (ProcessPoolExecutor) --
this pipeline is CPU-bound (numpy + neal, no GPU code anywhere), so
on a big multi-core server set --workers to roughly the physical
core count.

Usage:
    python run_all_solomon.py --keep 100 --num-seeds 8 --workers 16
    python run_all_solomon.py --keep 70,100 --num-seeds 8 --workers 16 \
        --output-dir results/multi_seed_full

Add --instances r101,r102 to restrict to a subset (useful for a quick
smoke test before committing to the full 45-instance x keep-level
sweep, which is a large amount of compute -- see README.md).
=====================================================================
"""
import argparse
import csv
import glob
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import run_multi_seed_experiment as rmse

DEFAULT_SOL_DIR = os.path.join(os.path.dirname(__file__), "data", "solomon-100")


def discover_instances(sol_dir):
    return sorted(
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(sol_dir, "*.txt"))
    )


def _run_one(argv_dict):
    """Runs in a worker process: rebuild an argparse.Namespace and call run()."""
    parser = rmse.build_args()
    args = parser.parse_args(["--instance", argv_dict["instance"]])
    for k, v in argv_dict.items():
        setattr(args, k, v)
    try:
        result = rmse.run(args)
        return {"ok": True, "instance": args.instance, "keep": args.keep, "result": result}
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole sweep
        return {
            "ok": False, "instance": args.instance, "keep": args.keep,
            "error": str(exc), "traceback": traceback.format_exc(),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sol-dir", default=DEFAULT_SOL_DIR)
    parser.add_argument("--instances", default="", help="comma-separated subset, e.g. r101,r102 "
                                                          "(default: all instances found in --sol-dir)")
    parser.add_argument("--keep", default="70,100", help="comma-separated keep levels, e.g. 70,100")
    parser.add_argument("--no-ortools", action="store_true", help="skip the OR-Tools benchmark solve")
    parser.add_argument("--ortools-time-limit", type=int, default=30)
    parser.add_argument("--num-seeds", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=601)
    parser.add_argument("--lns-iterations", type=int, default=120)
    parser.add_argument("--max-exact-n", type=int, default=700)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/multi_seed")
    parser.add_argument("--no-progress", action="store_true", default=True,
                         help="progress bars are disabled by default in the parallel sweep "
                              "(they interleave badly across worker processes)")
    args = parser.parse_args()

    instances = [s.strip() for s in args.instances.split(",") if s.strip()] or discover_instances(args.sol_dir)
    keep_levels = [int(s.strip()) for s in args.keep.split(",") if s.strip()]

    jobs = []
    for inst in instances:
        for keep in keep_levels:
            jobs.append({
                "instance": inst, "keep": keep, "sol_dir": args.sol_dir,
                "num_seeds": args.num_seeds, "seed_base": args.seed_base,
                "lns_iterations": args.lns_iterations, "max_exact_n": args.max_exact_n,
                "run_ortools": not args.no_ortools, "ortools_time_limit": args.ortools_time_limit,
                "output_dir": args.output_dir, "no_progress": True,
            })

    print(f"Sweeping {len(instances)} instances x {len(keep_levels)} keep-level(s) "
          f"= {len(jobs)} jobs, {args.workers} parallel workers", flush=True)

    all_results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            outcome = fut.result()
            tag = f"{outcome['instance']}_keep{outcome['keep']}"
            if outcome["ok"]:
                print(f"[DONE] {tag}", flush=True)
                all_results.append(outcome["result"])
            else:
                print(f"[FAILED] {tag}: {outcome['error']}", flush=True)
                failures.append(outcome)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_csv = os.path.join(args.output_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "instance", "keep", "K", "customers", "num_seeds", "pool_size",
            "best_single_lns_score",
            "ortools_seconds", "ortools_solved", "ortools_score",
            "exact_seconds", "exact_feasible", "exact_score",
            "neal_seconds", "neal_valid_samples", "neal_num_reads", "neal_best_score",
            "swap_seconds", "swap_valid", "swap_restarts", "swap_best_score",
            "total_seconds",
        ])
        for r in sorted(all_results, key=lambda r: (r["instance"], r["keep"])):
            ortools = r.get("ortools", {})
            exact = r.get("exact", {})
            neal = r.get("neal", {})
            swap = r.get("swap_annealer", {})
            w.writerow([
                r["instance"], r["keep"], r["K"], r["customers"], r["num_seeds"], r["pool_size"],
                r["best_single_lns_score"],
                ortools.get("seconds"), ortools.get("solved"), ortools.get("score"),
                exact.get("seconds"), exact.get("feasible"), exact.get("score"),
                neal.get("seconds"), neal.get("valid_samples"), neal.get("num_reads"), neal.get("best_score"),
                swap.get("seconds"), swap.get("valid"), swap.get("restarts"), swap.get("best_score"),
                r.get("total_seconds"),
            ])

    if failures:
        failures_path = os.path.join(args.output_dir, "failures.json")
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} job(s) failed -- see {failures_path}", flush=True)

    print(f"\nWrote combined summary ({len(all_results)} rows) -> {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
