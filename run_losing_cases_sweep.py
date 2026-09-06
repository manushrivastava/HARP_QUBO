"""
run_losing_cases_sweep.py
=====================================================================
Runs explore_losses.py over every case that lost to OR-Tools in the
90-case sweep (results/multi_seed_full/summary.csv), in parallel
across CPU cores. Does not touch run_multi_seed_experiment.py or any
other original-pipeline file.

The 15 losing cases below (instance, keep, exact_score, ortools_score)
were computed directly from results/multi_seed_full/summary.csv:
every row where exact_score > ortools_score. Sorted worst-relative-
loss first.

    rc206 keep=70   exact=6176.29  ortools=2310.34  (+167.33%)
    rc107 keep=70   exact=1541.62  ortools=1309.44  (+17.73%)
    rc108 keep=70   exact=1211.67  ortools=1043.85  (+16.08%)
    rc104 keep=100  exact=2295.84  ortools=2046.73  (+12.17%)
    c203  keep=70   exact=2539.42  ortools=2285.46  (+11.11%)
    r112  keep=100  exact=3711.95  ortools=3402.40  (+9.10%)
    rc203 keep=70   exact=1103.14  ortools=1012.33  (+8.97%)
    rc201 keep=70   exact=21082.41 ortools=19416.35 (+8.58%)
    rc103 keep=70   exact=2003.23  ortools=1884.83  (+6.28%)  [already tested by hand -- became a WIN]
    rc101 keep=70   exact=5539.30  ortools=5282.69  (+4.86%)  [already tested by hand -- still loses]
    rc104 keep=70   exact=915.56   ortools=884.11   (+3.56%)
    rc204 keep=70   exact=675.09   ortools=653.31   (+3.33%)
    rc205 keep=70   exact=3490.12  ortools=3459.40  (+0.89%)
    c203  keep=100  exact=604.58   ortools=600.21   (+0.73%)
    rc208 keep=70   exact=695.51   ortools=691.31   (+0.61%)

Usage:
    python run_losing_cases_sweep.py --target-pool-size 1000 --workers 4

Add --cases rc101:70,rc103:70 to restrict to a subset. Skips cases
that already have a results/explore_losses/<case>/result.json unless
--force-rerun is given (resumable, like run_all_solomon.py).
=====================================================================
"""
import argparse
import csv
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import explore_losses as el

ALL_LOSING_CASES = [
    ("rc206", 70), ("rc107", 70), ("rc108", 70), ("rc104", 100), ("c203", 70),
    ("r112", 100), ("rc203", 70), ("rc201", 70), ("rc103", 70), ("rc101", 70),
    ("rc104", 70), ("rc204", 70), ("rc205", 70), ("c203", 100), ("rc208", 70),
]


def _run_one(argv_dict):
    parser = el.build_args()
    args = parser.parse_args(["--instance", argv_dict["instance"], "--keep", str(argv_dict["keep"])])
    for k, v in argv_dict.items():
        setattr(args, k, v)
    try:
        result = el.run(args)
        return {"ok": True, "instance": args.instance, "keep": args.keep, "result": result}
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole sweep
        return {
            "ok": False, "instance": args.instance, "keep": args.keep,
            "error": str(exc), "traceback": traceback.format_exc(),
        }


def write_explore_summary(output_dir):
    rows = []
    if os.path.isdir(output_dir):
        for name in sorted(os.listdir(output_dir)):
            result_path = os.path.join(output_dir, name, "result.json")
            if os.path.exists(result_path):
                with open(result_path, encoding="utf-8") as f:
                    rows.append(json.load(f))
    summary_path = os.path.join(output_dir, "explore_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance", "keep", "family", "baseline_pool_size", "final_pool_size",
                    "target_reached", "ortools_score", "exact_score", "exact_timed_out",
                    "exact_beats_ortools", "exact_margin", "neal_best_score", "neal_beats_ortools",
                    "swap_best_score", "swap_beats_ortools", "total_seconds"])
        for r in rows:
            exact = r.get("exact", {})
            neal = r.get("neal", {})
            swap = r.get("swap_annealer", {})
            w.writerow([
                r.get("instance"), r.get("keep"), r.get("family"),
                r.get("baseline_pool_size"), r.get("pool_size"), r.get("target_reached"),
                r.get("ortools_score"), exact.get("score"), exact.get("timed_out"),
                exact.get("beats_ortools"), exact.get("margin_vs_ortools"),
                neal.get("best_score"), neal.get("beats_ortools"),
                swap.get("best_score"), swap.get("beats_ortools"),
                r.get("total_seconds"),
            ])
    return summary_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="",
                         help="comma-separated instance:keep pairs, e.g. rc101:70,rc103:70 "
                              "(default: all 15 known losing cases)")
    parser.add_argument("--target-pool-size", type=int, default=1000)
    parser.add_argument("--seed-start", type=int, default=701)
    parser.add_argument("--max-new-seeds", type=int, default=30)
    parser.add_argument("--insertion-noise", type=float, default=0.2)
    parser.add_argument("--remove-counts", default="10,15,20,25")
    parser.add_argument("--max-exact-n", type=int, default=2000)
    parser.add_argument("--exact-timeout-seconds", type=int, default=3600)
    parser.add_argument("--baseline-dir", default="results/multi_seed_full")
    parser.add_argument("--output-dir", default="results/explore_losses")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force-rerun", action="store_true")
    args = parser.parse_args()

    if args.cases:
        cases = []
        for tok in args.cases.split(","):
            tok = tok.strip()
            if not tok:
                continue
            inst, keep = tok.split(":")
            cases.append((inst.strip(), int(keep.strip())))
    else:
        cases = ALL_LOSING_CASES

    jobs = []
    n_skipped = 0
    for inst, keep in cases:
        result_path = os.path.join(args.output_dir, f"{inst}_keep{keep}", "result.json")
        if not args.force_rerun and os.path.exists(result_path):
            n_skipped += 1
            continue
        jobs.append({
            "instance": inst, "keep": keep,
            "baseline_dir": args.baseline_dir, "output_dir": args.output_dir,
            "target_pool_size": args.target_pool_size, "seed_start": args.seed_start,
            "max_new_seeds": args.max_new_seeds, "insertion_noise": args.insertion_noise,
            "remove_counts": args.remove_counts, "max_exact_n": args.max_exact_n,
            "exact_timeout_seconds": args.exact_timeout_seconds, "no_progress": True,
        })

    if n_skipped:
        print(f"Skipping {n_skipped} case(s) that already have a result.json "
              f"(pass --force-rerun to redo them anyway)", flush=True)
    print(f"Running {len(jobs)} case(s), {args.workers} parallel workers, "
          f"target pool size {args.target_pool_size}", flush=True)

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_run_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            outcome = fut.result()
            tag = f"{outcome['instance']}_keep{outcome['keep']}"
            if outcome["ok"]:
                print(f"[DONE] {tag}", flush=True)
            else:
                print(f"[FAILED] {tag}: {outcome['error']}", flush=True)
                failures.append(outcome)

    os.makedirs(args.output_dir, exist_ok=True)
    if failures:
        failures_path = os.path.join(args.output_dir, "failures.json")
        with open(failures_path, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} job(s) failed -- see {failures_path}", flush=True)

    summary_path = write_explore_summary(args.output_dir)
    print(f"\nWrote combined summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
