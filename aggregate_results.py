"""
aggregate_results.py
=====================================================================
Build summary.csv and family_summary.csv from whatever result.json
files currently exist under an output directory -- safe to run at any
time, including WHILE run_all_solomon.py is still running (to check
progress), or AFTER killing it early (e.g. hitting a time limit) to
salvage every case that finished, without losing the aggregated
tables that only run_all_solomon.py itself would otherwise write at
the very end of a fully-completed sweep.

Usage:
    python aggregate_results.py --output-dir results/multi_seed_full
=====================================================================
"""
import argparse
import csv
import glob
import json
import os

from run_multi_seed_experiment import family_of


def load_all_results(output_dir):
    results = []
    for path in glob.glob(os.path.join(output_dir, "*", "result.json")):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    return results


def write_summary(output_dir, all_results):
    summary_csv = os.path.join(output_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "instance", "family", "keep", "K", "customers", "num_seeds",
            "pool_size_before_cap", "master_pool_cap", "pool_size",
            "best_single_lns_score",
            "ortools_seconds", "ortools_solved", "ortools_score",
            "exact_seconds", "exact_feasible", "exact_score",
            "exact_beats_best_single_seed", "improvement_over_best_single_seed",
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
                r["instance"], r.get("family", family_of(r["instance"])), r["keep"], r["K"],
                r["customers"], r["num_seeds"],
                r.get("pool_size_before_cap"), r.get("master_pool_cap"), r["pool_size"],
                r["best_single_lns_score"],
                ortools.get("seconds"), ortools.get("solved"), ortools.get("score"),
                exact.get("seconds"), exact.get("feasible"), exact.get("score"),
                exact.get("beats_best_single_seed"), exact.get("improvement_over_best_single_seed"),
                neal.get("seconds"), neal.get("valid_samples"), neal.get("num_reads"), neal.get("best_score"),
                swap.get("seconds"), swap.get("valid"), swap.get("restarts"), swap.get("best_score"),
                r.get("total_seconds"),
            ])
    return summary_csv


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def write_family_summary(output_dir, all_results):
    family_summary_csv = os.path.join(output_dir, "family_summary.csv")
    families = {}
    for r in all_results:
        fam = r.get("family", family_of(r["instance"]))
        families.setdefault(fam, []).append(r)

    with open(family_summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "family", "n_cases",
            "exact_feasible_rate", "mean_exact_score", "mean_exact_seconds",
            "n_exact_skipped",
            "n_exact_beats_single_seed", "mean_improvement_over_single_seed",
            "mean_neal_valid_rate", "mean_neal_best_score",
            "mean_swap_valid_rate", "mean_swap_best_score",
            "ortools_solved_rate", "mean_ortools_score",
            "n_beat_or_tied_ortools",
        ])
        for fam in sorted(families):
            rows = families[fam]
            n = len(rows)
            exact_rows = [r.get("exact", {}) for r in rows]
            n_skipped = sum(1 for e in exact_rows if "skipped" in e)
            attempted = [e for e in exact_rows if "skipped" not in e]
            exact_feas = [1.0 if e.get("feasible") else 0.0 for e in attempted]
            exact_scores = [e.get("score") for e in attempted]
            exact_seconds = [e.get("seconds") for e in attempted]
            neal_rates = [
                (r.get("neal", {}).get("valid_samples") or 0) / (r.get("neal", {}).get("num_reads") or 1)
                for r in rows if r.get("neal")
            ]
            neal_scores = [r.get("neal", {}).get("best_score") for r in rows]
            swap_rates = [
                (r.get("swap_annealer", {}).get("valid") or 0) / (r.get("swap_annealer", {}).get("restarts") or 1)
                for r in rows if r.get("swap_annealer")
            ]
            swap_scores = [r.get("swap_annealer", {}).get("best_score") for r in rows]
            ort_solved = [1.0 if r.get("ortools", {}).get("solved") else 0.0 for r in rows]
            ort_scores = [r.get("ortools", {}).get("score") for r in rows]
            n_beat_ortools = sum(
                1 for r in rows
                if r.get("exact", {}).get("score") is not None and r.get("ortools", {}).get("score") is not None
                and r["exact"]["score"] <= r["ortools"]["score"] + 1e-6
            )
            n_beats_single_seed = sum(1 for e in attempted if e.get("beats_best_single_seed"))
            improvements = [e.get("improvement_over_best_single_seed") for e in attempted]
            w.writerow([
                fam, n,
                _mean(exact_feas), _mean(exact_scores), _mean(exact_seconds),
                n_skipped,
                n_beats_single_seed, _mean(improvements),
                _mean(neal_rates), _mean(neal_scores),
                _mean(swap_rates), _mean(swap_scores),
                _mean(ort_solved), _mean(ort_scores),
                n_beat_ortools,
            ])
    return family_summary_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    all_results = load_all_results(args.output_dir)
    print(f"Found {len(all_results)} completed result.json file(s) under {args.output_dir}")
    if not all_results:
        return

    summary_csv = write_summary(args.output_dir, all_results)
    family_summary_csv = write_family_summary(args.output_dir, all_results)
    print(f"Wrote {summary_csv}")
    print(f"Wrote {family_summary_csv}")

    by_family = {}
    for r in all_results:
        by_family.setdefault(r.get("family", family_of(r["instance"])), 0)
        by_family[r.get("family", family_of(r["instance"]))] += 1
    print(f"Progress by family: {by_family}  (out of 12 R, 17 C, 16 RC instances x 2 keep levels)")


if __name__ == "__main__":
    main()
