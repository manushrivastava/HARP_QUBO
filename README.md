# HARP_QUBO_V3 — Multi-seed LNS pooling experiment (full Solomon dataset)

Clean, minimal codebase for the "why quantum" storyline's core empirical
claims (points 3 and 4 of the paper's argument):

1. Pooling routes from **multiple independent LNS runs** (different seeds)
   beats the best single LNS run.
2. That pooling makes the route-selection problem **too large for the
   classical exact solver**, while classical heuristics (`neal` bit-flip
   QUBO sampler, and a cardinality-preserving swap-move annealer) stay
   fast but leave a quality/reliability gap.

This folder intentionally contains **no previous experiment results** —
only the pipeline code needed to generate fresh ones, and the full Solomon
100-customer benchmark dataset (45 instances: C1×9, C2×8, R1×12, RC1×8,
RC2×8).

The original HARP-QUBO manuscript (`HARP_QUBO_ASOC.pdf`) evaluated only 58
cases: R1 (12) + C1 (9) + RC1 (8) = 29 instances × keep{70,100}, excluding
C2/RC2 entirely. **This experiment deliberately uses all 45 instances**
(90 cases at keep{70,100}) to extend beyond the manuscript's original
scope with more data points for the multi-seed pooling claims. Everything
else about the base per-instance recipe (construction stage size, LNS
iteration counts, guarded-repair, master-pool cap) matches the manuscript's
Algorithm 2 exactly, family-dependent caps included — see "What matches
the manuscript, and what's new" below.

## What matches the manuscript, and what's new

| Stage | Manuscript (Algorithm 2) | This experiment |
|---|---|---|
| Construction pool | 240 random partitions, retain ≤1200 | same |
| LNS starts | 90 lowest-score partitions | same (`--num-starts 90`) |
| LNS trials per start | 220 destroy/repair iterations | same (`--lns-iterations 220`) |
| LNS sets retained | 180 lowest-score valid sets | same (`--keep-lns-sets 180`) |
| Guarded repair | applied to the 12 lowest-score retained sets | same (`--guarded-repair-sets 12`), applied **per seed** |
| Master-pool cap | 1400 (R-family) / 600 (C/RC-family) | same, auto-detected from the instance name |
| **Seeding** | **one seed** (100, with two named exceptions at 200) | **multiple independent seeds** (`--num-seeds`, default 8) — this is the actual question this experiment tests: does seed diversity beat the manuscript's single-seed pool? |
| Dataset scope | 58 cases (R1+C1+RC1 only, excludes C2/RC2) | all 45 instances (90 cases at both keep levels) |
| Selector tuning | adaptive SPSA loop retuning `(A, B, s, M)` per case | **not replicated** — out of scope for testing quantum-readiness; every case uses one fixed selector configuration |

## Important: this pipeline is CPU-only, not GPU-accelerated

Every solver here (`neal`'s simulated annealing, the swap-move annealer,
the exact branch-and-bound solver) is pure Python/NumPy running on the
CPU. **Nothing in this codebase uses a GPU.** On an H200 server, only the
CPU core count and RAM matter for this experiment — set `--workers` in
`run_all_solomon.py` to roughly the physical core count to parallelize
across instances, not to use the GPU.

## Environment setup

```bash
# create and activate a fresh environment (conda shown; venv works identically)
conda create -n harp_qubo_v3 python=3.11 -y
conda activate harp_qubo_v3

# install dependencies
pip install -r requirements.txt
```

Or with plain `venv`:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke test (run this first, on one instance)

Before committing to the full 45-instance sweep, verify the pipeline runs
end-to-end on your server with a **small** configuration:

```bash
python run_multi_seed_experiment.py --instance r101 --keep 100 \
    --num-seeds 2 --lns-iterations 20 --num-starts 20 --guarded-repair-sets 3 \
    --output-dir results/smoke_test
```

This should finish in a few minutes and write
`results/smoke_test/r101_keep100/result.json`. Check it looks sane
before scaling up.

## Full-scale run (all 45 Solomon instances, both keep levels, multiple seeds)

```bash
python run_all_solomon.py \
    --keep 70,100 \
    --num-seeds 8 \
    --max-exact-n 700 \
    --workers 16 \
    --output-dir results/multi_seed_full
```

`--keep 70,100` and all 45 instances are the defaults, so the above is
equivalent to just `python run_all_solomon.py --workers 16` — 90 total
cases (45 instances × 2 keep levels).

Notes on the parameters:

- **`--num-starts 90`, `--lns-iterations 220`, `--keep-lns-sets 180`,
  `--guarded-repair-sets 12`**: match the manuscript's Algorithm 2 exactly
  (see the table above). Master-pool cap is auto-selected per instance
  (1400 for R-family, 600 for C/RC-family) unless you pass
  `--master-pool-cap` to override with one fixed number everywhere.
- **`--num-seeds 8`**: this project's own multi-seed campaigns showed
  clearly diminishing returns in *unique* routes added per additional
  seed (e.g. one 8-seed run added only 368 unique routes total, most
  overlapping with routes already found) — 8 is a reasonable default,
  not a magic number. Feel free to push higher if compute allows; expect
  the pool-size growth to keep slowing down, not stop. Unlike the
  manuscript (one seed), this multi-seed loop is this experiment's actual
  point: testing whether seed diversity beats a single-seed pool.
- **`--max-exact-n 700`**: on a genuinely LNS-competitive ~600-700 route
  pool, the exact solver took **5–10 hours** on a single instance in
  this project's own testing, growing steeply and unpredictably beyond
  that (branch-and-bound timing is instance-specific, not smooth). If
  the server has the patience for longer exact solves, raise this — but
  budget accordingly: with the master-pool cap now at 1400 (R-family),
  many cases will exceed 700 routes and skip exact solving by default.
  Raise `--max-exact-n` deliberately, and expect real multi-hour (or
  longer) solves for the cases that do run, especially R-family.
- **`--workers 16`** (or whatever fits the server): each worker handles
  one (instance, keep-level) job end-to-end; jobs are independent and
  embarrassingly parallel.

To run only a subset first (recommended before the full sweep):

```bash
python run_all_solomon.py --instances r101,r102,c101,rc101 --keep 70,100 \
    --num-seeds 8 --workers 4 --output-dir results/pilot_run
```

To reproduce the manuscript's original 58-case scope exactly instead of
the full 45-instance extension, restrict to R1+C1+RC1 (exclude c2* and
rc2*):

```bash
python run_all_solomon.py \
    --instances r101,r102,r103,r104,r105,r106,r107,r108,r109,r110,r111,r112,c101,c102,c103,c104,c105,c106,c107,c108,c109,rc101,rc102,rc103,rc104,rc105,rc106,rc107,rc108 \
    --keep 70,100 --num-seeds 8 --workers 16 --output-dir results/manuscript_58_case
```

## Output

Each (instance, keep-level) job writes:

- `results/<output-dir>/<instance>_keep<keep>/merged_pool.csv` — the
  final multi-seed-pooled route pool (checkpointed after every seed, so
  it survives interruption).
- `results/<output-dir>/<instance>_keep<keep>/result.json` — every
  number needed for the paper: best single-seed LNS score, pool size,
  exact solver time/score (or why it was skipped), neal's validity rate
  and best score, swap-annealer's validity rate and best score, and
  wall-clock timings for every stage.

`run_all_solomon.py` additionally aggregates every `result.json` into:

- `results/<output-dir>/summary.csv` — one row per (instance, keep-level)
  case, direct use in the paper's tables/figures.
- `results/<output-dir>/family_summary.csv` — one row per R/C/RC family,
  matching the manuscript's family-wise reporting style: mean exact/neal/
  swap-annealer validity and score, OR-Tools solve rate, and how many
  cases matched-or-beat OR-Tools.

Any job that raises an exception is caught, logged to `failures.json`,
and does not abort the rest of the sweep.

## File overview

| File | Role |
|---|---|
| `new_qubo_objectives.py` | Solomon file loading, instance reduction (`load_solomon_txt`, `reduce_instance`) |
| `decomposed_vrptw_qubo.py` | Route metrics/scoring primitives |
| `compare_with_ortools.py` | OR-Tools benchmark (used as a QUBO baseline elsewhere in the pipeline) |
| `route_qubo_refinement.py` | Sliding-window intra-route QUBO refinement (dependency of `route_pool_qubo_vrptw.py`) |
| `route_pool_qubo_vrptw.py` | Core: `build_route_pool`, `build_route_pool_qubo`, `solve_route_pool_qubo` (neal), `solve_route_pool_exact` (branch-and-bound) |
| `route_repair.py` | Classical local-search route repair (extracted from a much larger ML-experiment file this storyline doesn't need) |
| `probe_strong_route_pool_repair.py` | `lns_route_sets` — the multi-seed LNS destroy/repair engine |
| `swap_move_annealer.py` | Cardinality-preserving classical fix for the QUBO selector (this project's own contribution) |
| `run_multi_seed_experiment.py` | Single-instance driver tying everything together |
| `run_all_solomon.py` | Parallel sweep over the full Solomon dataset |
| `data/solomon-100/` | All 45 Solomon 100-customer benchmark instances |
