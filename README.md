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
    --num-seeds 2 --lns-iterations 20 --num-starts 20 \
    --output-dir results/smoke_test
```

This should finish in a few minutes and write
`results/smoke_test/r101_keep100/result.json`. Check it looks sane
before scaling up.

## Full-scale run (all 45 Solomon instances, multiple seeds)

```bash
python run_all_solomon.py \
    --keep 100 \
    --num-seeds 8 \
    --lns-iterations 120 \
    --max-exact-n 700 \
    --workers 16 \
    --output-dir results/multi_seed_full
```

Notes on the parameters, from what was actually measured on this project
(smaller-scale laptop testing) before this run was handed to the server:

- **`--num-seeds 8`**: this project's own multi-seed campaigns showed
  clearly diminishing returns in *unique* routes added per additional
  seed (e.g. one 8-seed run added only 368 unique routes total, most
  overlapping with routes already found) — 8 is a reasonable default,
  not a magic number. Feel free to push higher if compute allows; expect
  the pool-size growth to keep slowing down, not stop.
- **`--lns-iterations 120`** and **`--num-starts 150`**: match the
  configuration that produced this project's best-documented results.
- **`--max-exact-n 700`**: on a genuinely LNS-competitive ~600-700 route
  pool, the exact solver took **5–10 hours** on a single instance in
  this project's own testing, growing steeply and unpredictably beyond
  that (branch-and-bound timing is instance-specific, not smooth). If
  the server has the patience for longer exact solves, raise this — but
  budget accordingly: a full 45-instance x keep-level sweep with exact
  solving enabled at N=700 could mean many CPU-days if run serially, or
  substantial parallel wall-clock time even with many workers, since
  each individual exact solve is itself single-threaded.
- **`--workers 16`** (or whatever fits the server): each worker handles
  one (instance, keep-level) job end-to-end; jobs are independent and
  embarrassingly parallel.

To run only a subset first (recommended before the full sweep):

```bash
python run_all_solomon.py --instances r101,r102,c101,rc101 --keep 100 \
    --num-seeds 8 --workers 4 --output-dir results/pilot_run
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

`run_all_solomon.py` additionally aggregates every `result.json` into
one combined `results/<output-dir>/summary.csv` for direct use in the
paper's tables/figures. Any job that raises an exception is caught,
logged to `failures.json`, and does not abort the rest of the sweep.

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
