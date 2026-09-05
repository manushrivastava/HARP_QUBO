"""
swap_move_annealer.py
=====================================================================
Cardinality-preserving classical fix for the route-pool selection
problem: a Metropolis-Hastings annealer whose only move is a SWAP
(remove one currently-selected route, add one currently-unselected
route), which always keeps exactly K routes selected -- unlike the
bit-flip moves neal's QUBO sampler uses, which do not respect the
"select exactly K" constraint and were shown (this project) to
collapse in validity as the pool grows (dense N-choose-2 cardinality
penalty coupling).

Objective minimized: coverage_weight * coverage_error + score_sum/1000
  coverage_error = sum over customers of |times_covered - 1|
  (0 coverage_error <=> every customer covered by exactly one selected route)

Geometric cooling T0 -> T1 over n_steps per restart; best-of-n_restarts
independent restarts reported.
=====================================================================
"""
import math
import random

from route_pool_qubo_vrptw import route_key


def _route_customer_sets(pool):
    return [frozenset(route_key(r["route"])) for r in pool]


def _objective(selected, pool, pool_customers, all_ids, coverage_weight):
    counts = {c: 0 for c in all_ids}
    score_sum = 0.0
    for idx in selected:
        score_sum += pool[idx]["score"]
        for c in pool_customers[idx]:
            if c in counts:
                counts[c] += 1
    cov_err = sum(abs(v - 1) for v in counts.values())
    return coverage_weight * cov_err + score_sum / 1000.0, cov_err, score_sum


def swap_anneal_one_restart(pool, pool_customers, all_ids, K, rng,
                             n_steps=5000, T0=3.0, T1=0.005, coverage_weight=50.0):
    N = len(pool)
    selected = set(rng.sample(range(N), K))
    cur_obj, cur_cov, cur_score = _objective(selected, pool, pool_customers, all_ids, coverage_weight)
    best_obj, best_sel, best_cov, best_score = cur_obj, set(selected), cur_cov, cur_score
    for step in range(n_steps):
        T = T0 * (T1 / T0) ** (step / n_steps)
        out_idx = rng.choice(tuple(selected))
        in_idx = rng.randrange(N)
        if in_idx in selected:
            continue
        new_sel = set(selected)
        new_sel.discard(out_idx)
        new_sel.add(in_idx)
        new_obj, new_cov, new_score = _objective(new_sel, pool, pool_customers, all_ids, coverage_weight)
        delta = new_obj - cur_obj
        if delta <= 0 or rng.random() < math.exp(-delta / T):
            selected, cur_obj, cur_cov, cur_score = new_sel, new_obj, new_cov, new_score
            if cur_obj < best_obj:
                best_obj, best_sel, best_cov, best_score = cur_obj, set(selected), cur_cov, cur_score
    return best_sel, best_cov, best_score


def swap_anneal(inst, pool, n_restarts=20, n_steps=5000, T0=3.0, T1=0.005,
                 coverage_weight=50.0, seed=2024):
    """
    Run n_restarts independent swap-move annealer restarts on `pool`
    (list of {"route": [...], "score": float}) for instance `inst`.

    Returns a dict: {"n_valid": int, "n_restarts": int, "best_score": float|None,
                      "best_selected": list[int]|None}
    """
    K = inst["n_vehicles"]
    all_ids = list(inst["kept_ids"])
    pool_customers = _route_customer_sets(pool)
    rng = random.Random(seed)

    n_valid = 0
    best_score = None
    best_selected = None
    for _ in range(n_restarts):
        sel, cov, score = swap_anneal_one_restart(
            pool, pool_customers, all_ids, K, rng,
            n_steps=n_steps, T0=T0, T1=T1, coverage_weight=coverage_weight,
        )
        if cov == 0:
            n_valid += 1
            if best_score is None or score < best_score:
                best_score = score
                best_selected = sorted(sel)

    return {
        "n_valid": n_valid,
        "n_restarts": n_restarts,
        "best_score": best_score,
        "best_selected": best_selected,
    }
