"""n_grid sweep experiment for L1-RQAOA k=3 (MAX-3-CUT).

Metrics:
  - approx_ratio  = cut_edges / n_edges   (returned by solve())
  - rel_ratio     = approx_ratio / opt_ratio  where opt_ratio = opt_cut_edges / n_edges
                    (only when brute-force feasible, i.e. n <= BF_LIMIT)
  - theoretical maximum for MAX-3-CUT = 2/3 ≈ 0.667
"""
import json
import sys
import time
import warnings
from itertools import product as iproduct

import numpy as np
import networkx as nx

sys.path.insert(0, "/workspace/data/cuaoa")
from rqaoa_maxkcut.core.rqaoa import RQAOA1Solver

warnings.filterwarnings("ignore")

BF_LIMIT = 12  # k^n <= k^12 = 531441 feasible


def maxkcut_optimal_count(graph: nx.Graph, k: int) -> int | None:
    """Brute-force optimal MAX-k-CUT edge count (unweighted). None if too large."""
    nodes = list(graph.nodes())
    n = len(nodes)
    if k ** n > 10 ** 6:
        return None
    best = 0
    for assignment in iproduct(range(k), repeat=n):
        cut = sum(
            1 for u, v in graph.edges()
            if assignment[nodes.index(u)] != assignment[nodes.index(v)]
        )
        if cut > best:
            best = cut
    return best


def make_random_graph(n: int, density: float = 0.5, seed: int = 0) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                G.add_edge(i, j)  # unweighted — same as k=2 benchmark
    return G


N_VALUES   = [10, 20, 30]
N_GRIDS    = [10, 20, 50, 100, 200, 500]
N_GRAPHS   = 5
N_CUTOFF   = 6
K          = 3
THEORY_MAX = (K - 1) / K   # 2/3 ≈ 0.667

results = {}  # {n: {n_grid: [{"time_ms", "approx_ratio", "opt_ratio", "rel_ratio"}, ...]}}

for n in N_VALUES:
    results[n] = {}
    graphs = [make_random_graph(n, seed=s) for s in range(N_GRAPHS)]

    # brute-force optimal (only when feasible)
    opts = []
    for G in graphs:
        if n <= BF_LIMIT:
            opt_cnt = maxkcut_optimal_count(G, K)
            opts.append(opt_cnt / G.number_of_edges() if G.number_of_edges() > 0 else None)
        else:
            opts.append(None)

    for ng in N_GRIDS:
        trials = []
        print(f"n={n}  n_grid={ng}", flush=True)
        for gi, (G, opt_ratio) in enumerate(zip(graphs, opts)):
            solver = RQAOA1Solver(G, k=K, n_cutoff=N_CUTOFF, n_grid=ng, use_gpu=False)
            t0 = time.perf_counter()
            coloring, approx_ratio = solver.solve()
            elapsed = (time.perf_counter() - t0) * 1e3

            rel_ratio = approx_ratio / opt_ratio if opt_ratio else None
            opt_str = f"{opt_ratio:.4f}" if opt_ratio is not None else "N/A"
            rel_str = f"{rel_ratio:.4f}" if rel_ratio is not None else "N/A"
            print(f"  graph {gi+1}/{N_GRAPHS}  time={elapsed:.0f}ms  "
                  f"approx={approx_ratio:.4f}  opt={opt_str}  rel={rel_str}", flush=True)
            trials.append({
                "time_ms":     elapsed,
                "approx_ratio": approx_ratio,
                "opt_ratio":   opt_ratio,
                "rel_ratio":   rel_ratio,
            })

        results[n][ng] = trials

out_path = "/workspace/data/experiment_ngrid_k3_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {out_path}")

# Summary
print(f"\nTheoretical MAX-3-CUT upper bound = {THEORY_MAX:.4f}")
print("\n=== Summary (mean approx_ratio / mean time) ===")
print(f"{'n_grid':>7}", end="")
for n in N_VALUES:
    print(f"  n={n:2d} approx  rel_ratio  time(ms)", end="")
print()
for ng in N_GRIDS:
    print(f"{ng:>7}", end="")
    for n in N_VALUES:
        trials = results[n][ng]
        ma = np.mean([t["approx_ratio"] for t in trials])
        rels = [t["rel_ratio"] for t in trials if t["rel_ratio"] is not None]
        mr = np.mean(rels) if rels else float("nan")
        mt = np.mean([t["time_ms"] for t in trials])
        print(f"  {ma:.4f}  {mr:8.4f}  {mt:8.0f}", end="")
    print()
