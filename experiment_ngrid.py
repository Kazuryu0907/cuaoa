"""
experiment_ngrid.py
===================
L1 RQAOA (k=2) の γ グリッドサーチ点数 n_grid を変えて
近似比と処理時間がどう変わるか実験。

条件:
  - Erdős–Rényi G(n, 0.5)、各 n・各 n_grid につき N_GRAPHS グラフ
  - n = 10, 20, 30
  - n_grid = 10, 20, 50, 100, 200, 500
  - k=2 (MaxCut), n_cutoff=5
"""

import sys
import time
import json
import statistics
from itertools import product as iproduct

import numpy as np
import networkx as nx

sys.path.insert(0, "/workspace/data/cuaoa")
from rqaoa_maxkcut.core.rqaoa import RQAOA1Solver

# ブルートフォース上限
BF_LIMIT = 20
N_GRAPHS = 5
N_CUTOFF = 5
SEED_BASE = 42
N_GRID_LIST = [10, 20, 50, 100, 200, 500]
N_LIST = [10, 20, 30]


# ---------------------------------------------------------------------------

def make_graph(n: int, seed: int) -> nx.Graph:
    g = nx.erdos_renyi_graph(n, 0.5, seed=seed)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        g = nx.convert_node_labels_to_integers(g)
    return g


def brute_maxcut(g: nx.Graph):
    if g.number_of_nodes() > BF_LIMIT:
        return None
    nodes = sorted(g.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    best = 0
    for bits in iproduct([0, 1], repeat=len(nodes)):
        cut = sum(1 for i, j in edges if bits[i] != bits[j])
        if cut > best:
            best = cut
    return best


def cut_from_coloring(g: nx.Graph, col: np.ndarray) -> int:
    return sum(1 for u, v in g.edges() if col[u] != col[v])


# ---------------------------------------------------------------------------

def run_one(g: nx.Graph, n_grid: int) -> dict:
    n_nodes = g.number_of_nodes()
    cutoff = min(N_CUTOFF, n_nodes - 2)
    t0 = time.perf_counter()
    solver = RQAOA1Solver(g, k=2, n_cutoff=cutoff, use_gpu=False,
                          verbose=False, n_grid=n_grid)
    col, _ = solver.solve()
    ms = (time.perf_counter() - t0) * 1e3
    cut = cut_from_coloring(g, col)
    opt = brute_maxcut(g)
    denom = opt if (opt and opt > 0) else g.number_of_edges()
    ratio = cut / denom if denom > 0 else float("nan")
    return {"time_ms": ms, "cut": cut, "opt": opt, "ratio": ratio}


# ---------------------------------------------------------------------------

def main():
    # n ごとにグラフを固定（n_grid 間で同じグラフを使う）
    graphs = {
        n: [make_graph(n, SEED_BASE + n * 100 + gi) for gi in range(N_GRAPHS)]
        for n in N_LIST
    }

    # 全結果を収集
    data = {}   # data[n][n_grid] = list of results

    for n in N_LIST:
        data[n] = {}
        for n_grid in N_GRID_LIST:
            results = []
            for gi, g in enumerate(graphs[n]):
                r = run_one(g, n_grid)
                results.append(r)
                print(f"  n={n:2d} n_grid={n_grid:4d} graph={gi+1}/{N_GRAPHS}"
                      f"  {r['time_ms']:7.0f}ms  ratio={r['ratio']:.4f}", flush=True)
            data[n][n_grid] = results

    # ---- サマリ ----
    print("\n" + "=" * 90)
    print("SUMMARY  (mean ± std over graphs)")
    print("=" * 90)

    for n in N_LIST:
        print(f"\n--- n={n} ---")
        opt_known = data[n][N_GRID_LIST[0]][0]["opt"] is not None
        denom_label = "opt" if opt_known else "|E|"
        print(f"{'n_grid':>8} | {'time_ms (mean)':>15} {'time_ms (std)':>13} | "
              f"{'ratio/{} (mean)'.format(denom_label):>18} {'ratio (std)':>11}")
        print("-" * 75)
        for n_grid in N_GRID_LIST:
            times = [r["time_ms"] for r in data[n][n_grid]]
            ratios = [r["ratio"] for r in data[n][n_grid]]
            t_mean = statistics.mean(times)
            t_std = statistics.stdev(times) if len(times) > 1 else 0.0
            r_mean = statistics.mean(ratios)
            r_std = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
            print(f"{n_grid:>8} | {t_mean:>13.0f}ms {t_std:>11.0f}ms | "
                  f"{r_mean:>16.4f}     {r_std:>9.4f}")

    # JSON 保存
    out = {}
    for n in N_LIST:
        out[str(n)] = {}
        for n_grid in N_GRID_LIST:
            out[str(n)][str(n_grid)] = data[n][n_grid]

    with open("/workspace/data/experiment_ngrid_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSaved to experiment_ngrid_results.json")


if __name__ == "__main__":
    main()
