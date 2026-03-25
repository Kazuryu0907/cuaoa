"""
benchmark_rqaoa_comparison.py
==============================
CUAOA RQAOA (statevector, k=2) vs Level-1 RQAOA (analytic, k=2/3) の比較ベンチマーク。

手法:
  cuaoa   : CUAOA statevector RQAOA (k=2 MaxCut, GPU L-BFGS optimizer, depth=1)
  L1-k=2  : Level-1 analytic RQAOA (k=2 MaxCut, CPU, simple γ grid search)
  L1-k=3  : Level-1 analytic RQAOA (k=3 MAX-3-CUT, CPU, analytic β opt)

グラフ: Erdős–Rényi G(n, 0.5) 非重み付き

スケーラビリティ:
  cuaoa  : n ≤ 20 が実用域 (2^n state vector; n=25+ は GPU メモリ逼迫)
  L1     : n=30 以上も可 (k^2 × k^2 = 9×9 行列のみ; メモリ O(1))
"""

import sys
import time
import json
from itertools import product as iproduct

import numpy as np
import networkx as nx

sys.path.insert(0, "/workspace/data/cuaoa")
from rqaoa_cuaoa import rqaoa_cuaoa, brute_force_maxcut
from rqaoa_maxkcut.core.rqaoa import RQAOA1Solver

# brute-force limit: O(k^n) operations
BF_LIMIT = {2: 20, 3: 12}


# =========================================================================
# Helpers
# =========================================================================

def make_graph(n: int, seed: int, p: float = 0.5) -> nx.Graph:
    g = nx.erdos_renyi_graph(n, p, seed=seed)
    if not nx.is_connected(g):
        g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
        g = nx.convert_node_labels_to_integers(g)
    return g


def maxkcut_brute(graph: nx.Graph, k: int):
    """Return optimal MAX-k-CUT value, or None if n > BF_LIMIT[k]."""
    n = graph.number_of_nodes()
    if n > BF_LIMIT.get(k, 8):
        return None
    nodes = sorted(graph.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in graph.edges()]
    best = 0
    for col in iproduct(range(k), repeat=n):
        cut = sum(1 for i, j in edges if col[i] != col[j])
        if cut > best:
            best = cut
    return best


def cut_value_from_coloring(graph: nx.Graph, coloring: np.ndarray) -> int:
    return sum(1 for u, v in graph.edges() if coloring[u] != coloring[v])


# =========================================================================
# Single-graph benchmark
# =========================================================================

def bench_cuaoa(graph: nx.Graph, n_cutoff: int, depth: int):
    W = nx.to_numpy_array(graph, dtype=np.float64)
    t0 = time.perf_counter()
    x, cut, _ = rqaoa_cuaoa(W, depth=depth, n_cutoff=n_cutoff, verbose=False)
    ms = (time.perf_counter() - t0) * 1e3
    # Brute-force optimal (k=2)
    opt = maxkcut_brute(graph, k=2)
    if opt is None:
        # Fall back to cuaoa's own brute force result
        _, opt = brute_force_maxcut(W) if graph.number_of_nodes() <= 20 else (None, graph.number_of_edges())
    ratio = cut / opt if opt > 0 else float("nan")
    return {"time_ms": ms, "cut": float(cut), "opt": float(opt), "ratio": ratio}


def bench_l1(graph: nx.Graph, k: int, n_cutoff: int, use_gpu: bool = False):
    t0 = time.perf_counter()
    solver = RQAOA1Solver(graph, k=k, n_cutoff=n_cutoff,
                          use_gpu=use_gpu, verbose=False)
    coloring, _ = solver.solve()
    ms = (time.perf_counter() - t0) * 1e3
    cut = cut_value_from_coloring(graph, coloring)
    opt = maxkcut_brute(graph, k=k)
    denom = opt if (opt is not None and opt > 0) else graph.number_of_edges()
    ratio = cut / denom if denom > 0 else float("nan")
    return {"time_ms": ms, "cut": cut, "opt": opt, "ratio": ratio}


# =========================================================================
# Main
# =========================================================================

def run(n_list_cuaoa, n_list_l1, n_graphs, n_cutoff, depth, seed_base):
    results = []

    # -- cuaoa + L1 for sizes where cuaoa is feasible --
    print("\n" + "="*72)
    print(f"Phase 1: cuaoa + L1 (n ∈ {n_list_cuaoa}, {n_graphs} graphs each)")
    print("="*72)

    for n in n_list_cuaoa:
        rows_n = []
        for gi in range(n_graphs):
            seed = seed_base + n * 1000 + gi
            g = make_graph(n, seed)
            nn = g.number_of_nodes()
            m = g.number_of_edges()
            print(f"  n={nn} m={m} graph={gi+1}/{n_graphs}", flush=True)

            cutoff = min(n_cutoff, nn - 2)

            # cuaoa
            try:
                r_c = bench_cuaoa(g, cutoff, depth)
                print(f"    cuaoa  : {r_c['time_ms']:6.0f} ms  cut={r_c['cut']:.0f}/{r_c['opt']:.0f}  ratio={r_c['ratio']:.4f}", flush=True)
            except Exception as e:
                r_c = {"time_ms": None, "cut": None, "opt": None, "ratio": None, "error": str(e)}
                print(f"    cuaoa  : ERROR {e}", flush=True)

            # L1 k=2
            try:
                r_k2 = bench_l1(g, k=2, n_cutoff=cutoff)
                print(f"    L1-k=2 : {r_k2['time_ms']:6.0f} ms  cut={r_k2['cut']}/{r_k2['opt']}  ratio={r_k2['ratio']:.4f}", flush=True)
            except Exception as e:
                r_k2 = {"time_ms": None, "cut": None, "opt": None, "ratio": None, "error": str(e)}
                print(f"    L1-k=2 : ERROR {e}", flush=True)

            # L1 k=3
            try:
                r_k3 = bench_l1(g, k=3, n_cutoff=cutoff)
                opt_str = str(r_k3["opt"]) if r_k3["opt"] is not None else f"≤{m}*"
                print(f"    L1-k=3 : {r_k3['time_ms']:6.0f} ms  cut={r_k3['cut']}/{opt_str}  ratio={r_k3['ratio']:.4f}", flush=True)
            except Exception as e:
                r_k3 = {"time_ms": None, "cut": None, "opt": None, "ratio": None, "error": str(e)}
                print(f"    L1-k=3 : ERROR {e}", flush=True)

            rows_n.append({"n": nn, "m": m, "seed": seed,
                           "cuaoa": r_c, "l1k2": r_k2, "l1k3": r_k3})

        results.extend(rows_n)

    # -- L1 only for larger n --
    n_list_l1_only = [n for n in n_list_l1 if n not in n_list_cuaoa]
    if n_list_l1_only:
        print("\n" + "="*72)
        print(f"Phase 2: L1 only (n ∈ {n_list_l1_only}, {n_graphs} graphs each, cuaoa skipped: 2^n too large)")
        print("="*72)

        for n in n_list_l1_only:
            for gi in range(n_graphs):
                seed = seed_base + n * 1000 + gi
                g = make_graph(n, seed)
                nn = g.number_of_nodes()
                m = g.number_of_edges()
                print(f"  n={nn} m={m} graph={gi+1}/{n_graphs}", flush=True)

                cutoff = min(n_cutoff, nn - 2)

                try:
                    r_k2 = bench_l1(g, k=2, n_cutoff=cutoff)
                    print(f"    L1-k=2 : {r_k2['time_ms']:6.0f} ms  cut={r_k2['cut']}  ratio={r_k2['ratio']:.4f}", flush=True)
                except Exception as e:
                    r_k2 = {"time_ms": None, "cut": None, "opt": None, "ratio": None, "error": str(e)}
                    print(f"    L1-k=2 : ERROR {e}", flush=True)

                try:
                    r_k3 = bench_l1(g, k=3, n_cutoff=cutoff)
                    opt_str = str(r_k3["opt"]) if r_k3["opt"] is not None else f"≤{m}*"
                    print(f"    L1-k=3 : {r_k3['time_ms']:6.0f} ms  cut={r_k3['cut']}/{opt_str}  ratio={r_k3['ratio']:.4f}", flush=True)
                except Exception as e:
                    r_k3 = {"time_ms": None, "cut": None, "opt": None, "ratio": None, "error": str(e)}
                    print(f"    L1-k=3 : ERROR {e}", flush=True)

                results.append({"n": nn, "m": m, "seed": seed,
                                "cuaoa": None, "l1k2": r_k2, "l1k3": r_k3})

    return results


def summarize(results):
    from collections import defaultdict
    import statistics

    by_n = defaultdict(list)
    for r in results:
        by_n[r["n"]].append(r)

    def agg(rows, method, field):
        vals = [r[method][field] for r in rows
                if r.get(method) and r[method].get(field) is not None]
        if not vals:
            return None, None
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0)

    print("\n" + "="*90)
    print("SUMMARY  (mean over graphs per n)")
    print("="*90)
    print(f"{'n':>4} {'m':>5} | {'cuaoa (ms)':>11} {'ratio':>6} |"
          f" {'L1-k2 (ms)':>11} {'ratio':>6} |"
          f" {'L1-k3 (ms)':>11} {'ratio':>7}")
    print("-"*90)

    for n in sorted(by_n):
        rows = by_n[n]
        m_mean = statistics.mean(r["m"] for r in rows)

        def fmt(val, std, is_ratio=False):
            if val is None:
                return "       —"
            if is_ratio:
                return f"  {val:.4f}"
            return f"  {val:7.0f}"

        ct, cs = agg(rows, "cuaoa", "time_ms")
        cr, _ = agg(rows, "cuaoa", "ratio")
        k2t, _ = agg(rows, "l1k2", "time_ms")
        k2r, _ = agg(rows, "l1k2", "ratio")
        k3t, _ = agg(rows, "l1k3", "time_ms")
        k3r, _ = agg(rows, "l1k3", "ratio")

        print(f"{n:>4} {m_mean:>5.0f} |"
              f"{fmt(ct,cs):>12} {fmt(cr,None,True):>7} |"
              f"{fmt(k2t,None):>12} {fmt(k2r,None,True):>7} |"
              f"{fmt(k3t,None):>12} {fmt(k3r,None,True):>8}")

    print()
    print("Legend:")
    print("  cuaoa  = CUAOA GPU statevector RQAOA   k=2 MaxCut     depth=1  (GPU: custatevec)")
    print("  L1-k=2 = Level-1 analytic RQAOA        k=2 MaxCut     CPU-only  (simple γ grid)")
    print("  L1-k=3 = Level-1 analytic RQAOA        k=3 MAX-3-CUT  CPU-only  (analytic β opt)")
    print("  ratio  = cut / opt_cut  (* = opt estimated as |E|, brute-force limit exceeded)")
    print()
    print("Key insight:")
    print("  cuaoa  needs 2^n state vector → GPU memory hard limit at n≈27")
    print("  L1-k=3 needs only k²×k² = 9×9 matrices → scales to n=300+ (GPU batch would be fast)")


# =========================================================================
# Entry point
# =========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cuaoa", nargs="+", type=int, default=[10, 12, 15, 18, 20],
                        help="graph sizes for cuaoa+L1 comparison")
    parser.add_argument("--n-l1", nargs="+", type=int, default=[25, 30],
                        help="additional sizes for L1 only (no cuaoa)")
    parser.add_argument("--n-graphs", type=int, default=3)
    parser.add_argument("--n-cutoff", type=int, default=5)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="benchmark_rqaoa_comparison.json")
    args = parser.parse_args()

    t_total = time.perf_counter()
    results = run(
        n_list_cuaoa=args.n_cuaoa,
        n_list_l1=args.n_cuaoa + args.n_l1,
        n_graphs=args.n_graphs,
        n_cutoff=args.n_cutoff,
        depth=args.depth,
        seed_base=args.seed,
    )
    summarize(results)
    print(f"\nTotal wall time: {(time.perf_counter()-t_total):.1f} s")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {args.out}")
