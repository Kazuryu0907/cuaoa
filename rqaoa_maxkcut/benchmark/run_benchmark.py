"""
benchmark/run_benchmark.py
==========================
RQAOA-1 vs Newman MAX-k-CUT comparison experiment.

Reproduces (a subset of) Section 5 of Bravyi et al. 2022:
for each (n, d) pair generate n_graphs random G[d,n] graphs and compare:
  - QAOA-1 expected value (no rounding; energy as fraction of |E|)
  - RQAOA-1 coloring
  - Newman SDP + rounding

Results are returned as a list of dicts (or a pandas DataFrame if pandas
is available).
"""

from __future__ import annotations

import time
import numpy as np
import networkx as nx
from typing import List, Dict, Optional

from .graph_gen import generate_3colorable_regular_graph
from ..core.hamiltonian import MaxKCutHamiltonian
from ..core.density_matrix import DensityMatrixSimulator
from ..core.angle_opt import AngleOptimizer
from ..core.rqaoa import RQAOA1Solver
from ..classical.newman import NewmanMaxKCut

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    _PANDAS_AVAILABLE = False


def compute_approx_ratio(
    graph: nx.Graph,
    coloring: np.ndarray,
    k: int,
) -> float:
    """
    Compute C(x) / C_max where C_max = |E| for 3-colourable graphs.

    Parameters
    ----------
    graph    : nx.Graph
    coloring : np.ndarray, shape (n,), values in Z_k
    k        : int

    Returns
    -------
    ratio : float ∈ [0, 1]
    """
    n_edges = graph.number_of_edges()
    if n_edges == 0:
        return 1.0
    cut = sum(1 for u, v in graph.edges() if coloring[u] != coloring[v])
    return cut / n_edges


def run_qaoa1_energy(
    graph: nx.Graph,
    k: int = 3,
    use_gpu: bool = False,
) -> float:
    """
    Compute the QAOA-1 expected energy (as fraction of |E|) for a graph.

    Uses the angle optimiser to find the best (γ*, β*) and returns the
    expected energy E(γ*, β*) / |E|.

    Parameters
    ----------
    graph   : nx.Graph
    k       : int
    use_gpu : bool

    Returns
    -------
    energy_ratio : float
    """
    ham = MaxKCutHamiltonian(graph, k)
    sim = DensityMatrixSimulator(ham, use_gpu=use_gpu)
    if k == 3:
        opt = AngleOptimizer(sim)
        _, _, E_opt = opt.optimize_gamma(graph)
        n_edges = graph.number_of_edges()
        return E_opt / n_edges if n_edges > 0 else 0.0
    else:
        return 0.0


def run_comparison(
    n_list: List[int] = [30, 60],
    d_list: List[int] = [4, 6],
    n_graphs: int = 5,
    n_newman_samples: int = 100,
    k: int = 3,
    n_cutoff: int = 6,
    use_gpu: bool = False,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> "pd.DataFrame | List[Dict]":
    """
    Run RQAOA-1 vs Newman comparison on G[d, n] random graphs.

    Parameters
    ----------
    n_list           : list of n values
    d_list           : list of d values
    n_graphs         : number of graphs per (n, d) pair
    n_newman_samples : random rounding trials for Newman
    k                : number of colours (default 3)
    n_cutoff         : RQAOA brute-force cutoff
    use_gpu          : use CuPy for RQAOA
    seed             : random seed base
    verbose          : print progress

    Returns
    -------
    DataFrame (or list of dicts if pandas unavailable) with columns:
        n, d, graph_id, algorithm, approx_ratio, time_s
    """
    records: List[Dict] = []
    rng_base = np.random.default_rng(seed)

    for n in n_list:
        for d in d_list:
            if verbose:
                print(f"\n=== n={n}, d={d} ===")

            for graph_id in range(n_graphs):
                graph_seed = int(rng_base.integers(0, 2 ** 31))
                try:
                    G = generate_3colorable_regular_graph(n, d, seed=graph_seed)
                except ValueError as e:
                    if verbose:
                        print(f"  graph_id={graph_id}: generation failed: {e}")
                    continue

                if verbose:
                    print(f"  graph_id={graph_id}  |V|={G.number_of_nodes()}"
                          f"  |E|={G.number_of_edges()}")

                # ---- QAOA-1 expected energy ----
                t0 = time.perf_counter()
                try:
                    qaoa_ratio = run_qaoa1_energy(G, k=k, use_gpu=use_gpu)
                except Exception as exc:
                    if verbose:
                        print(f"    QAOA-1 failed: {exc}")
                    qaoa_ratio = float("nan")
                t_qaoa = time.perf_counter() - t0

                records.append({
                    "n": n, "d": d, "graph_id": graph_id,
                    "algorithm": "QAOA1",
                    "approx_ratio": qaoa_ratio,
                    "time_s": t_qaoa,
                })
                if verbose:
                    print(f"    QAOA-1   ratio={qaoa_ratio:.4f}  t={t_qaoa:.2f}s")

                # ---- RQAOA-1 ----
                t0 = time.perf_counter()
                try:
                    solver = RQAOA1Solver(G, k=k, n_cutoff=n_cutoff,
                                          use_gpu=use_gpu)
                    coloring, rqaoa_ratio = solver.solve()
                except Exception as exc:
                    if verbose:
                        print(f"    RQAOA-1 failed: {exc}")
                    rqaoa_ratio = float("nan")
                t_rqaoa = time.perf_counter() - t0

                records.append({
                    "n": n, "d": d, "graph_id": graph_id,
                    "algorithm": "RQAOA1",
                    "approx_ratio": rqaoa_ratio,
                    "time_s": t_rqaoa,
                })
                if verbose:
                    print(f"    RQAOA-1  ratio={rqaoa_ratio:.4f}  t={t_rqaoa:.2f}s")

                # ---- Newman ----
                t0 = time.perf_counter()
                newman_rng = np.random.default_rng(graph_seed + 1)
                try:
                    newman = NewmanMaxKCut(G, k=k, n_samples=n_newman_samples,
                                          rng=newman_rng)
                    _, newman_ratio = newman.solve()
                except Exception as exc:
                    if verbose:
                        print(f"    Newman  failed: {exc}")
                    newman_ratio = float("nan")
                t_newman = time.perf_counter() - t0

                records.append({
                    "n": n, "d": d, "graph_id": graph_id,
                    "algorithm": "Newman",
                    "approx_ratio": newman_ratio,
                    "time_s": t_newman,
                })
                if verbose:
                    print(f"    Newman   ratio={newman_ratio:.4f}  t={t_newman:.2f}s")

    if _PANDAS_AVAILABLE:
        return pd.DataFrame(records)
    return records
