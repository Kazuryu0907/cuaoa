"""
core/hamiltonian.py
===================
MAX-k-CUT cost-function Hamiltonian.

Mathematical summary
--------------------
For a graph G=(V,E) with k colours the cost function is

    C(x) = Σ_{(i,j)∈E} (1 - δ_{x_i, x_j}),   x ∈ Z_k^n

In Fourier form (equation 24 of Bravyi et al. 2022):

    C = Σ_{u<v} C_{uv},   C_{uv} = Σ_{a∈Z_k} h_{uv}(a) Z_u^a Z_v^{-a}

where

    h_{uv}(a) = (1/k) Σ_{b∈Z_k} J_{uv}(b) ω^{ab},   ω = exp(2πi/k)

For an edge (u,v) ∈ E:  J_{uv}(b) = 1 - δ_{b,0}
=> h_{uv}(0) = (k-1)/k,  h_{uv}(a≠0) = -1/k

The inverse Fourier transform gives ĥ_{uv}(b):

    ĥ_{uv}(b) = Σ_{a∈Z_k} h_{uv}(a) ω^{ab}

For an edge (u,v) ∈ E:  ĥ_{uv}(0) = 0,  ĥ_{uv}(b≠0) = 1
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, Tuple


class MaxKCutHamiltonian:
    """
    MAX-k-CUT cost-function Hamiltonian representation.

    Builds h_{uv}(a) and ĥ_{uv}(b) for every edge in the graph.

    Parameters
    ----------
    graph : nx.Graph
        Undirected unweighted graph.
    k : int
        Number of colours (qudit dimension).
    """

    def __init__(self, graph: nx.Graph, k: int) -> None:
        self.graph = graph
        self.k = k
        self.n = graph.number_of_nodes()
        self.omega = np.exp(2j * np.pi / k)

        # Caches keyed by canonical edge (min(u,v), max(u,v))
        self._h: Dict[Tuple[int, int], np.ndarray] = {}
        self._h_hat: Dict[Tuple[int, int], np.ndarray] = {}

        # Zero arrays returned for non-edges
        self._zero = np.zeros(k, dtype=np.complex128)

        self._build_h()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _canonical(self, u: int, v: int) -> Tuple[int, int]:
        return (min(u, v), max(u, v))

    def _build_h(self) -> None:
        """
        Compute h_{uv}(a) and ĥ_{uv}(b) for every edge.

        For an edge (u,v):
            J_{uv}(b) = 1 - δ_{b,0}
            h_{uv}(0) = (1/k)(k-1) = (k-1)/k
            h_{uv}(a≠0) = (1/k)(0 - 1) = -1/k

        ĥ_{uv} is the DFT of h_{uv}:
            ĥ_{uv}(b) = Σ_a h_{uv}(a) ω^{ab}
            ĥ_{uv}(0) = Σ_a h_{uv}(a) = (k-1)/k - (k-1)/k = 0
            ĥ_{uv}(b≠0) = 1  (can be verified by direct substitution)
        """
        k = self.k
        # Pre-compute h and h_hat for a single edge (same for all edges)
        h_edge = np.empty(k, dtype=np.complex128)
        h_edge[0] = (k - 1) / k
        h_edge[1:] = -1.0 / k

        # DFT: ĥ(b) = Σ_a h(a) ω^{ab}
        a = np.arange(k)
        h_hat_edge = np.array(
            [np.sum(h_edge * self.omega ** (a * b)) for b in range(k)],
            dtype=np.complex128,
        )
        # ĥ(0) should be exactly 0, ĥ(b≠0) should be exactly 1; enforce numerically
        # (minor floating-point cleanup)
        h_hat_edge = h_hat_edge.real.copy().astype(np.complex128)
        # Convert to real since values are real for unweighted graphs
        h_edge_real = h_edge.real.copy()

        for u, v in self.graph.edges():
            key = self._canonical(u, v)
            self._h[key] = h_edge_real.astype(np.complex128)
            self._h_hat[key] = h_hat_edge

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_h(self, u: int, v: int) -> np.ndarray:
        """
        Return h_{uv}(a) as shape (k,) complex array.
        Returns zeros if (u,v) is not an edge.
        """
        key = self._canonical(u, v)
        return self._h.get(key, self._zero).copy()

    def get_h_hat(self, u: int, v: int) -> np.ndarray:
        """
        Return ĥ_{uv}(b) as shape (k,) complex array.
        Returns zeros if (u,v) is not an edge.
        """
        key = self._canonical(u, v)
        return self._h_hat.get(key, self._zero).copy()

    def has_edge(self, u: int, v: int) -> bool:
        return self.graph.has_edge(u, v)

    def edges(self):
        """Iterate over edges as (u, v) tuples."""
        return self.graph.edges()

    def max_cut_value(self, coloring: np.ndarray) -> float:
        """
        Compute C(x) = number of properly coloured edges.

        Parameters
        ----------
        coloring : np.ndarray, shape (n,), dtype int
            x ∈ Z_k^n

        Returns
        -------
        float  The number of cut edges.
        """
        val = 0.0
        for u, v in self.graph.edges():
            if coloring[u] != coloring[v]:
                val += 1.0
        return val
