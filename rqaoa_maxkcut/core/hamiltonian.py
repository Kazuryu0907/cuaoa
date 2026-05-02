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

        For a vanilla (unweighted) edge (u,v):
            J_{uv}(b) = 1 - δ_{b,0}
            h_{uv}(0) = (k-1)/k,  h_{uv}(a≠0) = -1/k
            ĥ_{uv}(0) = 0,        ĥ_{uv}(b≠0) = 1

        For RQAOA-contracted edges the graph carries an ``h_hat``
        edge attribute (a complex vector of length k) holding the
        accumulated Fourier coefficients. We read those when present
        and recover h via inverse DFT.
        """
        k = self.k
        # Default unweighted ĥ vector (ĥ(0)=0, ĥ(b≠0)=1)
        h_hat_default = np.zeros(k, dtype=np.complex128)
        h_hat_default[1:] = 1.0

        a_arr = np.arange(k)
        # Inverse DFT matrix: h(a) = (1/k) Σ_b ĥ(b) ω^{-ab}
        # Row a, column b → ω^{-ab} / k
        ab = a_arr[:, None] * a_arr[None, :]
        idft = self.omega ** (-ab) / k  # (k, k)

        for u, v, data in self.graph.edges(data=True):
            key = self._canonical(u, v)
            if "h_hat" in data:
                h_hat_edge = np.asarray(data["h_hat"], dtype=np.complex128)
            else:
                h_hat_edge = h_hat_default.copy()
            # h(a) = (1/k) Σ_b ĥ(b) ω^{-ab}
            h_edge = idft @ h_hat_edge   # (k,)
            self._h[key] = h_edge
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
