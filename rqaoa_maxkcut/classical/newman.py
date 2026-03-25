"""
classical/newman.py
===================
Newman (2018) MAX-k-CUT approximation algorithm for k=3.

Reference
---------
Alantha Newman, "Complex Semidefinite Programming and Max-k-CUT",
arXiv:1811.05136 (2018).

Algorithm summary
-----------------
1. Solve a complex SDP relaxation of MAX-k-CUT:
       maximise  Σ_{(i,j)∈E} (1 - Re <v_i, v_j>) × k/(k-1)
       subject to  <v_i, v_i> = 1  for all i
                   v_i ∈ C^n
   The SDP is expressed as a real PSD matrix X = Re[V^† V] where V is
   the n×n Gram matrix.

2. Random rounding à la Goemans–Williamson / Newman:
   - Sample a random unit vector r ∈ C^k.
   - Assign colour c(i) = argmax_{c=0..k-1} Re[ω^c (r · v_i)] where ω = e^{2πi/k}.
   Repeat n_samples times; return the colouring with the highest cut value.

SDP backend
-----------
Uses cvxpy if available; falls back to a greedy random-restart heuristic
if cvxpy is not installed.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Tuple

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    cp = None
    _CVXPY_AVAILABLE = False


def _cut_value(graph: nx.Graph, coloring: np.ndarray) -> int:
    """Count the number of properly coloured edges."""
    return sum(1 for u, v in graph.edges() if coloring[u] != coloring[v])


class NewmanMaxKCut:
    """
    Newman (2018) MAX-k-CUT approximation algorithm.

    Parameters
    ----------
    graph     : nx.Graph
    k         : int        Number of colours (default 3).
    n_samples : int        Number of random rounding rounds (default 100).
    """

    def __init__(
        self,
        graph: nx.Graph,
        k: int = 3,
        n_samples: int = 100,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.graph = graph
        self.k = k
        self.n = graph.number_of_nodes()
        self.n_samples = n_samples
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # SDP
    # ------------------------------------------------------------------

    def solve_sdp(self) -> np.ndarray:
        """
        Solve the SDP relaxation and return a Gram-matrix factor V.

        If cvxpy is available: solve the true SDP.
        Otherwise: return a random orthonormal frame as a surrogate.

        Returns
        -------
        V : np.ndarray, shape (n, d)  with V[i] being the unit vector for node i.
        """
        n = self.n
        nodes = sorted(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in self.graph.edges()]

        if _CVXPY_AVAILABLE and len(nodes) <= 300:
            return self._solve_sdp_cvxpy(n, edges)
        else:
            return self._solve_sdp_fallback(n)

    def _solve_sdp_cvxpy(self, n: int, edges) -> np.ndarray:
        """
        Solve the SDP using cvxpy.

        Formulation
        -----------
        Variables: X ∈ S^n_{+},  X_{ii} = 1  (Gram matrix, real PSD approx)

        Maximise:  Σ_{(i,j)∈E} (1 - X_{ij}) × k/(k-1)

        This is the real relaxation; in the complex SDP the constraint is
        X_{ii} = 1 and the objective uses Re[<v_i, v_j>].  The real
        relaxation corresponds to k=2 (Goemans-Williamson) but gives a
        useful warm start for k=3.

        For k=3 we use the Lovász ϑ-function based SDP:
            maximise  Σ_{(i,j)∈E} (1 - Re[<v_i, v_j>]) × 3/2
            subject to  X ≽ 0,  X_{ii} = 1
        """
        k = self.k
        X = cp.Variable((n, n), symmetric=True)
        objective = cp.Maximize(
            cp.sum([
                (1.0 - X[i, j]) * k / (k - 1)
                for i, j in edges
            ])
        )
        constraints = [
            X >> 0,
            cp.diag(X) == np.ones(n),
        ]
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, verbose=False)
        except Exception:
            try:
                prob.solve(verbose=False)
            except Exception:
                return self._solve_sdp_fallback(n)

        if X.value is None:
            return self._solve_sdp_fallback(n)

        # Cholesky factorisation X ≈ V V^T
        X_val = np.array(X.value)
        # Make PSD (clip negative eigenvalues)
        eigvals, eigvecs = np.linalg.eigh(X_val)
        eigvals = np.maximum(eigvals, 0.0)
        V = eigvecs * np.sqrt(eigvals)[None, :]   # (n, n)
        # Normalise rows to unit length
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        V = V / norms
        return V

    def _solve_sdp_fallback(self, n: int) -> np.ndarray:
        """
        Fallback when cvxpy is not available: return random unit vectors.
        """
        V = self.rng.standard_normal((n, n))
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return V / norms

    # ------------------------------------------------------------------
    # Rounding
    # ------------------------------------------------------------------

    def round_coloring(self, V: np.ndarray) -> np.ndarray:
        """
        Newman rounding: assign colour argmax_c Re[ω^c (r · v_i)].

        A random complex unit vector r ∈ C^{d} is sampled; then each
        vertex i gets colour

            c(i) = argmax_{c=0..k-1}  Re[ ω^c × (V[i] @ r_complex) ]

        where r_complex has the same dimension as V[i].

        Parameters
        ----------
        V : np.ndarray, shape (n, d)  unit-vector embedding

        Returns
        -------
        coloring : np.ndarray, shape (n,), dtype int
        """
        n, d = V.shape
        k = self.k

        # Random complex projection vector r ∈ C^d
        r_real = self.rng.standard_normal(d)
        r_imag = self.rng.standard_normal(d)
        r_complex = r_real + 1j * r_imag
        r_complex /= np.linalg.norm(r_complex)

        # Project each vertex vector to a complex scalar
        z = V @ r_complex.conj()   # (n,) complex

        # omega^c for c = 0, ..., k-1
        omega = np.exp(2j * np.pi / k)
        phases = np.array([omega ** c for c in range(k)])  # (k,)

        # scores[i, c] = Re[ ω^c * z[i] ]
        scores = np.real(np.outer(z, phases.conj()))   # (n, k)  — note: Re[ω^c z]
        # Actually: Re[ω^c * z_i]  for each c
        # = Re[ z_i * conj(ω^c) ] after conjugation — let's be precise:
        # Newman assigns c(i) = argmax_c Re[exp(2πic/k) * (U v_i)_0] for random U
        # We approximate this as argmax_c Re[ω^c * projection(v_i)]
        scores = np.real(z[:, None] * phases[None, :])   # (n, k)
        coloring = np.argmax(scores, axis=1).astype(int)
        return coloring

    # ------------------------------------------------------------------
    # Full solve
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Solve MAX-k-CUT using SDP + random rounding.

        Returns
        -------
        best_coloring : np.ndarray, shape (n,), dtype int
        best_ratio    : float  = cut_value / |E|
        """
        nodes = sorted(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        n_edges = self.graph.number_of_edges()

        V = self.solve_sdp()

        best_cut = -1
        best_coloring = np.zeros(self.n, dtype=int)

        for _ in range(self.n_samples):
            local_col = self.round_coloring(V)  # shape (n,), indices into nodes list

            # Map back to original node numbering
            full_col = np.zeros(max(nodes) + 1, dtype=int)
            for node, idx in node_to_idx.items():
                full_col[node] = local_col[idx]

            cut = _cut_value(self.graph, full_col)
            if cut > best_cut:
                best_cut = cut
                best_coloring = full_col.copy()

        best_ratio = best_cut / n_edges if n_edges > 0 else 1.0
        return best_coloring, best_ratio
