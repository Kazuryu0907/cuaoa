"""
core/rqaoa.py
=============
Level-1 RQAOA solver for MAX-k-CUT.

Algorithm (Bravyi et al. 2022, Section 2.2)
-------------------------------------------
Repeat until |V| ≤ n_cutoff:
  1. Optimise (γ*, β*) for the current graph.
  2. Compute M_{ij}(b) = μ_{ij}(Π_{ij}(b)) for every edge (i,j) and b∈Z_k.
  3. Select (i*, j*, b*) = argmax_{i<j, b} |M_{ij}(b)|.
  4. Record constraint x_{j*} = x_{i*} + b*  (mod k).
  5. Contract: merge j* into i*, updating edge weights.

Once |V| ≤ n_cutoff: brute-force the reduced problem.
Back-substitute constraints to recover the full colouring.

Graph contraction (step 5)
--------------------------
After fixing x_j = x_i + b the term Π_{j,h}(a) in the Hamiltonian
becomes Π_{i,h}(a - b):

    C_{jh}  →  C_{ih}  with  h_{ih}(a) += h_{jh}(a + b)

In NetworkX terms: merge node j into i, adding shifted edge weights.

Node labelling
--------------
Internally we work with NetworkX graphs whose nodes are the *original*
vertex indices.  The ``active`` set shrinks by one at each step.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from itertools import product as iproduct
from typing import Dict, List, Tuple

from .hamiltonian import MaxKCutHamiltonian
from .density_matrix import DensityMatrixSimulator
from .angle_opt import AngleOptimizer


class RQAOA1Solver:
    """
    Level-1 RQAOA for MAX-k-CUT.

    Parameters
    ----------
    graph    : nx.Graph   The input graph (nodes 0..n-1).
    k        : int        Number of colours.
    n_cutoff : int        Switch to brute force when |V| ≤ n_cutoff.
    use_gpu  : bool       Use CuPy if available.
    verbose  : bool       Print progress.
    """

    def __init__(
        self,
        graph: nx.Graph,
        k: int,
        n_cutoff: int = 6,
        use_gpu: bool = True,
        verbose: bool = False,
        n_grid: int = 20,
    ) -> None:
        self.original_graph = graph
        self.k = k
        self.n = graph.number_of_nodes()
        self.n_cutoff = n_cutoff
        self.use_gpu = use_gpu
        self.verbose = verbose
        self.n_grid = n_grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> Tuple[np.ndarray, float]:
        """
        Run Level-1 RQAOA and return a colouring.

        Returns
        -------
        coloring     : np.ndarray, shape (n,), dtype int, values in Z_k
        approx_ratio : float  = C(coloring) / |E|
        """
        # Work on a copy; relabel nodes to 0..n-1 if needed
        graph = nx.convert_node_labels_to_integers(self.original_graph.copy())
        k = self.k
        n_orig = graph.number_of_nodes()

        constraints: List[Dict] = []  # [{elim: (ref, shift)}]

        current_graph = graph.copy()

        while current_graph.number_of_nodes() > self.n_cutoff:
            ham = MaxKCutHamiltonian(current_graph, k)
            sim = DensityMatrixSimulator(ham, use_gpu=self.use_gpu)

            if k == 3:
                opt = AngleOptimizer(sim)
                gamma_opt, beta_opt, E_opt = opt.optimize_gamma(current_graph, n_grid=self.n_grid)
            else:
                # Fallback: simple grid search over gamma; beta = zeros
                gamma_opt, beta_opt = self._simple_gamma_search(sim, current_graph, n_grid=self.n_grid)

            if self.verbose:
                print(f"  n={current_graph.number_of_nodes()}  "
                      f"γ*={gamma_opt:.4f}  E={E_opt if k==3 else 'N/A':.4f}")

            current_graph, constraint = self._elimination_step(
                current_graph, ham, sim, gamma_opt, beta_opt
            )
            constraints.append(constraint)

        # Brute-force the reduced problem
        base_coloring = self._brute_force(current_graph, k)

        # Back-substitute
        full_coloring = self._reconstruct_coloring(
            base_coloring, constraints, n_orig
        )

        # Compute approx ratio
        n_edges = self.original_graph.number_of_edges()
        if n_edges == 0:
            approx_ratio = 1.0
        else:
            ham_orig = MaxKCutHamiltonian(
                nx.convert_node_labels_to_integers(self.original_graph), k
            )
            cut = ham_orig.max_cut_value(full_coloring)
            approx_ratio = cut / n_edges

        return full_coloring, approx_ratio

    # ------------------------------------------------------------------
    # Elimination step
    # ------------------------------------------------------------------

    def _elimination_step(
        self,
        graph: nx.Graph,
        ham: MaxKCutHamiltonian,
        sim: DensityMatrixSimulator,
        gamma_opt: float,
        beta_opt: np.ndarray,
    ) -> Tuple[nx.Graph, Dict]:
        """
        One variable-elimination step.

        Returns
        -------
        new_graph  : nx.Graph with one fewer node
        constraint : {elim_node: (ref_node, shift_b)}
        """
        k = self.k

        # --- Compute M_{ij}(b) for all edges ---
        best_abs = -1.0
        best_i, best_j, best_b = -1, -1, 0

        for i, j in graph.edges():
            rho = sim.compute_rho(i, j, gamma_opt)
            M = sim.compute_expectation(rho, i, j, beta_opt)  # (k,)
            for b in range(k):
                val = abs(M[b])
                if val > best_abs:
                    best_abs = val
                    best_i, best_j, best_b = i, j, b

        if best_i == -1:
            # No edges: pick arbitrary pair
            nodes = list(graph.nodes())
            best_i, best_j, best_b = nodes[0], nodes[1], 0

        if self.verbose:
            print(f"    eliminate {best_j} = {best_i} + {best_b}  "
                  f"|M|={best_abs:.4f}")

        # --- Contract: merge best_j into best_i ---
        new_graph = self._contract_node(graph, best_i, best_j, best_b, k)
        constraint = {best_j: (best_i, best_b)}

        return new_graph, constraint

    def _contract_node(
        self,
        graph: nx.Graph,
        ref: int,
        elim: int,
        shift: int,
        k: int,
    ) -> nx.Graph:
        """
        Contract node `elim` into `ref` under constraint x_elim = x_ref + shift.

        Each edge carries an ``h_hat`` attribute: a length-k complex
        vector of Fourier coefficients ĥ_{u,v}(b). For a fresh
        unweighted edge ĥ = (0, 1, …, 1).

        Substituting x_elim = x_ref + s into C_{elim,h} gives
            new ĥ_{ref,h}(b) ← old ĥ_{ref,h}(b) + ĥ_{elim,h}((b − s) mod k)

        which is the cyclic-shift accumulation used by Bravyi et al. 2022.
        For k=2 there is only one non-trivial shift so the previous
        unweighted-merge implementation happened to be correct; for
        k≥3 the shift bookkeeping is essential.
        """
        new_g = nx.Graph()
        for node in graph.nodes():
            if node != elim:
                new_g.add_node(node)

        h_hat_default = np.zeros(k, dtype=np.complex128)
        h_hat_default[1:] = 1.0

        def edge_h_hat(g, u, v):
            data = g.get_edge_data(u, v) or {}
            if "h_hat" in data:
                return np.asarray(data["h_hat"], dtype=np.complex128).copy()
            return h_hat_default.copy()

        # Carry over edges that don't involve elim
        for u, v in graph.edges():
            if u == elim or v == elim:
                continue
            new_g.add_edge(u, v, h_hat=edge_h_hat(graph, u, v))

        # Process (elim, h) edges → contribute to (ref, h)
        b_arr = np.arange(k)
        shift_idx = (b_arr - shift) % k   # ĥ_{elim,h}((b-s) mod k)
        for h in graph.neighbors(elim):
            if h == ref:
                continue
            h_hat_jh = edge_h_hat(graph, elim, h)
            shifted = h_hat_jh[shift_idx]
            if new_g.has_edge(ref, h):
                new_g[ref][h]["h_hat"] = (
                    new_g[ref][h]["h_hat"] + shifted
                )
            elif ref != h:
                new_g.add_edge(ref, h, h_hat=shifted)

        # Drop edges whose ĥ collapsed to zero (numerical convention:
        # edges with ĥ ≡ 0 contribute nothing to the cost)
        zero_edges = [
            (u, v) for u, v, d in new_g.edges(data=True)
            if np.allclose(d.get("h_hat", h_hat_default), 0.0, atol=1e-12)
        ]
        for u, v in zero_edges:
            new_g.remove_edge(u, v)

        return new_g

    # ------------------------------------------------------------------
    # Brute force
    # ------------------------------------------------------------------

    def _brute_force(self, graph: nx.Graph, k: int) -> np.ndarray:
        """
        Exactly solve the (possibly contracted) MAX-k-CUT cost function
        on a small graph by exhaustive search.

        Each edge contributes ĥ_e((x_v − x_u) mod k) to the cost, where
        ĥ_e is the per-edge Fourier coefficient vector. For a fresh
        unweighted edge ĥ = (0, 1, …, 1), recovering 1{x_v ≠ x_u}.
        For RQAOA-contracted edges ĥ encodes the cyclic-shift bookkeeping.
        """
        nodes = sorted(graph.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}

        h_hat_default = np.zeros(k, dtype=np.complex128)
        h_hat_default[1:] = 1.0

        edge_data = []
        for u, v, data in graph.edges(data=True):
            i = node_to_idx[u]
            j = node_to_idx[v]
            h_hat_edge = data.get("h_hat", h_hat_default)
            edge_data.append((i, j, np.real(h_hat_edge)))

        best_cut = -np.inf
        best_coloring = np.zeros(n, dtype=int)

        for coloring_tuple in iproduct(range(k), repeat=n):
            cost = 0.0
            for i, j, h_hat_edge in edge_data:
                cost += h_hat_edge[(coloring_tuple[j] - coloring_tuple[i]) % k]
            if cost > best_cut:
                best_cut = cost
                best_coloring = np.array(coloring_tuple, dtype=int)

        result = np.zeros(max(nodes) + 1, dtype=int)
        for node, idx in node_to_idx.items():
            result[node] = best_coloring[idx]
        return result

    # ------------------------------------------------------------------
    # Back-substitution
    # ------------------------------------------------------------------

    def _reconstruct_coloring(
        self,
        base_coloring: np.ndarray,
        constraints: List[Dict],
        n_orig: int,
    ) -> np.ndarray:
        """
        Reconstruct the full colouring by back-substituting constraints.

        Constraints are in order of elimination.  We apply them in
        *reverse* order: last eliminated first.

        constraint entry: {elim: (ref, shift)}
        x_elim = (x_ref + shift) mod k
        """
        k = self.k
        coloring = base_coloring.copy()
        # Ensure the array is large enough
        if len(coloring) < n_orig:
            new_col = np.zeros(n_orig, dtype=int)
            new_col[:len(coloring)] = coloring
            coloring = new_col

        for constraint in reversed(constraints):
            for elim, (ref, shift) in constraint.items():
                coloring[elim] = (coloring[ref] + shift) % k

        return coloring[:n_orig]

    # ------------------------------------------------------------------
    # Fallback for k ≠ 3
    # ------------------------------------------------------------------

    def _simple_gamma_search(
        self,
        sim: DensityMatrixSimulator,
        graph: nx.Graph,
        n_grid: int = 20,
    ) -> Tuple[float, np.ndarray]:
        """
        Simple γ grid search for k ≠ 3 using uniform β = 0.

        Returns
        -------
        gamma_opt : float
        beta_opt  : np.ndarray, shape (k,) = zeros
        """
        k = self.k
        beta = np.zeros(k)
        best_gamma = 0.0
        best_E = -np.inf

        for gamma in np.linspace(0, np.pi, n_grid, endpoint=False):
            E = 0.0
            for u, v in graph.edges():
                rho = sim.compute_rho(u, v, gamma)
                M = sim.compute_expectation(rho, u, v, beta)
                # Sum M(b) for b != 0 approximates the edge contribution
                E += float(np.sum(M[1:]))
            if E > best_E:
                best_E = E
                best_gamma = gamma

        return best_gamma, beta
