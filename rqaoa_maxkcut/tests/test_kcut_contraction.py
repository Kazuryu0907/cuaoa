"""
Regression tests for the k≥3 RQAOA contraction bug fix.

The old `_contract_node` discarded the cyclic-shift component of edges
incident on the eliminated node, yielding incorrect costs for k≥3.
After the fix:

  - Edges carry an `h_hat` attribute (length-k complex array).
  - Contraction shifts ĥ by `s` along the (elim, h) edges.
  - The new brute-force cost reads ĥ_e((x_v − x_u) mod k) per edge.

These tests verify both that the fix is correct on contrived small
graphs and that k=2 (MaxCut) is unaffected.
"""
from __future__ import annotations

import itertools
import os
import sys

import networkx as nx
import numpy as np
import pytest

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_cuaoa_dir = os.path.dirname(os.path.dirname(_tests_dir))
sys.path.insert(0, _cuaoa_dir)

from rqaoa_maxkcut.core.rqaoa import RQAOA1Solver
from rqaoa_maxkcut.core.hamiltonian import MaxKCutHamiltonian


def _exact_kcut(graph: nx.Graph, k: int) -> float:
    """Brute-force optimal MAX-k-CUT value for a small graph."""
    nodes = sorted(graph.nodes())
    best = 0
    for col in itertools.product(range(k), repeat=len(nodes)):
        m = {n: col[i] for i, n in enumerate(nodes)}
        cut = sum(1 for u, v in graph.edges() if m[u] != m[v])
        best = max(best, cut)
    return float(best)


def _build_solver(graph: nx.Graph, k: int) -> RQAOA1Solver:
    return RQAOA1Solver(graph, k=k, n_cutoff=2, use_gpu=False, verbose=False)


# ---------------------------------------------------------------------
# Direct contraction tests
# ---------------------------------------------------------------------

def test_contract_triangle_k3():
    """K_3, k=3: contracting v0--v1 with shift=1 should leave a single
    edge (v0, v2) whose ĥ encodes the merged cost from both originals."""
    G = nx.complete_graph(3)
    solver = _build_solver(G, k=3)
    new_g = solver._contract_node(G, ref=0, elim=1, shift=1, k=3)
    assert set(new_g.nodes()) == {0, 2}
    assert new_g.has_edge(0, 2)
    h_hat = new_g[0][2]["h_hat"]
    # Original (0,2) ĥ = [0,1,1]; contribution from (1,2) shifted by 1
    # gives ĥ((b-1) mod 3) of [0,1,1] = [1, 0, 1]. Sum = [1, 1, 2].
    np.testing.assert_allclose(np.real(h_hat), [1.0, 1.0, 2.0], atol=1e-12)


def test_contract_path_k3_shift_zero_no_op():
    """shift=0 keeps the edge structure unchanged in ĥ space (just
    relabels x_elim ≡ x_ref)."""
    G = nx.path_graph(3)  # 0-1-2
    solver = _build_solver(G, k=3)
    new_g = solver._contract_node(G, ref=0, elim=1, shift=0, k=3)
    assert new_g.has_edge(0, 2)
    # New (0,2): ĥ((b-0) mod 3) of original (1,2) = [0,1,1].
    h_hat = new_g[0][2]["h_hat"]
    np.testing.assert_allclose(np.real(h_hat), [0.0, 1.0, 1.0], atol=1e-12)


def test_contract_zero_drops_edge():
    """An edge whose ĥ collapses to zero should be removed."""
    # Set up: triangle with the (1,2) edge having a custom ĥ that
    # happens to cancel with the (0,2) shifted contribution.
    G = nx.Graph()
    G.add_edge(0, 1, h_hat=np.array([0, 1, 1], dtype=np.complex128))
    G.add_edge(1, 2, h_hat=np.array([0, -1, -1], dtype=np.complex128))
    G.add_edge(0, 2, h_hat=np.array([0, 1, 1], dtype=np.complex128))
    solver = _build_solver(G, k=3)
    new_g = solver._contract_node(G, ref=0, elim=1, shift=0, k=3)
    # (0,2) old [0,1,1] + shift-0 of (1,2) [0,-1,-1] = [0,0,0] → drop
    assert not new_g.has_edge(0, 2)


def test_contract_k2_matches_unweighted():
    """k=2: shift=1 contraction should still yield a single edge between
    ref and the remaining neighbour (the original unweighted-merge
    behaviour). ĥ encodes the sign flip."""
    G = nx.complete_graph(3)
    solver = _build_solver(G, k=2)
    new_g = solver._contract_node(G, ref=0, elim=1, shift=1, k=2)
    assert new_g.has_edge(0, 2)
    h_hat = new_g[0][2]["h_hat"]
    # (0,2) old [0,1] + shift-1 of (1,2) [0,1]: ĥ((b-1)%2) → [1,0]
    # Sum = [1, 1]. h_hat = [1, 1].
    np.testing.assert_allclose(np.real(h_hat), [1.0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------
# Brute-force cost on contracted graphs
# ---------------------------------------------------------------------

def test_brute_force_unweighted_unchanged():
    """For a fresh (no h_hat) graph the brute force should still give
    the unweighted MAX-k-CUT value."""
    G = nx.cycle_graph(5)
    for k in (2, 3, 4):
        solver = _build_solver(G, k=k)
        coloring = solver._brute_force(G, k)
        # Compute the cut via the canonical unweighted definition
        cut = sum(1 for u, v in G.edges() if coloring[u] != coloring[v])
        assert cut == int(_exact_kcut(G, k))


# ---------------------------------------------------------------------
# Full-pipeline smoke: K_3 with k=3 must be perfectly cut (3 edges, 3 colours)
# ---------------------------------------------------------------------

def test_k3_complete_3_perfect():
    G = nx.complete_graph(3)
    solver = RQAOA1Solver(G, k=3, n_cutoff=2, use_gpu=False, verbose=False,
                          n_grid=10)
    coloring, ratio = solver.solve()
    assert ratio == pytest.approx(1.0, abs=1e-9), (
        f"K_3 with k=3 should be perfectly 3-coloured; got ratio={ratio}"
    )


def test_k3_complete_4_optimal():
    """K_4, k=3: max cut = 5 (one colour class of size 2, two singletons)."""
    G = nx.complete_graph(4)
    solver = RQAOA1Solver(G, k=3, n_cutoff=2, use_gpu=False, verbose=False,
                          n_grid=10)
    coloring, ratio = solver.solve()
    # |E| = 6. Optimal cut = 5 (colour multiset {0,0,1,2} or perms).
    assert ratio >= 5.0 / 6.0 - 1e-9
