"""
tests/test_density_matrix.py
============================
Unit tests for DensityMatrixSimulator.

All tests run on CPU (use_gpu=False) so they work without a GPU.
Compatible with pytest if available; also runnable directly.
"""

from __future__ import annotations

import sys
import os
import traceback

import numpy as np
import networkx as nx

# Allow running directly without installing the package.
# rqaoa_maxkcut lives in /workspace/data/cuaoa/, so we need that directory on sys.path.
_tests_dir = os.path.dirname(os.path.abspath(__file__))
_cuaoa_dir = os.path.dirname(os.path.dirname(_tests_dir))  # .../cuaoa/
sys.path.insert(0, _cuaoa_dir)

from rqaoa_maxkcut.core.hamiltonian import MaxKCutHamiltonian
from rqaoa_maxkcut.core.density_matrix import DensityMatrixSimulator

try:
    import pytest
    _PYTEST = True
except ImportError:
    _PYTEST = False


# ---------------------------------------------------------------------------
# Simple test runner (used when pytest is unavailable)
# ---------------------------------------------------------------------------

_RESULTS: list = []


def _run(name, fn):
    try:
        fn()
        _RESULTS.append((name, "PASSED", None))
        print(f"  PASSED  {name}")
    except Exception as e:
        _RESULTS.append((name, "FAILED", traceback.format_exc()))
        print(f"  FAILED  {name}: {e}")


def _skip(name, reason):
    _RESULTS.append((name, "SKIPPED", reason))
    print(f"  SKIPPED {name}: {reason}")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def make_k4_graph() -> nx.Graph:
    return nx.complete_graph(4)


def make_path_graph(n: int = 4) -> nx.Graph:
    return nx.path_graph(n)


def make_cycle_graph(n: int = 6) -> nx.Graph:
    return nx.cycle_graph(n)


def make_sim(graph: nx.Graph, k: int = 3, use_gpu: bool = False):
    ham = MaxKCutHamiltonian(graph, k)
    return DensityMatrixSimulator(ham, use_gpu=use_gpu)


def digit(idx: int, pos: int, k: int, n: int) -> int:
    return (idx // (k ** (n - 1 - pos))) % k


# ---------------------------------------------------------------------------
# Hamiltonian tests
# ---------------------------------------------------------------------------

def test_h_values_k3():
    """For k=3 edge: h(0)=2/3, h(1)=h(2)=-1/3."""
    G = nx.path_graph(2)
    ham = MaxKCutHamiltonian(G, k=3)
    h = ham.get_h(0, 1)
    assert h.shape == (3,)
    np.testing.assert_allclose(h[0].real, 2/3, atol=1e-10)
    np.testing.assert_allclose(h[1].real, -1/3, atol=1e-10)
    np.testing.assert_allclose(h[2].real, -1/3, atol=1e-10)


def test_h_hat_values_k3():
    """For k=3 edge: ĥ(0)=0, ĥ(1)=ĥ(2)=1."""
    G = nx.path_graph(2)
    ham = MaxKCutHamiltonian(G, k=3)
    h_hat = ham.get_h_hat(0, 1)
    assert h_hat.shape == (3,)
    np.testing.assert_allclose(h_hat[0].real, 0.0, atol=1e-10)
    np.testing.assert_allclose(h_hat[1].real, 1.0, atol=1e-10)
    np.testing.assert_allclose(h_hat[2].real, 1.0, atol=1e-10)


def test_h_nonedge_is_zero():
    """Non-edge should return zeros."""
    G = nx.path_graph(3)
    ham = MaxKCutHamiltonian(G, k=3)
    h = ham.get_h(0, 2)
    np.testing.assert_allclose(h, 0.0, atol=1e-12)


def test_h_hat_nonedge_is_zero():
    G = nx.path_graph(3)
    ham = MaxKCutHamiltonian(G, k=3)
    h_hat = ham.get_h_hat(0, 2)
    np.testing.assert_allclose(h_hat, 0.0, atol=1e-12)


def test_h_values_k4():
    """For k=4 edge: h(0)=3/4, h(a≠0)=-1/4."""
    G = nx.path_graph(2)
    ham = MaxKCutHamiltonian(G, k=4)
    h = ham.get_h(0, 1)
    np.testing.assert_allclose(h[0].real, 3/4, atol=1e-10)
    for a in range(1, 4):
        np.testing.assert_allclose(h[a].real, -1/4, atol=1e-10)


def test_h_sum_equals_zero():
    """Σ_a h_{uv}(a) = ĥ(0) = 0 for edges."""
    G = nx.cycle_graph(5)
    ham = MaxKCutHamiltonian(G, k=3)
    for u, v in G.edges():
        h = ham.get_h(u, v)
        np.testing.assert_allclose(h.sum().real, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Density matrix property tests
# ---------------------------------------------------------------------------

def test_rho_shape():
    G = make_k4_graph()
    sim = make_sim(G, k=3)
    rho = sim.compute_rho(0, 1, 0.5)
    assert rho.shape == (9, 9)


def test_rho_trace_one():
    """Tr[ρ_{uv}] = 1."""
    G = make_k4_graph()
    sim = make_sim(G, k=3)
    for u, v in [(0, 1), (0, 2), (1, 3)]:
        for gamma in [0.3, 0.7, 1.2]:
            rho = sim.compute_rho(u, v, gamma)
            tr = np.trace(rho)
            np.testing.assert_allclose(tr.real, 1.0, atol=1e-9,
                                       err_msg=f"Tr[ρ_{u}{v}] ≠ 1 at γ={gamma}")
            np.testing.assert_allclose(abs(tr.imag), 0.0, atol=1e-9)


def test_rho_hermitian():
    """ρ_{uv} = ρ_{uv}†."""
    G = make_k4_graph()
    sim = make_sim(G, k=3)
    for u, v in [(0, 1), (1, 2)]:
        rho = sim.compute_rho(u, v, 0.5)
        diff = np.max(np.abs(rho - rho.conj().T))
        assert diff < 1e-10, f"ρ_{u}{v} is not Hermitian: max_diff={diff}"


def test_rho_psd():
    """ρ_{uv} has all eigenvalues ≥ -ε (PSD)."""
    G = make_k4_graph()
    sim = make_sim(G, k=3)
    for u, v in [(0, 1), (0, 3)]:
        rho = sim.compute_rho(u, v, 0.5)
        eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
        assert eigvals.min() >= -1e-8, (
            f"ρ_{u}{v} has negative eigenvalue {eigvals.min()}"
        )


def test_rho_trace_one_gamma_zero():
    """At γ=0, ρ_{uv} = |+><+|^{⊗2} (maximally mixed)."""
    G = make_k4_graph()
    k = 3
    sim = make_sim(G, k=k)
    rho = sim.compute_rho(0, 1, gamma=0.0)
    expected = np.ones((k**2, k**2)) / k**2
    np.testing.assert_allclose(rho, expected, atol=1e-10)


def test_rho_path_graph():
    """On a path graph, ρ_{0,2} (connected via node 1) is valid."""
    G = make_path_graph(4)
    sim = make_sim(G, k=3)
    rho = sim.compute_rho(0, 2, gamma=0.3)
    np.testing.assert_allclose(np.trace(rho).real, 1.0, atol=1e-9)
    eigvals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    assert eigvals.min() >= -1e-8


def test_small_graph_brute_force():
    """
    K4, k=3, γ=0.5: compute_rho should match the brute-force
    2-qudit marginal of exp(iγC)|+>^4.
    """
    G = nx.complete_graph(4)
    k = 3
    n = 4
    gamma = 0.5

    ham = MaxKCutHamiltonian(G, k)
    sim = DensityMatrixSimulator(ham, use_gpu=False)

    # Full state vector
    dim = k ** n
    C_diag = np.zeros(dim, dtype=np.float64)
    for u, v in G.edges():
        h_hat_uv = ham.get_h_hat(u, v).real
        for idx in range(dim):
            au = digit(idx, u, k, n)
            av = digit(idx, v, k, n)
            C_diag[idx] += h_hat_uv[(au - av) % k]

    state_evolved = np.exp(1j * gamma * C_diag) / np.sqrt(dim)
    psi = state_evolved.reshape([k] * n)

    # Trace out qudits 2 and 3
    rho_bf = np.einsum("abef,cdef->abcd", psi, psi.conj()).reshape(k**2, k**2)
    rho_computed = sim.compute_rho(0, 1, gamma)

    np.testing.assert_allclose(rho_computed, rho_bf, atol=1e-8,
                               err_msg="compute_rho does not match brute-force marginal")


def test_gpu_cpu_consistency():
    """CuPy (GPU) and NumPy (CPU) give consistent ρ_{uv}."""
    try:
        import cupy
    except ImportError:
        print("    (skipping: CuPy not available)")
        return

    G = make_cycle_graph(6)
    k = 3
    ham = MaxKCutHamiltonian(G, k)
    sim_cpu = DensityMatrixSimulator(ham, use_gpu=False)
    sim_gpu = DensityMatrixSimulator(ham, use_gpu=True)

    for u, v in [(0, 1), (0, 3)]:
        rho_cpu = sim_cpu.compute_rho(u, v, gamma=0.7)
        rho_gpu = sim_gpu.compute_rho(u, v, gamma=0.7)
        np.testing.assert_allclose(rho_cpu, rho_gpu, atol=1e-7,
                                   err_msg=f"CPU/GPU mismatch for ρ_{u}{v}")


# ---------------------------------------------------------------------------
# Expectation value tests
# ---------------------------------------------------------------------------

def test_M_sums_to_one():
    """Σ_b M_{ij}(b) = 1 (probability sum)."""
    G = make_cycle_graph(4)
    k = 3
    ham = MaxKCutHamiltonian(G, k)
    sim = DensityMatrixSimulator(ham, use_gpu=False)
    beta = np.zeros(k)
    for u, v in G.edges():
        rho = sim.compute_rho(u, v, gamma=0.5)
        M = sim.compute_expectation(rho, u, v, beta)
        np.testing.assert_allclose(M.sum(), 1.0, atol=1e-9,
                                   err_msg=f"Σ_b M_{u}{v}(b) ≠ 1")


def test_M_nonnegative():
    """M_{ij}(b) ≥ 0 (probability)."""
    G = make_cycle_graph(4)
    k = 3
    ham = MaxKCutHamiltonian(G, k)
    sim = DensityMatrixSimulator(ham, use_gpu=False)
    beta = np.array([0.1, -0.2, 0.1])
    for u, v in G.edges():
        rho = sim.compute_rho(u, v, gamma=0.5)
        M = sim.compute_expectation(rho, u, v, beta)
        assert np.all(M >= -1e-9), f"M_{u}{v} has negative entry: {M}"


def test_M_beta_zero_uniform():
    """At β=0 and γ=0, M(b) = 1/k for all b."""
    G = nx.path_graph(2)
    k = 3
    ham = MaxKCutHamiltonian(G, k)
    sim = DensityMatrixSimulator(ham, use_gpu=False)
    beta = np.zeros(k)
    rho = sim.compute_rho(0, 1, gamma=0.0)
    M = sim.compute_expectation(rho, 0, 1, beta)
    np.testing.assert_allclose(M, 1/k, atol=1e-9)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_h_values_k3,
    test_h_hat_values_k3,
    test_h_nonedge_is_zero,
    test_h_hat_nonedge_is_zero,
    test_h_values_k4,
    test_h_sum_equals_zero,
    test_rho_shape,
    test_rho_trace_one,
    test_rho_hermitian,
    test_rho_psd,
    test_rho_trace_one_gamma_zero,
    test_rho_path_graph,
    test_small_graph_brute_force,
    test_gpu_cpu_consistency,
    test_M_sums_to_one,
    test_M_nonnegative,
    test_M_beta_zero_uniform,
]


if __name__ == "__main__":
    print("Running density_matrix tests...")
    for fn in ALL_TESTS:
        _run(fn.__name__, fn)
    passed = sum(1 for _, s, _ in _RESULTS if s == "PASSED")
    failed = sum(1 for _, s, _ in _RESULTS if s == "FAILED")
    skipped = sum(1 for _, s, _ in _RESULTS if s == "SKIPPED")
    print(f"\n{passed} passed / {failed} failed / {skipped} skipped")
    if failed > 0:
        sys.exit(1)
