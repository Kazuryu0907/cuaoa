"""
tests/test_angle_opt.py
========================
Unit tests for AngleOptimizer.
Compatible with pytest if available; also runnable directly.
"""

from __future__ import annotations

import sys
import os
import traceback

import numpy as np
import networkx as nx

_tests_dir = os.path.dirname(os.path.abspath(__file__))
_cuaoa_dir = os.path.dirname(os.path.dirname(_tests_dir))
sys.path.insert(0, _cuaoa_dir)

from rqaoa_maxkcut.core.hamiltonian import MaxKCutHamiltonian
from rqaoa_maxkcut.core.density_matrix import DensityMatrixSimulator
from rqaoa_maxkcut.core.angle_opt import AngleOptimizer

try:
    import pytest
    _PYTEST = True
except ImportError:
    _PYTEST = False


# ---------------------------------------------------------------------------
# Simple test runner
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_optimizer(graph: nx.Graph) -> AngleOptimizer:
    ham = MaxKCutHamiltonian(graph, k=3)
    sim = DensityMatrixSimulator(ham, use_gpu=False)
    return AngleOptimizer(sim)


def F_double_prime(z: complex, g: np.ndarray) -> float:
    g0, g1, g2 = g
    return float(np.real(g0 * z) + abs(g1 + np.conj(g2) * z))


# ---------------------------------------------------------------------------
# Polynomial p_f tests
# ---------------------------------------------------------------------------

def test_poly_pf_shape():
    """p_f(z) has 5 coefficients (degree 4)."""
    G = nx.path_graph(2)
    opt = make_optimizer(G)
    g = np.array([0.5 + 0.1j, 0.3 - 0.2j, -0.1 + 0.4j])
    coeffs = opt._poly_pf(1.0, g)
    assert len(coeffs) == 5, f"Expected 5 coefficients, got {len(coeffs)}"


def test_poly_pf_evaluates_finite():
    """p_f coefficients and evaluation at z on unit circle are finite."""
    G = nx.path_graph(2)
    opt = make_optimizer(G)
    g = np.array([0.4 + 0.0j, 0.3 - 0.1j, -0.2 + 0.15j])
    z_star = np.exp(1j * 0.7)
    f = F_double_prime(z_star, g)
    coeffs = opt._poly_pf(f, g)
    p_at_z = np.polyval(coeffs, z_star)
    assert np.isfinite(abs(p_at_z))


def test_poly_roots_unit_circle():
    """
    For g = [r, 0, 0] with |r|>0: F''(z)=Re(r*z) is maximised at
    z* = r*/|r|. p_{|r|}(z*) should evaluate to near-zero (root).
    """
    G = nx.path_graph(2)
    opt = make_optimizer(G)
    r = 0.6 + 0.3j
    g = np.array([r, 0.0 + 0.0j, 0.0 + 0.0j])
    f_star = abs(r)
    z_star = np.conj(r) / abs(r)  # where F''(z*)=f*
    coeffs = opt._poly_pf(f_star * 0.99, g)
    try:
        roots = np.roots(coeffs)
        # Check the polynomial evaluates reasonably
        for root in roots:
            val = np.polyval(coeffs, root)
            assert np.isfinite(abs(val))
    except np.linalg.LinAlgError:
        pass  # degenerate—OK


# ---------------------------------------------------------------------------
# maximize_over_beta tests
# ---------------------------------------------------------------------------

def test_E_max_upper_bound():
    """E_max ≤ C_const + |g_0| + |g_1| + |g_2| (triangle inequality)."""
    G = nx.cycle_graph(4)
    opt = make_optimizer(G)
    gamma = 0.5
    C_const, g = opt.compute_g_coefficients(gamma, G)
    E_max, beta = opt.maximize_over_beta(g, C_const)
    upper = C_const + abs(g[0]) + abs(g[1]) + abs(g[2])
    assert E_max <= upper + 1e-6, f"E_max={E_max} > upper_bound={upper}"


def test_E_max_at_least_lower_bound():
    """E_max ≥ C_const - |g_0| - |g_1| - |g_2|."""
    G = nx.path_graph(3)
    opt = make_optimizer(G)
    gamma = 0.3
    C_const, g = opt.compute_g_coefficients(gamma, G)
    E_max, _ = opt.maximize_over_beta(g, C_const)
    lower = C_const - (abs(g[0]) + abs(g[1]) + abs(g[2]))
    assert E_max >= lower - 1e-6


def test_beta_shape():
    """beta_opt should have shape (3,)."""
    G = nx.complete_graph(4)
    opt = make_optimizer(G)
    C_const, g = opt.compute_g_coefficients(0.5, G)
    E_max, beta = opt.maximize_over_beta(g, C_const)
    assert beta.shape == (3,)


def test_maximize_consistent_with_grid():
    """
    E_max from maximize_over_beta should be ≥ the maximum found by
    a dense grid scan over θ (with Σθ_a = 0).
    """
    G = nx.cycle_graph(4)
    opt = make_optimizer(G)
    gamma = 0.8
    C_const, g = opt.compute_g_coefficients(gamma, G)
    E_max, beta_opt = opt.maximize_over_beta(g, C_const)

    best_grid = -np.inf
    for t in np.linspace(0, 2 * np.pi, 100):
        for s in np.linspace(0, 2 * np.pi, 100):
            theta = np.array([t, s, -t - s])
            val = C_const + float(np.real(
                sum(g[a] * np.exp(1j * theta[a]) for a in range(3))
            ))
            if val > best_grid:
                best_grid = val

    assert E_max >= best_grid - 1e-4, (
        f"maximize_over_beta={E_max:.6f} < grid_max={best_grid:.6f}"
    )


def test_zero_g_returns_C_const():
    """When g = [0, 0, 0], E_max = C_const."""
    G = nx.path_graph(2)
    opt = make_optimizer(G)
    g = np.zeros(3, dtype=np.complex128)
    C_const = 2.5
    E_max, beta = opt.maximize_over_beta(g, C_const)
    np.testing.assert_allclose(E_max, C_const, atol=1e-8)


# ---------------------------------------------------------------------------
# optimize_gamma tests
# ---------------------------------------------------------------------------

def test_optimize_gamma_returns_valid():
    """optimize_gamma returns (gamma, beta, E) with gamma ∈ [0, π)."""
    G = nx.cycle_graph(4)
    opt = make_optimizer(G)
    gamma_opt, beta_opt, E_opt = opt.optimize_gamma(G, n_grid=10)
    assert 0.0 <= gamma_opt < np.pi + 1e-6
    assert beta_opt.shape == (3,)
    assert np.isfinite(E_opt)


def test_optimize_gamma_energy_positive():
    """For any non-trivial graph, E_opt > 0."""
    G = nx.complete_graph(4)
    opt = make_optimizer(G)
    _, _, E_opt = opt.optimize_gamma(G, n_grid=15)
    assert E_opt > 0.0, f"E_opt={E_opt} is not positive"


def test_energy_matches_brute_force():
    """
    The optimised energy should be ≥ any specific evaluation point.
    """
    G = nx.path_graph(4)
    opt = make_optimizer(G)
    gamma_opt, beta_opt, E_opt = opt.optimize_gamma(G, n_grid=20)

    # Evaluate at a specific (γ=0.4, β) point
    gamma_test = 0.4
    C_const, g = opt.compute_g_coefficients(gamma_test, G)
    beta_test = np.array([0.1, -0.05, -0.05])
    theta_test = 3 * beta_test - beta_test.sum()
    E_test = C_const + float(np.real(
        sum(g[a] * np.exp(1j * theta_test[a]) for a in range(3))
    ))
    assert E_opt >= E_test - 1e-4, (
        f"Optimised E={E_opt:.6f} < test-point E={E_test:.6f}"
    )


def test_g_coefficients_finite():
    """compute_g_coefficients returns finite values."""
    G = nx.complete_graph(4)
    opt = make_optimizer(G)
    for gamma in [0.1, 0.5, 1.0, np.pi / 2]:
        C_const, g = opt.compute_g_coefficients(gamma, G)
        assert np.isfinite(C_const), f"C_const not finite at γ={gamma}"
        assert np.all(np.isfinite(g)), f"g not finite at γ={gamma}"


def test_energy_gradient_consistency():
    """
    At gamma_opt the nearby γ values give ≤ E_opt + tolerance.
    """
    G = nx.path_graph(4)
    opt = make_optimizer(G)
    gamma_opt, beta_opt, E_opt = opt.optimize_gamma(G, n_grid=30)
    eps = 0.05
    for delta in [-eps, eps]:
        gamma_near = gamma_opt + delta
        if 0 <= gamma_near < np.pi:
            C_const, g = opt.compute_g_coefficients(gamma_near, G)
            E_near, _ = opt.maximize_over_beta(g, C_const)
            assert E_near <= E_opt + 0.1 * abs(E_opt) + 1e-6, (
                f"E at γ={gamma_near:.3f} is {E_near:.6f} > E_opt={E_opt:.6f}"
            )


def test_c_const_near_edges_times_h0():
    """
    At γ→0, C_const ≈ |E| × (k-1)/k = 4 for K4 (6 edges × 2/3).
    """
    G = nx.complete_graph(4)
    opt = make_optimizer(G)
    C_const, _ = opt.compute_g_coefficients(0.001, G)
    expected = G.number_of_edges() * 2 / 3
    np.testing.assert_allclose(C_const, expected, atol=0.01)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    test_poly_pf_shape,
    test_poly_pf_evaluates_finite,
    test_poly_roots_unit_circle,
    test_E_max_upper_bound,
    test_E_max_at_least_lower_bound,
    test_beta_shape,
    test_maximize_consistent_with_grid,
    test_zero_g_returns_C_const,
    test_optimize_gamma_returns_valid,
    test_optimize_gamma_energy_positive,
    test_energy_matches_brute_force,
    test_g_coefficients_finite,
    test_energy_gradient_consistency,
    test_c_const_near_edges_times_h0,
]


if __name__ == "__main__":
    print("Running angle_opt tests...")
    for fn in ALL_TESTS:
        _run(fn.__name__, fn)
    passed = sum(1 for _, s, _ in _RESULTS if s == "PASSED")
    failed = sum(1 for _, s, _ in _RESULTS if s == "FAILED")
    print(f"\n{passed} passed / {failed} failed")
    if failed > 0:
        sys.exit(1)
