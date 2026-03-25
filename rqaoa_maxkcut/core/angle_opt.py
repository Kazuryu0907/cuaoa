"""
core/angle_opt.py
=================
Analytic β-optimisation for Level-1 QAOA on MAX-k-CUT with k=3.

Reference: Bravyi et al. 2022, Appendix A.

Energy decomposition (k=3)
--------------------------
For fixed γ the energy has the form

    E(β) = C_const + Re Σ_{a∈Z_3} g_a exp(i θ_a)

where θ_a = 3β_a - β̄,  β̄ = β_0 + β_1 + β_2,  and Σ_a θ_a = 0.

Setting z = exp(iθ_0) and using θ_1 + θ_2 = -θ_0, the constrained
maximum over θ reduces to

    F''(z) = Re(g_0 z) + |g_1 + g_2* z|

maximised over the unit circle |z|=1.

The optimal z* is a root of the degree-4 polynomial p_f(z)=0 where f
is the maximum value of F''.  We find f* by binary search and then
recover θ_1 from the phase of (g_1 + g_2* z*).

Finally β is obtained from θ via β_a = θ_a / 3  (choosing β̄ = 0).
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Tuple

try:
    import cupy as cp
except ImportError:
    cp = None

from .hamiltonian import MaxKCutHamiltonian
from .density_matrix import DensityMatrixSimulator


class AngleOptimizer:
    """
    Level-1 QAOA angle optimiser (k=3 only).

    Parameters
    ----------
    simulator : DensityMatrixSimulator
    """

    def __init__(self, simulator: DensityMatrixSimulator) -> None:
        self.sim = simulator
        self.ham = simulator.ham
        if simulator.k != 3:
            raise ValueError("AngleOptimizer is only implemented for k=3")
        self.k = 3
        self._xp = simulator.xp
        self._use_gpu = simulator._use_gpu

        # Precompute DFT matrix U and U2 = U⊗U on the compute device.
        # U_{ab} = ω^{ab}/√k, shape (k,k)
        omega = np.exp(2j * np.pi / self.k)
        a_arr = np.arange(self.k)
        U_np = np.array(
            [[omega ** (a * b) for b in range(self.k)] for a in range(self.k)],
            dtype=np.complex128,
        ) / np.sqrt(self.k)
        U2_np = np.kron(U_np, U_np)  # (k^2, k^2)
        self._U2_dev = self._xp.asarray(U2_np)          # on GPU/CPU device
        self._U2H_dev = self._xp.asarray(U2_np.conj().T)  # U2†

        # Precompute index pairs for g-coefficient extraction (k=3)
        k = self.k
        self._idx_C_const = [
            ((a + 1) % k * k + a, a * k + (a + 1) % k)
            for a in range(k)
        ]
        # For g[a]: (row, col) = (idx(a+1,a-1), idx(a,a)) and (idx(a-1,a+1), idx(a,a))
        self._idx_g = [
            (
                (a + 1) % k * k + (a - 1) % k,
                (a - 1) % k * k + (a + 1) % k,
                a * k + a,
            )
            for a in range(k)
        ]

    # ------------------------------------------------------------------
    # g-coefficient computation
    # ------------------------------------------------------------------

    def compute_g_coefficients(
        self,
        gamma: float,
        graph: nx.Graph,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute E(β) = C_const + Re Σ_a g_a exp(iθ_a) coefficients.

        Derivation
        ----------
        In the |φ_a> basis the energy contribution from edge (p,q) is

            E_{pq}(β) = Σ_{c,d,m} h_{pq}(m) exp(i(β_{c+m} + β_{d-m} - β_c - β_d))
                        × (ρ_φ^{pq})_{(c+m,d-m),(c,d)}

        where ρ_φ = (U†⊗U†) ρ (U⊗U) is ρ in the φ-basis,
        U_{ab} = ω^{ab}/√k  (DFT / √k).

        The m=0 term is β-independent → C_const.
        The m=1 and m=2 terms give the g_a coefficients.

        For k=3 with h_{pq}(1) = h_{pq}(2) = -1/3 (real), gathering
        terms with phase exp(iθ_a) = exp(i(3β_a - β̄)) gives:

            g_a = 2 × h_{pq}(1) × Σ_{p<q}
                  [ (ρ_φ)_{(a+1,a-1),(a,a)} + (ρ_φ)_{(a-1,a+1),(a,a)} ]

        and the constant term from the spec:

            C_const = Σ_{p<q} [ h_{pq}(0)
                     + 2 Re Σ_a h_{pq}(1) (ρ_φ)_{(a+1,a),(a,a+1)} ]

        Note: all indices are mod k.

        Parameters
        ----------
        gamma : float
        graph : nx.Graph  (should match self.ham.graph)

        Returns
        -------
        C_const : float
        g       : np.ndarray shape (3,), complex128
        """
        xp = self._xp
        k = self.k
        k2 = k * k
        ham = self.ham
        h0 = (k - 1) / k   # h_{pq}(0) for any edge
        h1 = -1.0 / k       # h_{pq}(1) = h_{pq}(2) for any edge

        # --- Batch: compute all edge rhos at once on GPU ---
        # rho_batch: (m, k^2, k^2)  — lives on device
        rho_batch = self.sim.compute_rho_batch(gamma)

        # --- Transform to |φ_a> basis: ρ_φ[e] = U2† @ rho[e] @ U2 ---
        # Batched matmul: (m, k^2, k^2)
        # xp.matmul broadcasts over leading batch dimension
        rho_phi_batch = xp.matmul(
            self._U2H_dev[None],                 # (1, k^2, k^2)
            xp.matmul(rho_batch, self._U2_dev[None])   # (m, k^2, k^2)
        )  # (m, k^2, k^2)

        m = rho_phi_batch.shape[0]

        # --- Accumulate C_const ---
        # h0 contribution: h0 × m  (each edge contributes h0 × Tr[ρ_φ] = h0)
        C_const = float(h0 * m)

        # 2 Re Σ_a h1 ρ_φ[e, (a+1)*k+a, a*k+(a+1)]  summed over e and a
        for row_i, col_i in self._idx_C_const:
            vals = rho_phi_batch[:, row_i, col_i]   # (m,)  on device
            C_const += float(2.0 * h1 * xp.sum(vals).real)

        # --- Accumulate g ---
        # Collect the needed matrix elements into a CPU array in one transfer
        # g[a] = 2*h1 * Σ_e (rho_phi[e, row1, col] + rho_phi[e, row2, col])
        rows1 = [r for r, _, _ in self._idx_g]
        rows2 = [r for _, r, _ in self._idx_g]
        cols  = [c for _, _, c in self._idx_g]

        # Gather: shape (k, m) for row1 and row2
        elems1 = rho_phi_batch[:, rows1, cols].T   # (k, m) — batch gather
        elems2 = rho_phi_batch[:, rows2, cols].T   # (k, m)
        sums = (elems1 + elems2).sum(axis=1)        # (k,) on device

        if self._use_gpu:
            sums_np = cp.asnumpy(sums)
        else:
            sums_np = np.asarray(sums)

        g = 2.0 * h1 * sums_np  # (k,) complex128

        return C_const, g

    # ------------------------------------------------------------------
    # β optimisation
    # ------------------------------------------------------------------

    def _poly_pf(self, f: float, g: np.ndarray) -> np.ndarray:
        """
        Coefficients of p_f(z) in descending order (for np.roots).

        p_f(z) = -(g_0²/4) z⁴
               + (g_1* g_2* + f g_0) z³
               + (|g_1|² + |g_2|² - f² - |g_0|²/2) z²
               + (g_1 g_2 + f g_0*) z
               - (g_0*)²/4
        """
        g0, g1, g2 = g
        return np.array([
            -g0 ** 2 / 4.0,
            np.conj(g1) * np.conj(g2) + f * g0,
            abs(g1) ** 2 + abs(g2) ** 2 - f ** 2 - abs(g0) ** 2 / 2.0,
            g1 * g2 + f * np.conj(g0),
            -(np.conj(g0)) ** 2 / 4.0,
        ])

    def _F_double_prime(self, z: complex, g: np.ndarray) -> float:
        """F''(z) = Re(g_0 z) + |g_1 + g_2* z|"""
        g0, g1, g2 = g
        return float(np.real(g0 * z) + abs(g1 + np.conj(g2) * z))

    def maximize_over_beta(
        self, g: np.ndarray, C_const: float
    ) -> Tuple[float, np.ndarray]:
        """
        Maximise F''(z) = Re(g_0 z) + |g_1 + g_2* z| over |z|=1.

        Algorithm
        ---------
        1. Binary-search for f* = max F''(z) using the polynomial p_f(z).
        2. Recover z* from the unit-circle roots of p_{f*}(z).
        3. Derive θ_1 from the phase of (g_1 + g_2* z*).
        4. Set θ_2 = -θ_0 - θ_1 and β_a = θ_a / 3.

        Returns
        -------
        E_max    : float
        beta_opt : np.ndarray, shape (3,)
        """
        g0, g1, g2 = g
        k = self.k

        # Upper bound for binary search
        f_max_bound = abs(g0) + abs(g1) + abs(g2) + 1e-10

        # Handle degenerate case: all g are (nearly) zero
        if f_max_bound < 1e-12:
            return float(C_const), np.zeros(k)

        # Binary search for f*
        # A candidate f is achievable iff p_f has a root z* on the unit
        # circle with F''(z*) ≈ f.
        f_lo = 0.0
        f_hi = f_max_bound
        tol = 1e-9
        best_f = 0.0
        best_z = 1.0 + 0j

        for _ in range(80):
            f_mid = (f_lo + f_hi) / 2.0
            coeffs = self._poly_pf(f_mid, g)

            # Skip if leading coefficient is (near) zero — polynomial degenerates
            if abs(coeffs[0]) < 1e-14:
                # Try a small perturbation; if truly zero g0, handle separately
                # When g0=0, p_f(z) is degree 2 (z^0 and z^1 terms only)
                # Actually coeffs[0]=0 means degree reduces; handle via fallback
                # Just evaluate candidate roots from the remaining coefficients
                try:
                    roots = np.roots(coeffs[1:])
                except np.linalg.LinAlgError:
                    f_hi = f_mid
                    continue
            else:
                try:
                    roots = np.roots(coeffs)
                except np.linalg.LinAlgError:
                    f_hi = f_mid
                    continue

            # Check if any root lies on the unit circle (|z| ≈ 1)
            achievable = False
            for z in roots:
                if abs(abs(z) - 1.0) < 1e-4:
                    fz = self._F_double_prime(z, g)
                    if abs(fz - f_mid) < 1e-3 * (f_mid + 1e-10):
                        achievable = True
                        if fz > best_f:
                            best_f = fz
                            best_z = z
                        break

            if achievable:
                f_lo = f_mid
                best_f = max(best_f, f_mid)
            else:
                f_hi = f_mid

            if f_hi - f_lo < tol:
                break

        # Final sweep: evaluate at all roots of p_{f*} with refined f*
        f_star = (f_lo + f_hi) / 2.0
        coeffs_star = self._poly_pf(f_star, g)
        try:
            if abs(coeffs_star[0]) < 1e-14:
                roots_star = np.roots(coeffs_star[1:])
            else:
                roots_star = np.roots(coeffs_star)
        except np.linalg.LinAlgError:
            roots_star = np.array([best_z])

        # Pick best unit-circle root
        for z in roots_star:
            if abs(abs(z) - 1.0) < 1e-3:
                fz = self._F_double_prime(z, g)
                if fz > best_f:
                    best_f = fz
                    best_z = z

        # Also do a direct grid scan on unit circle to guard against
        # cases where the polynomial approach misses the global maximum
        n_scan = 360
        for angle in np.linspace(0, 2 * np.pi, n_scan, endpoint=False):
            z_candidate = np.exp(1j * angle)
            fz = self._F_double_prime(z_candidate, g)
            if fz > best_f:
                best_f = fz
                best_z = z_candidate

        # Recover θ from z* = exp(iθ_0)
        theta_0 = float(np.angle(best_z))

        # θ_1 is determined by the phase of (g_1 + g_2* z*)
        inner = g1 + np.conj(g2) * best_z
        if abs(inner) < 1e-14:
            theta_1 = 0.0
        else:
            theta_1 = float(np.angle(inner))

        theta_2 = -theta_0 - theta_1  # Σ θ_a = 0

        # β_a = θ_a / 3  (choosing β̄ = 0)
        beta_opt = np.array([theta_0, theta_1, theta_2]) / 3.0

        E_max = float(C_const + best_f)
        return E_max, beta_opt

    # ------------------------------------------------------------------
    # γ optimisation
    # ------------------------------------------------------------------

    def _g_from_rho_phi_batch(self, rho_phi_batch, m: int):
        """
        Extract C_const and g from a batch of phi-basis rho matrices.

        Parameters
        ----------
        rho_phi_batch : array (m, k^2, k^2) on device
        m : int  number of edges

        Returns
        -------
        C_const : float
        g       : np.ndarray shape (k,), complex128
        """
        xp = self._xp
        k = self.k
        h0 = (k - 1) / k
        h1 = -1.0 / k

        C_const = float(h0 * m)
        for row_i, col_i in self._idx_C_const:
            vals = rho_phi_batch[:, row_i, col_i]
            C_const += float(2.0 * h1 * xp.sum(vals).real)

        rows1 = [r for r, _, _ in self._idx_g]
        rows2 = [r for _, r, _ in self._idx_g]
        cols  = [c for _, _, c in self._idx_g]
        elems1 = rho_phi_batch[:, rows1, cols].T   # (k, m)
        elems2 = rho_phi_batch[:, rows2, cols].T   # (k, m)
        sums = (elems1 + elems2).sum(axis=1)        # (k,) on device

        if self._use_gpu:
            sums_np = cp.asnumpy(sums)
        else:
            sums_np = np.asarray(sums)

        return C_const, 2.0 * h1 * sums_np

    def optimize_gamma(
        self,
        graph: nx.Graph,
        n_grid: int = 50,
    ) -> Tuple[float, np.ndarray, float]:
        """
        Two-stage grid search over γ ∈ [0, π).

        Fast path: compute rho for all edges AND all γ values at once via
        ``compute_rho_batch_gamma``, then extract g coefficients in batch.

        Returns
        -------
        gamma_opt : float
        beta_opt  : np.ndarray, shape (3,)
        E_opt     : float
        """
        xp = self._xp
        k2 = self.k ** 2

        def _eval_gammas(gammas_np):
            """Return list of (E_max, beta) for each gamma in gammas_np."""
            # Batch all rhos: (ng, m, k^2, k^2)
            rho_all = self.sim.compute_rho_batch_gamma(gammas_np)
            ng, m = rho_all.shape[:2]

            # Batch phi-basis transform: (ng, m, k^2, k^2)
            # rho_phi[g,e] = U2† @ rho[g,e] @ U2
            # xp.matmul broadcasts over leading dims
            rho_phi_all = xp.matmul(
                self._U2H_dev[None, None],
                xp.matmul(rho_all, self._U2_dev[None, None])
            )  # (ng, m, k^2, k^2)

            results = []
            for gi in range(ng):
                C_const, g = self._g_from_rho_phi_batch(rho_phi_all[gi], m)
                E_max, beta = self.maximize_over_beta(g, C_const)
                results.append((E_max, beta))
            return results

        # ---- Stage 1 ----
        gammas1 = np.linspace(0.0, np.pi, n_grid, endpoint=False)
        results1 = _eval_gammas(gammas1)
        best_idx = max(range(n_grid), key=lambda i: results1[i][0])
        best_E1 = results1[best_idx][0]

        # ---- Stage 2: refine around best_idx ----
        lo_idx = max(0, best_idx - 1)
        hi_idx = min(n_grid - 1, best_idx + 1)
        gammas2 = np.linspace(gammas1[lo_idx], gammas1[hi_idx], n_grid, endpoint=True)
        results2 = _eval_gammas(gammas2)
        best_idx2 = max(range(n_grid), key=lambda i: results2[i][0])
        best_E2 = results2[best_idx2][0]

        if best_E2 >= best_E1:
            return float(gammas2[best_idx2]), results2[best_idx2][1], float(best_E2)
        else:
            return float(gammas1[best_idx]), results1[best_idx][1], float(best_E1)
