"""
core/density_matrix.py
======================
Level-1 QAOA 2-body marginal density matrix ρ_{uv}.

Algorithm (Bravyi et al. 2022, Section 4):

    η ← exp(iγ C_{uv}) |+><+|^{⊗2} exp(-iγ C_{uv})
    for w in neighbours(u) ∪ neighbours(v) \\ {u,v}:
        η ← E_w(η)
    ρ_{uv} = η

where

    E_w(η) = (1/k) Σ_{a∈Z_k} D_w(a) η D_w(a)†

    D_w(a)|c,d> = exp(+iγ ĥ_{uw}((c-a) mod k)
                      + iγ ĥ_{vw}((d-a) mod k)) |c,d>

Note: the correct sign is +iγ (not −iγ as sometimes written in derivations).
This follows from factoring the full n-qudit marginal:
  ρ_{uv}[cd,c'd'] ∝ exp(iγ(ĥ_{uv}[(c-d)]−ĥ_{uv}[(c'-d')]))
                     × Π_{w∈W} Σ_a exp(iγ(ĥ_{uw}[(c-a)]+ĥ_{vw}[(d-a)]
                                         −ĥ_{uw}[(c'-a)]−ĥ_{vw}[(d'-a)]))
where each factor in the product equals E_w applied to the running η
with D_w using +iγ phase.

The matrix ρ_{uv} has shape (k², k²) and represents the 2-qudit
marginal in the standard basis |0>, …, |k-1>.

GPU/CPU transparency
--------------------
Set ``use_gpu=True`` (default) to attempt CuPy; falls back to NumPy
if CuPy is unavailable.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    _CUPY_AVAILABLE = False

from .hamiltonian import MaxKCutHamiltonian


class DensityMatrixSimulator:
    """
    GPU-accelerated (or CPU) computation of the Level-1 QAOA 2-body
    marginal density matrix ρ_{uv}.

    Complexity per pair: O(k^5 (d_u + d_v)) where d_u, d_v are degrees.
    Batch GPU mode: O(m × k^5 × max_W) for all edges simultaneously.

    Parameters
    ----------
    hamiltonian : MaxKCutHamiltonian
    use_gpu : bool
        Use CuPy if available; otherwise falls back to NumPy silently.
    """

    def __init__(
        self,
        hamiltonian: MaxKCutHamiltonian,
        use_gpu: bool = True,
    ) -> None:
        self.ham = hamiltonian
        self.k = hamiltonian.k

        # GPU is only beneficial when tensors are large enough to amortise
        # kernel-launch overhead.  For k=3 (9×9 matrices) NumPy is faster;
        # GPU wins for k≥5 where batch tensors exceed ~1M elements.
        k = hamiltonian.k
        n_edges = hamiltonian.graph.number_of_edges()
        _gpu_worth = (k ** 2) ** 2 * n_edges >= 50_000  # heuristic threshold

        if use_gpu and _CUPY_AVAILABLE and _gpu_worth:
            self.xp = cp
            self._use_gpu = True
        else:
            self.xp = np
            self._use_gpu = False

        # Pre-compute initial rho0 and per-edge neighbor h_hat tables
        # so they are available for batched GPU computation.
        self._precompute_edge_tables()

    def _precompute_edge_tables(self) -> None:
        """
        Precompute per-edge neighbor h_hat tables for batched rho computation.

        Stores (as GPU/CPU arrays):
          _edges          : list[(u,v)]  length m
          _h_hat_uv       : (m, k)       h_hat for each edge pair
          _h_hat_uw_table : (m, max_W, k)  h_hat[u,w] for each edge's neighbors
          _h_hat_vw_table : (m, max_W, k)  h_hat[v,w] for each edge's neighbors
          _max_W          : int
        """
        xp = self.xp
        k = self.k
        graph = self.ham.graph
        self._edges = list(graph.edges())
        m = len(self._edges)

        # Neighbour sets for each edge
        neighbor_lists = []
        max_W = 0
        for u, v in self._edges:
            W = sorted((set(graph.neighbors(u)) | set(graph.neighbors(v))) - {u, v})
            neighbor_lists.append(W)
            max_W = max(max_W, len(W))
        self._max_W = max_W

        # Build numpy tables, then send to device once
        h_hat_uv_np = np.zeros((m, k), dtype=np.complex128)
        h_hat_uw_np = np.zeros((m, max_W, k), dtype=np.complex128)
        h_hat_vw_np = np.zeros((m, max_W, k), dtype=np.complex128)

        for e, ((u, v), W_list) in enumerate(zip(self._edges, neighbor_lists)):
            h_hat_uv_np[e] = self.ham.get_h_hat(u, v)
            for pos, w in enumerate(W_list):
                h_hat_uw_np[e, pos] = self.ham.get_h_hat(u, w)
                h_hat_vw_np[e, pos] = self.ham.get_h_hat(v, w)
        # Rows beyond len(W_list) remain 0 → exp(0)=1 → identity channel ✓

        self._h_hat_uv_dev = xp.asarray(h_hat_uv_np)       # (m, k)
        self._h_hat_uw_dev = xp.asarray(h_hat_uw_np)        # (m, max_W, k)
        self._h_hat_vw_dev = xp.asarray(h_hat_vw_np)        # (m, max_W, k)

        # Lookup: edge tuple → index in self._edges
        self._edge_index = {(u, v): e for e, (u, v) in enumerate(self._edges)}

        # Pre-build index arrays (reused across gammas)
        self._c_idx = xp.arange(k, dtype=xp.int32)
        self._d_idx = xp.arange(k, dtype=xp.int32)
        self._a_idx = xp.arange(k, dtype=xp.int32)
        self._cd_mod = (self._c_idx[:, None] - self._d_idx[None, :]) % k  # (k,k)
        # ca_mod[a,c] = (c-a)%k,  da_mod[a,d] = (d-a)%k
        self._ca_mod = (self._c_idx[None, :] - self._a_idx[:, None]) % k  # (k,k)
        self._da_mod = (self._d_idx[None, :] - self._a_idx[:, None]) % k  # (k,k)

        # Shared rho0 = |+><+|^⊗2  (k^2 × k^2)
        plus = xp.ones(k, dtype=xp.complex128) / np.sqrt(k)
        plus2 = xp.outer(plus, plus)
        self._rho0 = xp.outer(plus2.ravel(), plus2.ravel().conj())  # (k^2, k^2)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_rho_batch(self, gamma: float):
        """
        Compute ρ_{uv} for ALL edges simultaneously on GPU (or CPU).

        Returns
        -------
        eta : array, shape (m, k^2, k^2), dtype complex128
              On GPU (CuPy) when use_gpu=True.

        This is the fast path used by AngleOptimizer; all m edge
        density matrices are computed in a single vectorised pass.
        """
        xp = self.xp
        k = self.k
        k2 = k * k
        m = len(self._edges)

        # --- Init: η[e] = exp(iγ C_{uv}) |+><+|^⊗2 exp(-iγ C_{uv}) ---
        # phase[e, c, d] = exp(iγ h_hat_uv[e, (c-d)%k])
        h_hat_cd = self._h_hat_uv_dev[:, self._cd_mod]   # (m, k, k)
        phase_uv = xp.exp(1j * gamma * h_hat_cd).reshape(m, k2)  # (m, k^2)

        # η[e,i,j] = phase[e,i] * rho0[i,j] * conj(phase[e,j])
        eta = phase_uv[:, :, None] * self._rho0[None] * phase_uv[:, None, :].conj()
        # shape: (m, k^2, k^2)

        # --- Apply E_w channels: E_w(η)_{ij} = η_{ij} × eff_w_{ij} ---
        # _h_hat_uw_dev shape: (m, max_W, k)
        for pos in range(self._max_W):
            huw = self._h_hat_uw_dev[:, pos, :]   # (m, k)  ← correct axis
            hvw = self._h_hat_vw_dev[:, pos, :]   # (m, k)

            phase_uw = xp.exp(+1j * gamma * huw[:, self._ca_mod])   # (m, k_a, k_c)
            phase_vw = xp.exp(+1j * gamma * hvw[:, self._da_mod])   # (m, k_a, k_d)

            d_diag = (phase_uw[:, :, :, None] * phase_vw[:, :, None, :]).reshape(m, k, k2)
            # (m, k, k^2)

            # eff_w = (1/k) d_diag^T @ conj(d_diag): (m, k^2, k) @ (m, k, k^2)
            eff_w = xp.matmul(d_diag.transpose(0, 2, 1), d_diag.conj()) / k
            eta = eta * eff_w   # element-wise

        return eta  # (m, k^2, k^2)  — stays on GPU/CPU device

    def compute_rho_batch_gamma(self, gammas):
        """
        Compute ρ_{uv} for ALL edges AND ALL γ values simultaneously.

        This is the highest-throughput path: for n_gamma gamma values and m
        edges we launch a single set of GPU kernels on tensors of shape
        (n_gamma, m, k^2, k^2).  The GPU can parallelise over the full
        n_gamma × m outer product, giving large occupancy even for small k.

        Parameters
        ----------
        gammas : array-like, shape (n_gamma,)

        Returns
        -------
        eta : array, shape (n_gamma, m, k^2, k^2), dtype complex128
              Lives on GPU when use_gpu=True.
        """
        xp = self.xp
        k = self.k
        k2 = k * k
        m = len(self._edges)
        gammas_dev = xp.asarray(gammas, dtype=xp.float64)  # (n_gamma,)
        ng = len(gammas_dev)

        # --- Init η: shape (n_gamma, m, k^2, k^2) ---
        # h_hat_cd[e, c, d] = h_hat_uv[e, (c-d)%k],  shape (m, k, k)
        h_hat_cd = self._h_hat_uv_dev[:, self._cd_mod]   # (m, k, k)

        # phase_uv[ng, m, k^2] = exp(iγ h_hat_cd[m, k, k])
        # Broadcasting: gammas (ng,1,1,1) × h_hat_cd (1,m,k,k) → (ng,m,k,k)
        phase_uv = xp.exp(
            1j * gammas_dev[:, None, None, None] * h_hat_cd[None, :, :, :]
        ).reshape(ng, m, k2)  # (ng, m, k^2)

        # η[ng,m,i,j] = phase[ng,m,i] * rho0[i,j] * conj(phase[ng,m,j])
        eta = (
            phase_uv[:, :, :, None]
            * self._rho0[None, None, :, :]
            * phase_uv[:, :, None, :].conj()
        )  # (ng, m, k^2, k^2)

        # --- Apply E_w channels ---
        # Key insight: E_w(η)_{ij} = η_{ij} × eff_w_{ij}
        # where eff_w_{ij} = (1/k) Σ_a d_diag[a,i] × conj(d_diag[a,j])
        #                   = (1/k) [d_diag^T @ conj(d_diag)]_{ij}
        # This replaces the O(k^5) intermediate tensor with a batched matmul
        # followed by an element-wise multiply (no k^5 tensor needed).
        #
        # _h_hat_uw_dev shape: (m, max_W, k)
        # huw_ca[m, pos, k_a, k_c] = h_hat_uw[m, pos, (c-a)%k]
        huw_ca = self._h_hat_uw_dev[:, :, self._ca_mod]   # (m, max_W, k, k)
        hvw_da = self._h_hat_vw_dev[:, :, self._da_mod]   # (m, max_W, k, k)

        for pos in range(self._max_W):
            # huw_pos[m, k_a, k_c], hvw_pos[m, k_a, k_d]
            huw_pos = huw_ca[:, pos, :, :]   # (m, k, k)
            hvw_pos = hvw_da[:, pos, :, :]   # (m, k, k)

            # phase_uw[ng, m, a, c] = exp(+iγ h_hat_uw[m,(c-a)%k])
            phase_uw = xp.exp(
                +1j * gammas_dev[:, None, None, None] * huw_pos[None, :, :, :]
            )  # (ng, m, k_a, k_c)
            phase_vw = xp.exp(
                +1j * gammas_dev[:, None, None, None] * hvw_pos[None, :, :, :]
            )  # (ng, m, k_a, k_d)

            # d_diag[ng, m, k_a, k^2]
            d_diag = (phase_uw[:, :, :, :, None] * phase_vw[:, :, :, None, :]).reshape(
                ng, m, k, k2
            )

            # eff_w[ng, m, k^2, k^2] = (1/k) d_diag^T @ conj(d_diag)
            # (ng, m, k^2, k) @ (ng, m, k, k^2)  → no k^5 tensor!
            eff_w = xp.matmul(d_diag.transpose(0, 1, 3, 2), d_diag.conj()) / k
            # (ng, m, k^2, k^2)

            # E_w(η) = η ⊙ eff_w  (element-wise multiply)
            eta = eta * eff_w

        return eta  # (ng, m, k^2, k^2)

    def compute_rho(self, u: int, v: int, gamma: float) -> np.ndarray:
        """
        Compute ρ_{uv} for a given γ.

        Parameters
        ----------
        u, v  : int   Vertex indices (must be different).
        gamma : float  Phase separation angle γ.

        Returns
        -------
        rho : np.ndarray, shape (k^2, k^2), dtype complex128
              Returned as a NumPy array regardless of backend.
        """
        xp = self.xp
        k = self.k
        k2 = k * k

        # Use precomputed tables if (u,v) is a known edge; fall back otherwise.
        edge_key = (u, v)
        if edge_key in self._edge_index:
            e = self._edge_index[edge_key]
            h_hat_uv = self._h_hat_uv_dev[e]   # (k,) on device

            phase = xp.exp(1j * gamma * h_hat_uv[self._cd_mod]).ravel()  # (k^2,)
            eta = phase[:, None] * self._rho0 * phase[None, :].conj()

            huw_e = self._h_hat_uw_dev[e]   # (max_W, k)
            hvw_e = self._h_hat_vw_dev[e]   # (max_W, k)
            for pos in range(self._max_W):
                eta = self._apply_channel_precomp(eta, huw_e[pos], hvw_e[pos], gamma)
        else:
            # Fallback for non-edge pairs (used in rqaoa._simple_gamma_search)
            h_hat_uv = xp.asarray(self.ham.get_h_hat(u, v))
            phase = xp.exp(1j * gamma * h_hat_uv[self._cd_mod]).ravel()
            eta = phase[:, None] * self._rho0 * phase[None, :].conj()
            neighbours_uv = (
                set(self.ham.graph.neighbors(u))
                | set(self.ham.graph.neighbors(v))
            ) - {u, v}
            for w in neighbours_uv:
                eta = self._apply_channel(eta, u, v, w, gamma)

        if self._use_gpu:
            return cp.asnumpy(eta)
        return np.asarray(eta)

    def compute_expectation(
        self,
        rho: np.ndarray,
        u: int,
        v: int,
        beta: np.ndarray,
    ) -> np.ndarray:
        """
        Compute M_{uv}(b) = μ_{uv}(Π_{uv}(b)) for each b ∈ Z_k.

        μ_{uv}(O) = Tr[ ρ_{uv} · B(-β)^{⊗2} O B(β)^{⊗2} ]

        where Π_{uv}(b) = Σ_{a∈Z_k} |a, (a+b)%k><a, (a+b)%k|.

        Parameters
        ----------
        rho  : np.ndarray, shape (k^2, k^2)
        u, v : int  (unused — kept for API symmetry)
        beta : np.ndarray, shape (k,)

        Returns
        -------
        M : np.ndarray, shape (k,), real
        """
        xp = self.xp
        k = self.k

        rho_xp = xp.asarray(rho)
        beta_xp = xp.asarray(beta)

        # Build B(β) and B(-β)
        B_pos = self._build_B(beta_xp)   # (k, k)
        B_neg = self._build_B(-beta_xp)  # (k, k)

        # B^{⊗2} = kron(B, B), shape (k^2, k^2)
        B2_pos = xp.kron(B_pos, B_pos)
        B2_neg = xp.kron(B_neg, B_neg)

        # Rotated density matrix: ρ̃ = B(β)^{⊗2} ρ B(-β)^{⊗2}
        rho_rot = B2_pos @ rho_xp @ B2_neg   # (k^2, k^2)

        # M(b) = Tr[ρ̃ · Π(b)]
        # Π(b) = Σ_a |a*(k)+(a+b)%k><a*(k)+(a+b)%k|  (diagonal projector)
        # So Tr[ρ̃ · Π(b)] = Σ_a ρ̃[a*k + (a+b)%k, a*k + (a+b)%k]
        a_arr = xp.arange(k, dtype=xp.int64)
        M = xp.empty(k, dtype=xp.complex128)
        for b in range(k):
            indices = a_arr * k + (a_arr + b) % k   # shape (k,)
            M[b] = xp.sum(rho_rot[indices, indices])

        result = M.real
        if self._use_gpu:
            return cp.asnumpy(result)
        return np.asarray(result)

    def _build_B(self, beta: "xp.ndarray") -> "xp.ndarray":
        """
        Build B(β) = Σ_a exp(iβ_a) |φ_a><φ_a|, shape (k, k).

        |φ_a>_b = ω^{ab} / √k   (columns of the DFT matrix / √k)
        Z|φ_a> = |φ_{a+1}>
        """
        xp = self.xp
        k = self.k
        omega = xp.exp(xp.asarray(2j * np.pi / k))

        a_arr = xp.arange(k, dtype=xp.int64)
        b_arr = xp.arange(k, dtype=xp.int64)

        # phi_matrix[a, b] = ω^{ab} / √k,  shape (k, k)
        ab = a_arr[:, None] * b_arr[None, :]   # (k, k)
        phi_matrix = omega ** ab / np.sqrt(k)  # (k, k)  rows = |φ_a>

        # B = Σ_a exp(iβ_a) |φ_a><φ_a|
        #   = phi_matrix.T @ diag(exp(iβ)) @ phi_matrix.conj()
        # (phi_matrix[a,:] = <b|φ_a>, so outer product is |φ_a><φ_a| in std basis)
        weights = xp.exp(1j * beta)   # (k,)
        # B[i,j] = Σ_a w_a * φ_a[i] * conj(φ_a[j])
        #        = Σ_a w_a * phi_matrix[a,i] * conj(phi_matrix[a,j])
        B = xp.einsum("a,ai,aj->ij", weights, phi_matrix, phi_matrix.conj())
        return B

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_channel_precomp(
        self,
        eta: "xp.ndarray",
        h_hat_uw: "xp.ndarray",
        h_hat_vw: "xp.ndarray",
        gamma: float,
    ) -> "xp.ndarray":
        """
        Apply E_w using precomputed h_hat arrays (shape (k,) on device).
        Identical math to _apply_channel but avoids dict lookups.
        """
        xp = self.xp
        k = self.k

        phase_uw = xp.exp(+1j * gamma * h_hat_uw[self._ca_mod])  # (k_a, k_c)
        phase_vw = xp.exp(+1j * gamma * h_hat_vw[self._da_mod])  # (k_a, k_d)

        d_diag = (phase_uw[:, :, None] * phase_vw[:, None, :]).reshape(k, k * k)
        # eff_w = (1/k) d_diag^T @ conj(d_diag): (k^2, k) @ (k, k^2)
        eff_w = (d_diag.T @ d_diag.conj()) / k
        return eta * eff_w

    def _apply_channel(
        self,
        eta: "xp.ndarray",
        u: int,
        v: int,
        w: int,
        gamma: float,
    ) -> "xp.ndarray":
        """
        Apply the channel E_w(η) = (1/k) Σ_{a∈Z_k} D_w(a) η D_w(a)†.

        D_w(a)|c,d> = exp(-iγ ĥ_{uw}((c-a) mod k)
                          - iγ ĥ_{vw}((d-a) mod k)) |c,d>

        Vectorised over a using broadcasting:
        d_diag has shape (k, k^2); result = (1/k) Σ_a outer-product update.

        Complexity: O(k^5)
        """
        xp = self.xp
        k = self.k

        h_hat_uw = xp.asarray(self.ham.get_h_hat(u, w))  # (k,)
        h_hat_vw = xp.asarray(self.ham.get_h_hat(v, w))  # (k,)

        # Pre-compute index arrays
        c_idx = xp.arange(k, dtype=xp.int64)   # (k,)
        d_idx = xp.arange(k, dtype=xp.int64)   # (k,)
        a_idx = xp.arange(k, dtype=xp.int64)   # (k,)

        # For each a: d_diag[a, c*k+d] = exp(-iγ ĥ_{uw}((c-a)%k) - iγ ĥ_{vw}((d-a)%k))
        # shape of intermediate arrays: (k_a, k_c, k_d) → reshape to (k, k^2)
        # (c-a) mod k:  shape (k_a, k_c)
        ca_mod = (c_idx[None, :] - a_idx[:, None]) % k   # (k, k)
        # (d-a) mod k:  shape (k_a, k_d)
        da_mod = (d_idx[None, :] - a_idx[:, None]) % k   # (k, k)

        # Phase contributions from u-w and v-w edges.
        # Note: the sign is +iγ (not -iγ as naively written in the spec).
        # The correct channel formula is derived from the factored marginal:
        #   rho_{uv}[cd,c'd'] ∝ Π_w Σ_a exp(+iγ(ĥ_{uw}[(c-a)]+ĥ_{vw}[(d-a)]
        #                                         -ĥ_{uw}[(c'-a)]-ĥ_{vw}[(d'-a)]))
        # which matches E_w with D_w(a)[cd] = exp(+iγ ĥ_{uw}[(c-a)] + iγ ĥ_{vw}[(d-a)]).
        phase_uw = xp.exp(+1j * gamma * h_hat_uw[ca_mod])  # (k_a, k_c)
        phase_vw = xp.exp(+1j * gamma * h_hat_vw[da_mod])  # (k_a, k_d)

        # Full diagonal: d_diag[a, c*k+d] = phase_uw[a,c] * phase_vw[a,d]
        # shape: (k_a, k_c, k_d) → (k_a, k^2)
        d_diag = (phase_uw[:, :, None] * phase_vw[:, None, :]).reshape(k, k * k)
        # shape: (k, k^2)

        # E_w(η) = (1/k) Σ_a D_w(a) η D_w(a)†
        # Since D_w(a) is diagonal with diagonal d_diag[a,:]:
        #   [D_w(a) η D_w(a)†]_{ij} = d_diag[a,i] * η_{ij} * conj(d_diag[a,j])
        # Vectorised over a using broadcasting:
        #   contrib[a, i, j] = d_diag[a, i] * η[i, j] * conj(d_diag[a, j])
        # Shape: (k, k^2, k^2); sum over axis 0 and divide by k.
        d_col = d_diag[:, :, None]              # (k, k^2, 1)
        d_row = d_diag[:, None, :].conj()       # (k, 1, k^2)
        contrib = d_col * eta[None, :, :] * d_row   # (k, k^2, k^2)
        result = contrib.sum(axis=0) / k             # (k^2, k^2)

        return result
