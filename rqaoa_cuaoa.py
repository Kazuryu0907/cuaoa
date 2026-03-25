"""
RQAOA (Recursive QAOA) using CUAOA for MaxCut.

アルゴリズム概要:
  1. 現在のグラフ W で QAOA を実行して最適パラメータを取得
  2. 状態ベクトルから <Z_i> と <Z_i Z_j> を計算
  3. |<Z_i Z_j>| が最大のエッジ (i*, j*) を選択
  4. Z_i* = c * Z_j* に固定 (c = sign(<Z_i* Z_j*>))
  5. グラフを縮約して node i* を除去
  6. n ≤ n_cutoff になるまで繰り返し
  7. 残った小問題をブルートフォースで解き、後ろ向き代入で全解を復元

参考実装: /workspace/data/open_qaoa_test/exp_val_gpu.py, example_gpu_rqaoa.py
"""

import time
from itertools import product as iproduct

import numpy as np
import pycuaoa


# ---------------------------------------------------------------------------
# 期待値計算
# ---------------------------------------------------------------------------

def compute_expectations(sv: np.ndarray, n: int):
    """
    状態ベクトルから <Z_i> と <Z_i Z_j> を全ペア計算する。

    Parameters
    ----------
    sv : np.ndarray, shape (2**n,), complex
    n  : int  qubit 数

    Returns
    -------
    exp_z   : np.ndarray (n,)        <Z_i>
    corr_zz : np.ndarray (n, n)      <Z_i Z_j>
    """
    probs = np.abs(sv) ** 2  # (2^n,)

    # ビット行列を一括生成: bit_matrix[s, k] = (s >> (n-1-k)) & 1
    indices = np.arange(2 ** n, dtype=np.int64)
    bit_matrix = ((indices[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1).astype(np.float64)

    Z = 1.0 - 2.0 * bit_matrix  # (2^n, n)  0->+1, 1->-1

    # <Z_i> = Σ_s P(s) Z_s[i]
    exp_z = probs @ Z  # (n,)

    # <Z_i Z_j> = Σ_s P(s) Z_s[i] Z_s[j]  → matmul で全ペアを一括計算
    weighted_Z = probs[:, None] * Z  # (2^n, n)
    corr_zz = weighted_Z.T @ Z       # (n, n)

    return exp_z, corr_zz


# ---------------------------------------------------------------------------
# グラフ縮約
# ---------------------------------------------------------------------------

def reduce_graph(W: np.ndarray, elim: int, ref: int, c: int):
    """
    node `elim` を Z_elim = c * Z_ref に固定して除去する。

    MaxCut Hamiltonian は H ∝ Σ_{i<j} W[i,j] Z_i Z_j なので、
    Z_elim = c * Z_ref を代入すると:
      W[elim, k] * Z_elim * Z_k  →  c * W[elim, k] * Z_ref * Z_k

    つまり W'[ref, k] += c * W[elim, k] (k ≠ elim, ref)

    Parameters
    ----------
    W    : np.ndarray (n, n)  対称重み行列
    elim : int  除去するノードのローカルインデックス
    ref  : int  残すノードのローカルインデックス
    c    : ±1  相関の符号

    Returns
    -------
    W_reduced : np.ndarray (n-1, n-1)
    keep      : list[int]  残したローカルインデックスの一覧 (長さ n-1)
    """
    n = len(W)
    W_new = W.copy()

    for k in range(n):
        if k != elim and k != ref:
            W_new[ref, k] += c * W_new[elim, k]
            W_new[k, ref] += c * W_new[k, elim]

    keep = [i for i in range(n) if i != elim]
    W_reduced = W_new[np.ix_(keep, keep)]
    return W_reduced, keep


# ---------------------------------------------------------------------------
# ブルートフォース
# ---------------------------------------------------------------------------

def brute_force_maxcut(W: np.ndarray):
    """
    小グラフの MaxCut をブルートフォースで解く。

    Returns
    -------
    x_pm1     : np.ndarray (n,)  ±1 割り当て
    cut_value : float
    """
    n = len(W)
    best_val = -np.inf
    best_x = None

    for bits in iproduct([0, 1], repeat=n):
        x = np.array([1 - 2 * b for b in bits], dtype=np.float64)
        # MaxCut = Σ_{i<j} W[i,j] * (1 - x[i]*x[j]) / 2
        val = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                val += W[i, j] * (1.0 - x[i] * x[j]) / 2.0
        if val > best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val


def maxcut_value(W: np.ndarray, x: np.ndarray) -> float:
    """±1 割り当て x の MaxCut 値を計算する。"""
    n = len(W)
    val = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            val += W[i, j] * (1.0 - x[i] * x[j]) / 2.0
    return val


# ---------------------------------------------------------------------------
# メインの RQAOA
# ---------------------------------------------------------------------------

def rqaoa_cuaoa(
    W: np.ndarray,
    depth: int = 1,
    n_cutoff: int = 5,
    verbose: bool = True,
):
    """
    RQAOA (Recursive QAOA) for MaxCut using CUAOA.

    Parameters
    ----------
    W        : np.ndarray (n, n)  対称重み行列
    depth    : int                QAOA 回路の深さ p
    n_cutoff : int                ブルートフォースに切り替えるノード数
    verbose  : bool

    Returns
    -------
    x_pm1      : np.ndarray (n,)  各ノードの ±1 割り当て
    cut_value  : float            MaxCut 値
    history    : list[dict]       各ステップの詳細情報
    """
    n_orig = len(W)
    assert n_orig > n_cutoff, f"n={n_orig} must be > n_cutoff={n_cutoff}"
    assert depth >= 1

    W_curr = W.astype(np.float64).copy()
    active_nodes = list(range(n_orig))   # active_nodes[i] = 元のノードインデックス
    substitutions = {}                   # {elim_orig: (ref_orig, c)} 順序付き
    history = []

    t_total = time.perf_counter()
    step = 0

    while len(active_nodes) > n_cutoff:
        n_curr = len(active_nodes)
        t_step = time.perf_counter()

        if verbose:
            print(f"\n--- Step {step}  (n={n_curr}) ---")

        # ---- QAOA 最適化 ----
        t0 = time.perf_counter()
        sim = pycuaoa.CUAOA(W_curr, depth=depth)
        h = pycuaoa.create_handle(n_curr)
        result = sim.optimize(h)
        t_opt = time.perf_counter() - t0

        params = result.parameters
        if verbose:
            print(f"  optimize : {t_opt*1e3:.0f} ms  iter={result.iteration}"
                  f"  n_evals={result.n_evals}")
            print(f"  betas  = {[f'{b:.4f}' for b in params.betas]}")
            print(f"  gammas = {[f'{g:.4f}' for g in params.gammas]}")

        # ---- 状態ベクトル → 期待値 ----
        t0 = time.perf_counter()
        sv = sim.statevector(h)
        exp_z, corr_zz = compute_expectations(sv, n_curr)
        t_exp = time.perf_counter() - t0
        h.destroy()

        if verbose:
            print(f"  statevec + expval : {t_exp*1e3:.0f} ms")
            print(f"  <Z_i> range : [{exp_z.min():.4f}, {exp_z.max():.4f}]")

        # ---- 最強相関エッジを探す ----
        # エッジが存在するペアのみ対象
        best_corr = -1.0
        best_i, best_j = -1, -1

        for i in range(n_curr):
            for j in range(i + 1, n_curr):
                if abs(W_curr[i, j]) > 0:
                    v = abs(corr_zz[i, j])
                    if v > best_corr:
                        best_corr = v
                        best_i, best_j = i, j

        # エッジがない場合は全ペアから選ぶ (disconnected graph のフォールバック)
        if best_i == -1:
            for i in range(n_curr):
                for j in range(i + 1, n_curr):
                    v = abs(corr_zz[i, j])
                    if v > best_corr:
                        best_corr = v
                        best_i, best_j = i, j

        c = 1 if corr_zz[best_i, best_j] >= 0 else -1
        orig_elim = active_nodes[best_i]
        orig_ref  = active_nodes[best_j]

        if verbose:
            print(f"  eliminate node {orig_elim} = {c:+d} * node {orig_ref}"
                  f"  |corr|={best_corr:.4f}")

        # ---- 代入を記録 ----
        substitutions[orig_elim] = (orig_ref, c)

        # ---- グラフ縮約 ----
        W_curr, keep_local = reduce_graph(W_curr, best_i, best_j, c)
        active_nodes = [active_nodes[k] for k in keep_local]

        t_step_total = time.perf_counter() - t_step
        history.append({
            "step": step,
            "n_before": n_curr,
            "eliminated": orig_elim,
            "ref": orig_ref,
            "c": c,
            "corr": float(corr_zz[best_i, best_j]),
            "abs_corr": float(best_corr),
            "t_opt_ms": t_opt * 1e3,
            "t_exp_ms": t_exp * 1e3,
            "t_step_ms": t_step_total * 1e3,
        })
        step += 1

    # ---- ブルートフォース ----
    n_bf = len(active_nodes)
    if verbose:
        print(f"\n--- Brute force (n={n_bf}) ---")

    t0 = time.perf_counter()
    x_reduced, _ = brute_force_maxcut(W_curr)
    t_bf = time.perf_counter() - t0

    if verbose:
        print(f"  brute force : {t_bf*1e3:.0f} ms")

    # ---- 後ろ向き代入で全解を復元 ----
    x_orig = np.zeros(n_orig, dtype=np.float64)
    for i, orig_idx in enumerate(active_nodes):
        x_orig[orig_idx] = x_reduced[i]

    # 除去順の逆順で代入
    for elim_node in reversed(list(substitutions.keys())):
        ref_node, c = substitutions[elim_node]
        x_orig[elim_node] = c * x_orig[ref_node]

    # ---- MaxCut 値を計算 ----
    cut_value = maxcut_value(W, x_orig)

    t_elapsed = time.perf_counter() - t_total
    if verbose:
        print(f"\n=== RQAOA done  total={t_elapsed*1e3:.0f} ms  cut={cut_value:.4f} ===")

    return x_orig, cut_value, history


# ---------------------------------------------------------------------------
# デモ
# ---------------------------------------------------------------------------

def _demo_small():
    """5 ノード MaxCut でブルートフォースと比較。"""
    print("=" * 60)
    print("Demo: 5-node MaxCut (RQAOA vs brute force)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 5
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.6:
                w = rng.uniform(0.5, 2.0)
                W[i, j] = W[j, i] = w

    print(f"\nGraph W (n={n}):")
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] > 0:
                print(f"  ({i},{j}) w={W[i,j]:.3f}")

    x_bf, cut_bf = brute_force_maxcut(W)
    print(f"\nBrute force: cut={cut_bf:.4f}  x={x_bf.astype(int).tolist()}")

    x_rq, cut_rq, hist = rqaoa_cuaoa(W, depth=1, n_cutoff=3, verbose=True)
    print(f"\nRQAOA:       cut={cut_rq:.4f}  x={x_rq.astype(int).tolist()}")
    print(f"Approx ratio: {cut_rq / cut_bf:.4f}")


def _demo_medium():
    """10 ノード MaxCut のベンチマーク。"""
    print("\n" + "=" * 60)
    print("Demo: 10-node random MaxCut (RQAOA)")
    print("=" * 60)

    rng = np.random.default_rng(123)
    n = 10
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.5:
                w = rng.uniform(0.5, 2.0)
                W[i, j] = W[j, i] = w

    x_rq, cut_rq, hist = rqaoa_cuaoa(W, depth=2, n_cutoff=5, verbose=True)

    print(f"\nRQAOA: cut={cut_rq:.4f}")
    print("\nStep history:")
    print(f"  {'step':>4}  {'n':>3}  {'elim':>5}  {'ref':>4}  {'c':>2}  {'|corr|':>7}  {'t_opt(ms)':>10}")
    for h in hist:
        print(f"  {h['step']:>4}  {h['n_before']:>3}  {h['eliminated']:>5}  {h['ref']:>4}"
              f"  {h['c']:>+2}  {h['abs_corr']:>7.4f}  {h['t_opt_ms']:>10.1f}")


if __name__ == "__main__":
    _demo_small()
    _demo_medium()
