"""
CUAOA を使った QAOA サンプルスクリプト
- MaxCut 問題を解く
- GPU 動作の確認とデバッグ情報の出力
"""

import time
import numpy as np
import pycuaoa
from pycuaoa import utils as pycuaoa_utils


# ─────────────────────────────────────────────
# 1. GPU デバッグ: 利用可能な CUDA デバイスを確認
# ─────────────────────────────────────────────
def print_gpu_info():
    print("=" * 60)
    print("GPU デバッグ情報")
    print("=" * 60)
    devices = pycuaoa_utils.get_cuda_devices_info()
    if not devices:
        print("[WARNING] CUDA デバイスが見つかりません")
        return
    for dev in devices:
        print(f"  デバイス ID   : {dev.id}")
        print(f"  デバイス名    : {dev.name}")
        print(f"  総メモリ (MB) : {dev.memory / 1024**2:.1f}")
        print()


# ─────────────────────────────────────────────
# 2. 問題定義 (MaxCut on a 5-node graph)
# ─────────────────────────────────────────────
def make_maxcut_graph():
    """5 ノードの MaxCut 問題 (正方形 + 対角線)"""
    W = np.array([
        [0, 1, 0, 0, 1],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 0, 0, 1, 0],
    ], dtype=np.float64)
    return W


# ─────────────────────────────────────────────
# 3. メインの QAOA ルーティン
# ─────────────────────────────────────────────
def run_qaoa(W: np.ndarray, depth: int = 3, num_shots: int = 1024):
    n = W.shape[0]
    print("=" * 60)
    print(f"QAOA 実行: ノード数={n}, 深さ={depth}")
    print("=" * 60)

    # GPU ハンドルを作成 (GPU メモリを確保)
    print("[GPU] ハンドルを作成中...")
    t0 = time.perf_counter()
    handle = pycuaoa.create_handle(n)
    t1 = time.perf_counter()
    print(f"[GPU] ハンドル作成完了 ({(t1 - t0)*1e3:.2f} ms)")

    # CUAOA インスタンスを作成
    print("\n[QAOA] インスタンスを初期化中...")
    sim = pycuaoa.CUAOA(W, depth=depth)
    print(f"  ノード数  : {sim.get_num_nodes()}")
    print(f"  深さ (p)  : {sim.get_depth()}")
    params = sim.get_parameters()
    print(f"  初期 betas : {params.betas}")
    print(f"  初期 gammas: {params.gammas}")

    # ─── 初期状態の期待値 ───
    print("\n[GPU] 初期期待値を計算中...")
    t0 = time.perf_counter()
    ev_initial = sim.expectation_value(handle)
    t1 = time.perf_counter()
    print(f"  初期期待値 : {ev_initial:.6f}  ({(t1 - t0)*1e3:.2f} ms)")

    # ─── 状態ベクトル取得 ───
    print("\n[GPU] 状態ベクトルを取得中...")
    t0 = time.perf_counter()
    sv = sim.statevector(handle)
    t1 = time.perf_counter()
    probs = np.abs(sv) ** 2
    top5_idx = np.argsort(probs)[-5:][::-1]
    print(f"  状態ベクトルサイズ: {sv.shape}  ({(t1 - t0)*1e3:.2f} ms)")
    print(f"  確率の合計 (GPU 正確性チェック): {probs.sum():.8f}  ← 1.0 に近いはず")
    print(f"  上位 5 ビット列:")
    for idx in top5_idx:
        bitstr = format(idx, f"0{n}b")
        print(f"    |{bitstr}⟩  P={probs[idx]:.4f}")

    # ─── 勾配計算 ───
    print("\n[GPU] 勾配を計算中...")
    t0 = time.perf_counter()
    grads, ev_grad = sim.gradients(handle)
    t1 = time.perf_counter()
    print(f"  勾配計算時の期待値: {ev_grad:.6f}  ({(t1 - t0)*1e3:.2f} ms)")
    print(f"  ∂E/∂beta  : {grads.betas}")
    print(f"  ∂E/∂gamma : {grads.gammas}")

    # ─── L-BFGS 最適化 ───
    print("\n[GPU] L-BFGS 最適化を実行中...")
    lbfgs_params = pycuaoa.LBFGSParameters(
        max_iterations=200,
        epsilon=1e-6,
    )
    t0 = time.perf_counter()
    result = sim.optimize(handle, lbfgs_params)
    t1 = time.perf_counter()
    opt_time = t1 - t0

    print(f"  最適化完了 ({opt_time*1e3:.2f} ms)")
    print(f"  反復回数    : {result.iteration}")
    print(f"  関数評価数  : {result.n_evals}")
    opt_params = result.parameters
    print(f"  最適 betas  : {opt_params.betas}")
    print(f"  最適 gammas : {opt_params.gammas}")

    # ─── 最適化後の期待値 ───
    ev_opt = sim.expectation_value(handle)
    print(f"\n[GPU] 最適化後の期待値: {ev_opt:.6f}")
    print(f"  改善量 (初期→最適): {ev_initial:.4f} → {ev_opt:.4f}")

    # ─── サンプリング ───
    print(f"\n[GPU] {num_shots} ショットをサンプリング中...")
    t0 = time.perf_counter()
    sample_set = sim.sample(handle, num_shots=num_shots, seed=42)
    t1 = time.perf_counter()
    print(f"  サンプリング完了 ({(t1 - t0)*1e3:.2f} ms)")
    energies = sample_set.energies()
    print(f"  エネルギー最小値 : {energies.min():.4f}")
    print(f"  エネルギー平均値 : {energies.mean():.4f}")
    print(f"  エネルギー最大値 : {energies.max():.4f}")
    print(f"  サンプル期待値   : {sample_set.expectation_value():.6f}")

    # ─── 最頻出ビット列 ───
    samples = sample_set.samples()
    unique, counts = np.unique(samples, axis=0, return_counts=True)
    top_idx = np.argsort(counts)[-3:][::-1]
    print(f"\n  上位 3 ビット列 (全 {len(unique)} ユニーク):")
    for i in top_idx:
        bitstr = "".join(str(int(b)) for b in unique[i])
        cut_val = compute_cut(unique[i].astype(bool), W)
        print(f"    [{bitstr}]  出現回数={counts[i]}  カット値={cut_val:.0f}")

    return handle, sim, result, ev_opt


# ─────────────────────────────────────────────
# 4. ブルートフォースとの比較
# ─────────────────────────────────────────────
def run_brute_force(W: np.ndarray, handle):
    n = W.shape[0]
    print("\n" + "=" * 60)
    print("ブルートフォース解法 (GPU 並列)")
    print("=" * 60)
    bf = pycuaoa.BruteFroce(W)
    t0 = time.perf_counter()
    bf_result = bf.solve(handle)
    t1 = time.perf_counter()
    print(f"  ブルートフォース完了 ({(t1 - t0)*1e3:.2f} ms)")
    bf_samples = bf_result.samples()
    bf_energies = bf_result.energies()
    best_idx = np.argmin(bf_energies)
    best_bits = bf_samples[best_idx]
    bitstr = "".join(str(int(b)) for b in best_bits)
    print(f"  最適解 : [{bitstr}]")
    print(f"  最適エネルギー: {bf_energies[best_idx]:.4f}")
    print(f"  全 {2**n} 通り中の最小値")
    return bf_energies[best_idx]


# ─────────────────────────────────────────────
# 5. ユーティリティ
# ─────────────────────────────────────────────
def compute_cut(bits: np.ndarray, W: np.ndarray) -> float:
    """ビット列から MaxCut 値を計算"""
    n = len(bits)
    cut = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if bits[i] != bits[j]:
                cut += W[i, j]
    return cut


# ─────────────────────────────────────────────
# 6. from_map API のデモ
# ─────────────────────────────────────────────
def run_from_map_demo(handle):
    print("\n" + "=" * 60)
    print("from_map API デモ (カスタム問題定義)")
    print("=" * 60)
    # 3 ノード MaxCut: (0,1), (1,2), (0,2) の完全グラフ
    terms = {
        (0, 1): 1.0,
        (1, 2): 1.0,
        (0, 2): 1.0,
    }
    print(f"  問題: 項 = {terms}")
    sim = pycuaoa.CUAOA.from_map(3, terms, depth=2)

    # 別のハンドルを作成 (3 ノード用)
    h3 = pycuaoa.create_handle(3)
    result = sim.optimize(h3)
    ev = sim.expectation_value(h3)
    print(f"  最適化後の期待値: {ev:.6f}")
    print(f"  反復回数: {result.iteration}")
    h3.destroy()
    print("  [GPU] 3 ノード用ハンドルを解放")


# ─────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # GPU 情報を表示
    print_gpu_info()

    # 問題を設定
    W = make_maxcut_graph()
    print("隣接行列 W:\n", W, "\n")

    # QAOA を実行
    handle, sim, result, ev_qaoa = run_qaoa(W, depth=3, num_shots=2048)

    # ブルートフォースとの比較
    ev_bf = run_brute_force(W, handle)

    # 近似比
    print("\n" + "=" * 60)
    print("結果サマリー")
    print("=" * 60)
    approx_ratio = abs(ev_qaoa) / abs(ev_bf) if ev_bf != 0 else float("nan")
    print(f"  QAOA 期待値       : {ev_qaoa:.6f}")
    print(f"  ブルートフォース最小: {ev_bf:.6f}")
    print(f"  近似比 (QAOA/BF)  : {approx_ratio:.4f}")

    # from_map デモ
    run_from_map_demo(handle)

    # GPU リソースを解放
    print("\n[GPU] メインハンドルを解放...")
    handle.destroy()
    print("[GPU] 完了")
