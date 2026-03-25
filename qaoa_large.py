"""
CUAOA 大規模ベンチマーク
- ランダム重み付き MaxCut を複数サイズで実行
- GPU メモリ・実行時間・近似比をサイズごとに計測
"""

import time
import numpy as np
import pycuaoa
from pycuaoa import utils as pycuaoa_utils


RNG = np.random.default_rng(42)


def random_maxcut(n: int, edge_prob: float = 0.5) -> np.ndarray:
    """ランダム重み付き MaxCut の隣接行列を生成"""
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if RNG.random() < edge_prob:
                w = RNG.uniform(0.1, 1.0)
                W[i, j] = w
                W[j, i] = w
    return W


def brute_force_optimal(W: np.ndarray) -> float:
    """全探索で最適カット値を返す (小サイズ用)"""
    n = W.shape[0]
    best = 0.0
    for mask in range(1 << n):
        cut = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if ((mask >> i) & 1) != ((mask >> j) & 1):
                    cut += W[i, j]
        if cut > best:
            best = cut
    return best


def run_benchmark(n: int, depth: int, num_shots: int = 512):
    print(f"\n{'='*60}")
    print(f"  ノード数={n:2d}, 深さ p={depth}, shots={num_shots}")
    print(f"{'='*60}")

    W = random_maxcut(n)
    edges = int((W > 0).sum() // 2)
    print(f"  エッジ数: {edges}  (密度 {edges / (n*(n-1)/2):.1%})")

    # ハンドル作成
    t0 = time.perf_counter()
    handle = pycuaoa.create_handle(n)
    t_handle = (time.perf_counter() - t0) * 1e3

    # CUAOA 初期化
    sim = pycuaoa.CUAOA(W, depth=depth)

    # 初期期待値
    t0 = time.perf_counter()
    ev0 = sim.expectation_value(handle)
    t_ev0 = (time.perf_counter() - t0) * 1e3

    # 最適化
    lbfgs = pycuaoa.LBFGSParameters(max_iterations=300, epsilon=1e-7)
    t0 = time.perf_counter()
    result = sim.optimize(handle, lbfgs)
    t_opt = (time.perf_counter() - t0) * 1e3

    ev_opt = sim.expectation_value(handle)

    # サンプリング
    t0 = time.perf_counter()
    samples = sim.sample(handle, num_shots=num_shots, seed=0)
    t_sample = (time.perf_counter() - t0) * 1e3

    energies = samples.energies()
    best_sample_energy = energies.min()

    # 状態ベクトル (サイズ確認)
    t0 = time.perf_counter()
    sv = sim.statevector(handle)
    t_sv = (time.perf_counter() - t0) * 1e3
    prob_sum = float((np.abs(sv) ** 2).sum())

    print(f"  状態ベクトルサイズ : 2^{n} = {2**n:,} 要素  ({sv.nbytes/1024:.1f} KB)")
    print(f"  確率の合計チェック : {prob_sum:.8f}")
    print(f"  ハンドル作成       : {t_handle:8.2f} ms")
    print(f"  初期期待値         : {t_ev0:8.2f} ms  → {ev0:.4f}")
    print(f"  L-BFGS 最適化      : {t_opt:8.2f} ms  → {ev_opt:.4f}  ({result.iteration} 反復, {result.n_evals} 評価)")
    print(f"  サンプリング       : {t_sample:8.2f} ms  最良サンプル={best_sample_energy:.4f}")
    print(f"  状態ベクトル取得   : {t_sv:8.2f} ms")

    # ブルートフォース比較 (16 ノード以下のみ)
    approx_ratio = None
    if n <= 16:
        bf_opt = brute_force_optimal(W)
        approx_ratio = abs(ev_opt) / bf_opt if bf_opt > 0 else float("nan")
        print(f"  ブルートフォース最適: {bf_opt:.4f}  近似比={approx_ratio:.4f}")
    else:
        print(f"  ブルートフォース: スキップ (n>{16})")

    handle.destroy()

    return {
        "n": n,
        "depth": depth,
        "t_handle_ms": t_handle,
        "t_ev0_ms": t_ev0,
        "t_opt_ms": t_opt,
        "t_sample_ms": t_sample,
        "t_sv_ms": t_sv,
        "ev_initial": ev0,
        "ev_opt": ev_opt,
        "best_sample": best_sample_energy,
        "iterations": result.iteration,
        "n_evals": result.n_evals,
        "approx_ratio": approx_ratio,
        "sv_bytes": sv.nbytes,
    }


if __name__ == "__main__":
    # GPU 情報
    print("=" * 60)
    print("GPU デバッグ情報")
    print("=" * 60)
    for dev in pycuaoa_utils.get_cuda_devices_info():
        print(f"  [{dev.id}] {dev.name}  メモリ={dev.memory/1024**3:.2f} GB")

    # ベンチマーク設定: (ノード数, 深さ)
    configs = [
        (8,  2),
        (10, 2),
        (12, 3),
        (14, 3),
        (16, 3),
        (18, 4),
        (20, 4),
        (22, 4),
        (24, 5),
    ]

    results = []
    for n, p in configs:
        try:
            r = run_benchmark(n, p, num_shots=1024)
            results.append(r)
        except Exception as e:
            print(f"\n[ERROR] n={n}, p={p}: {e}")
            break

    # サマリーテーブル
    print("\n\n" + "=" * 90)
    print("ベンチマーク サマリー")
    print("=" * 90)
    print(f"{'n':>4} {'p':>3} {'状態ベクトル':>12} {'最適化(ms)':>12} {'反復':>6} {'初期EV':>10} {'最適EV':>10} {'近似比':>8}")
    print("-" * 90)
    for r in results:
        sv_size = f"2^{r['n']}={2**r['n']:,}"
        ar = f"{r['approx_ratio']:.4f}" if r["approx_ratio"] is not None else "  N/A "
        print(f"{r['n']:>4} {r['depth']:>3} {sv_size:>12} {r['t_opt_ms']:>12.1f} {r['iterations']:>6} {r['ev_initial']:>10.4f} {r['ev_opt']:>10.4f} {ar:>8}")
