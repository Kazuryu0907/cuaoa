"""
QAOA vs RQAOA 性能比較ベンチマーク

比較指標:
  - 近似比 (cut / optimal)
  - 総実行時間
  - 最良サンプルのカット値

テスト条件:
  - ランダム重み付き MaxCut (密度 50%)
  - 各サイズで複数のランダムシード
  - ブルートフォースで最適値を算出 (n<=16)
"""

import time
from itertools import product as iproduct

import numpy as np
import pycuaoa

from rqaoa_cuaoa import rqaoa_cuaoa, brute_force_maxcut, maxcut_value


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def random_maxcut_graph(n: int, density: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.uniform(0.5, 2.0)
                W[i, j] = W[j, i] = w
    return W


def qaoa_best_sample(W: np.ndarray, depth: int, n_shots: int = 1024):
    """
    QAOA を実行し、サンプリングで得た最良カット値を返す。

    Returns
    -------
    cut_value : float
    t_total   : float  実行時間 [s]
    t_opt     : float  最適化時間 [s]
    """
    n = len(W)
    t0 = time.perf_counter()

    sim = pycuaoa.CUAOA(W, depth=depth)
    h = pycuaoa.create_handle(n)
    result = sim.optimize(h)
    t_opt = time.perf_counter() - t0

    ss = sim.sample(h, n_shots)
    h.destroy()

    # サンプルを ±1 に変換してカット値を計算
    samples = ss.samples().astype(np.float64)  # (n_shots, n), bool -> 0/1
    x_pm1 = 1.0 - 2.0 * samples               # 0->+1, 1->-1

    # 全サンプルのカット値を行列演算で一括計算
    # cut = Σ_{i<j} W[i,j] * (1 - x[i]*x[j]) / 2
    #     = (W.sum() - x^T W x) / 4
    xWx = np.einsum('si,ij,sj->s', x_pm1, W, x_pm1)  # (shots,)
    cuts = (W.sum() - xWx) / 4.0

    best_cut = float(cuts.max())
    t_total = time.perf_counter() - t0

    return best_cut, t_total, t_opt


# ---------------------------------------------------------------------------
# 単一インスタンスの比較
# ---------------------------------------------------------------------------

def compare_single(W: np.ndarray, depth: int, n_cutoff: int, n_shots: int = 1024):
    n = len(W)

    # ブルートフォース最適値
    if n <= 20:
        x_bf, cut_opt = brute_force_maxcut(W)
    else:
        cut_opt = None

    # QAOA
    cut_qaoa, t_qaoa, t_qaoa_opt = qaoa_best_sample(W, depth, n_shots)

    # RQAOA
    t0 = time.perf_counter()
    x_rq, cut_rqaoa, hist = rqaoa_cuaoa(W, depth=depth, n_cutoff=n_cutoff, verbose=False)
    t_rqaoa = time.perf_counter() - t0

    return {
        "n": n,
        "cut_opt": cut_opt,
        "cut_qaoa": cut_qaoa,
        "cut_rqaoa": cut_rqaoa,
        "ratio_qaoa": cut_qaoa / cut_opt if cut_opt else None,
        "ratio_rqaoa": cut_rqaoa / cut_opt if cut_opt else None,
        "t_qaoa": t_qaoa,
        "t_rqaoa": t_rqaoa,
        "rqaoa_steps": len(hist),
    }


# ---------------------------------------------------------------------------
# ベンチマーク本体
# ---------------------------------------------------------------------------

def run_benchmark():
    configs = [
        # (n, depth, n_cutoff, n_seeds)
        (8,  2, 4, 5),
        (10, 2, 5, 5),
        (12, 3, 5, 3),
        (14, 3, 5, 3),
        (16, 3, 6, 3),
    ]

    print("=" * 80)
    print("QAOA vs RQAOA Benchmark  (random weighted MaxCut, density=50%)")
    print("=" * 80)

    all_results = []

    for n, depth, n_cutoff, n_seeds in configs:
        print(f"\n--- n={n}  depth={depth}  n_cutoff={n_cutoff}  seeds={n_seeds} ---")

        seed_results = []
        for seed in range(n_seeds):
            W = random_maxcut_graph(n, density=0.5, seed=seed * 100 + n)
            r = compare_single(W, depth=depth, n_cutoff=n_cutoff)
            seed_results.append(r)

            if r["cut_opt"] is not None:
                print(f"  seed={seed}  opt={r['cut_opt']:.3f}"
                      f"  QAOA={r['cut_qaoa']:.3f}({r['ratio_qaoa']:.3f})"
                      f"  RQAOA={r['cut_rqaoa']:.3f}({r['ratio_rqaoa']:.3f})"
                      f"  t_qaoa={r['t_qaoa']*1e3:.0f}ms"
                      f"  t_rqaoa={r['t_rqaoa']*1e3:.0f}ms")
            else:
                print(f"  seed={seed}"
                      f"  QAOA={r['cut_qaoa']:.3f}"
                      f"  RQAOA={r['cut_rqaoa']:.3f}"
                      f"  t_qaoa={r['t_qaoa']*1e3:.0f}ms"
                      f"  t_rqaoa={r['t_rqaoa']*1e3:.0f}ms")

        # 統計
        ratios_qaoa  = [r["ratio_qaoa"]  for r in seed_results if r["ratio_qaoa"]  is not None]
        ratios_rqaoa = [r["ratio_rqaoa"] for r in seed_results if r["ratio_rqaoa"] is not None]
        t_qaoa_avg   = np.mean([r["t_qaoa"]  for r in seed_results]) * 1e3
        t_rqaoa_avg  = np.mean([r["t_rqaoa"] for r in seed_results]) * 1e3

        summary = {
            "n": n, "depth": depth,
            "ratio_qaoa_mean":  np.mean(ratios_qaoa)  if ratios_qaoa  else None,
            "ratio_rqaoa_mean": np.mean(ratios_rqaoa) if ratios_rqaoa else None,
            "t_qaoa_ms":  t_qaoa_avg,
            "t_rqaoa_ms": t_rqaoa_avg,
        }
        all_results.append(summary)

        print(f"  >> QAOA  avg_ratio={summary['ratio_qaoa_mean']:.3f}  avg_time={t_qaoa_avg:.0f}ms"
              if summary["ratio_qaoa_mean"] else
              f"  >> QAOA  avg_time={t_qaoa_avg:.0f}ms")
        print(f"  >> RQAOA avg_ratio={summary['ratio_rqaoa_mean']:.3f}  avg_time={t_rqaoa_avg:.0f}ms"
              if summary["ratio_rqaoa_mean"] else
              f"  >> RQAOA avg_time={t_rqaoa_avg:.0f}ms")

    # ---------------------------------------------------------------------------
    # 結果テーブル
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'n':>4}  {'depth':>5}  {'QAOA ratio':>10}  {'RQAOA ratio':>11}  "
          f"{'QAOA time':>9}  {'RQAOA time':>10}  {'speedup':>7}")
    print("-" * 80)
    for s in all_results:
        r_q  = f"{s['ratio_qaoa_mean']:.3f}"  if s["ratio_qaoa_mean"]  else "  n/a "
        r_rq = f"{s['ratio_rqaoa_mean']:.3f}" if s["ratio_rqaoa_mean"] else "  n/a "
        spd  = f"{s['t_qaoa_ms']/s['t_rqaoa_ms']:.2f}x" if s["t_rqaoa_ms"] > 0 else "  n/a"
        print(f"{s['n']:>4}  {s['depth']:>5}  {r_q:>10}  {r_rq:>11}  "
              f"{s['t_qaoa_ms']:>8.0f}ms  {s['t_rqaoa_ms']:>9.0f}ms  {spd:>7}")


if __name__ == "__main__":
    run_benchmark()
