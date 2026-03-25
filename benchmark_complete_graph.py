"""
完全グラフ (K_n) における QAOA / RQAOA 解の性能評価

比較指標:
  - 近似比 (achieved cut / optimal cut)
  - QAOA: depth p=1, 2, 3
  - RQAOA: depth p=1, 2

条件:
  - 完全グラフ K_n (全エッジ重み=1)
  - 最適 MaxCut = floor(n/2) * ceil(n/2)  (理論値、ブルートフォース確認)
  - n=6~16 はブルートフォース比較, n=20 は比較なし

実行方法:
    cd cuaoa
    /root/miniconda3/envs/conda_env/bin/python benchmark_complete_graph.py
"""

import json
import time

import numpy as np
import pycuaoa

from rqaoa_cuaoa import brute_force_maxcut, rqaoa_cuaoa


# ---------------------------------------------------------------------------
# グラフ生成
# ---------------------------------------------------------------------------

def make_complete_graph(n: int) -> np.ndarray:
    """K_n: 全ての辺が重み1で結ばれた完全グラフ。"""
    W = np.ones((n, n)) - np.eye(n)
    return W


def optimal_cut_complete(n: int) -> float:
    """K_n の MaxCut 最適値 = floor(n/2) * ceil(n/2)."""
    return float((n // 2) * ((n + 1) // 2))


# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------

def run_qaoa(W: np.ndarray, depth: int, n_shots: int = 2048):
    n = len(W)
    t0 = time.perf_counter()

    sim = pycuaoa.CUAOA(W, depth=depth)
    h = pycuaoa.create_handle(n)
    result = sim.optimize(h)
    t_opt = time.perf_counter() - t0

    ss = sim.sample(h, n_shots)
    h.destroy()

    samples = ss.samples().astype(np.float64)
    x_pm1   = 1.0 - 2.0 * samples
    xWx  = np.einsum('si,ij,sj->s', x_pm1, W, x_pm1)
    cuts = (W.sum() - xWx) / 4.0

    t_total = time.perf_counter() - t0
    return float(cuts.mean()), t_total, t_opt, result.n_evals


# ---------------------------------------------------------------------------
# RQAOA
# ---------------------------------------------------------------------------

def run_rqaoa(W: np.ndarray, depth: int, n_cutoff: int = 5):
    t0 = time.perf_counter()
    _, cut, history = rqaoa_cuaoa(W, depth=depth, n_cutoff=n_cutoff, verbose=False)
    t = time.perf_counter() - t0
    return float(cut), t, len(history)


# ---------------------------------------------------------------------------
# 単一インスタンス評価
# ---------------------------------------------------------------------------

def evaluate_instance(
    W: np.ndarray,
    qaoa_depths: list,
    rqaoa_depths: list,
    n_cutoff: int,
    n_shots: int,
    compute_optimal: bool,
):
    n = len(W)
    cut_opt = optimal_cut_complete(n) if compute_optimal else None

    results = {"n": n, "cut_opt": cut_opt, "qaoa": {}, "rqaoa": {}}

    for p in qaoa_depths:
        cut, t_total, t_opt, n_evals = run_qaoa(W, depth=p, n_shots=n_shots)
        results["qaoa"][p] = {
            "cut": cut,
            "ratio": cut / cut_opt if cut_opt else None,
            "t_total_ms": t_total * 1e3,
            "t_opt_ms": t_opt * 1e3,
            "n_evals": n_evals,
        }

    for p in rqaoa_depths:
        cut, t, steps = run_rqaoa(W, depth=p, n_cutoff=n_cutoff)
        results["rqaoa"][p] = {
            "cut": cut,
            "ratio": cut / cut_opt if cut_opt else None,
            "t_ms": t * 1e3,
            "steps": steps,
        }

    return results


# ---------------------------------------------------------------------------
# ベンチマーク設定
# ---------------------------------------------------------------------------

CONFIGS = [
    # (n, compute_optimal)
    (6,  True),
    (8,  True),
    (10, True),
    (12, True),
    (14, True),
    (16, True),
    (20, False),
]

QAOA_DEPTHS  = [1]
RQAOA_DEPTHS = [1]
N_CUTOFF     = 5
N_SHOTS      = 2048
N_TRIALS     = 5   # K_n は 1 インスタンスなので複数回試行して平均


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 80)
    print("Benchmark: QAOA / RQAOA on Complete Graph K_n  (CUAOA backend, avg cut)")
    print(f"  QAOA depths={QAOA_DEPTHS}   RQAOA depths={RQAOA_DEPTHS}")
    print(f"  n_cutoff={N_CUTOFF}  n_shots={N_SHOTS}  n_trials={N_TRIALS}")
    print("  metric: average cut value over shots (not best sample)")
    print("=" * 80)

    all_results = []

    for n, compute_opt in CONFIGS:
        n_edges = n * (n - 1) // 2
        cut_opt = optimal_cut_complete(n) if compute_opt else None

        print(f"\n{'='*60}")
        opt_disp = f"{cut_opt:.0f}" if cut_opt else "N/A"
        print(f"K_{n}  edges={n_edges}  optimal={opt_disp}")
        print(f"{'='*60}")

        W = make_complete_graph(n)
        trial_results = []

        for trial in range(N_TRIALS):
            r = evaluate_instance(
                W,
                qaoa_depths=QAOA_DEPTHS,
                rqaoa_depths=RQAOA_DEPTHS,
                n_cutoff=N_CUTOFF,
                n_shots=N_SHOTS,
                compute_optimal=compute_opt,
            )
            trial_results.append(r)

            parts = []
            for p in QAOA_DEPTHS:
                q = r["qaoa"][p]
                s = f"QAOA={q['cut']:.2f}"
                if q["ratio"]:
                    s += f"({q['ratio']:.3f})"
                parts.append(s)
            for p in RQAOA_DEPTHS:
                rq = r["rqaoa"][p]
                s = f"RQAOA={rq['cut']:.2f}"
                if rq["ratio"]:
                    s += f"({rq['ratio']:.3f})"
                parts.append(s)
            print(f"  trial={trial}  {' | '.join(parts)}")

        # 集計
        summary = {"n": n, "n_edges": n_edges, "cut_opt": cut_opt,
                   "n_trials": N_TRIALS, "qaoa": {}, "rqaoa": {}}

        for p in QAOA_DEPTHS:
            cuts   = [r["qaoa"][p]["cut"] for r in trial_results]
            ratios = [r["qaoa"][p]["ratio"] for r in trial_results
                      if r["qaoa"][p]["ratio"] is not None]
            t_ms   = [r["qaoa"][p]["t_total_ms"] for r in trial_results]
            summary["qaoa"][p] = {
                "cut_mean": float(np.mean(cuts)),
                "cut_std":  float(np.std(cuts)),
                "ratio_mean": float(np.mean(ratios)) if ratios else None,
                "ratio_std":  float(np.std(ratios))  if ratios else None,
                "t_ms_mean":  float(np.mean(t_ms)),
            }

        for p in RQAOA_DEPTHS:
            cuts   = [r["rqaoa"][p]["cut"] for r in trial_results]
            ratios = [r["rqaoa"][p]["ratio"] for r in trial_results
                      if r["rqaoa"][p]["ratio"] is not None]
            t_ms   = [r["rqaoa"][p]["t_ms"] for r in trial_results]
            summary["rqaoa"][p] = {
                "cut_mean": float(np.mean(cuts)),
                "cut_std":  float(np.std(cuts)),
                "ratio_mean": float(np.mean(ratios)) if ratios else None,
                "ratio_std":  float(np.std(ratios))  if ratios else None,
                "t_ms_mean":  float(np.mean(t_ms)),
            }

        all_results.append(summary)

        print(f"  [統計]")
        for p in QAOA_DEPTHS:
            s = summary["qaoa"][p]
            ratio_str = f"  ratio={s['ratio_mean']:.3f}±{s['ratio_std']:.3f}" if s["ratio_mean"] else ""
            print(f"    QAOA  p={p}  cut={s['cut_mean']:.2f}±{s['cut_std']:.2f}{ratio_str}")
        for p in RQAOA_DEPTHS:
            s = summary["rqaoa"][p]
            ratio_str = f"  ratio={s['ratio_mean']:.3f}±{s['ratio_std']:.3f}" if s["ratio_mean"] else ""
            print(f"    RQAOA p={p}  cut={s['cut_mean']:.2f}±{s['cut_std']:.2f}{ratio_str}")

    # サマリテーブル
    print("\n" + "=" * 80)
    print("Summary  (average cut / optimal,  metric=mean over shots & trials)")
    print("=" * 80)
    print(f"{'n':>4}  {'opt':>5}  {'QAOA p=1 ratio':>16}  {'RQAOA p=1 ratio':>17}  {'QAOA t(ms)':>10}  {'RQAOA t(ms)':>11}")
    print("-" * 70)
    for s in all_results:
        q  = s["qaoa"][1]
        rq = s["rqaoa"][1]
        opt_s  = f"{s['cut_opt']:.0f}" if s["cut_opt"] else "N/A"
        qr_s   = f"{q['ratio_mean']:.3f}±{q['ratio_std']:.3f}" if q["ratio_mean"] else f"cut={q['cut_mean']:.2f}"
        rqr_s  = f"{rq['ratio_mean']:.3f}±{rq['ratio_std']:.3f}" if rq["ratio_mean"] else f"cut={rq['cut_mean']:.2f}"
        print(f"{s['n']:>4}  {opt_s:>5}  {qr_s:>16}  {rqr_s:>17}  {q['t_ms_mean']:>10.0f}  {rq['t_ms_mean']:>11.0f}")

    out_path = "../benchmark_complete_graph_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました。")
    return all_results


if __name__ == "__main__":
    run_benchmark()
