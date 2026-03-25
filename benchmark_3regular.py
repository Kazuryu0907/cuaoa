"""
Random 3-Regular グラフにおける QAOA / RQAOA 解の性能評価

比較指標:
  - 近似比 (achieved cut / optimal cut)  ← n <= 16 はブルートフォース最適値と比較
  - QAOA: depth p=1, 2, 3 での近似比とサンプリング時間
  - RQAOA: depth p=1, 2 での近似比と総時間

条件:
  - Unweighted Random 3-Regular グラフ (W[i,j] in {0,1})
  - 複数ランダムシードで平均
  - n=8~20 の 6 サイズ

実行方法:
    cd cuaoa
    /root/miniconda3/envs/conda_env/bin/python benchmark_3regular.py
"""

import json
import time

import networkx as nx
import numpy as np
import pycuaoa

from rqaoa_cuaoa import brute_force_maxcut, rqaoa_cuaoa


# ---------------------------------------------------------------------------
# グラフ生成
# ---------------------------------------------------------------------------

def make_3regular(n: int, seed: int) -> np.ndarray:
    """Random 3-Regular グラフの重み行列 (0/1) を返す。"""
    G = nx.random_regular_graph(3, n, seed=seed)
    W = np.zeros((n, n))
    for u, v in G.edges():
        W[u, v] = W[v, u] = 1.0
    return W


# ---------------------------------------------------------------------------
# QAOA (サンプリングで最良カット値を取得)
# ---------------------------------------------------------------------------

def run_qaoa(W: np.ndarray, depth: int, n_shots: int = 2048):
    """
    CUAOA で QAOA を実行し、最良サンプルのカット値と時間を返す。

    Returns
    -------
    cut_best : float   サンプリング中の最良カット値
    t_total  : float   総実行時間 [s]
    t_opt    : float   最適化時間 [s]
    n_evals  : int     評価回数
    """
    n = len(W)
    t0 = time.perf_counter()

    sim = pycuaoa.CUAOA(W, depth=depth)
    h = pycuaoa.create_handle(n)
    result = sim.optimize(h)
    t_opt = time.perf_counter() - t0

    ss = sim.sample(h, n_shots)
    h.destroy()

    samples = ss.samples().astype(np.float64)  # (n_shots, n), 0/1
    x_pm1   = 1.0 - 2.0 * samples              # 0->+1, 1->-1

    # cut = (total_W - x^T W x) / 4
    xWx  = np.einsum('si,ij,sj->s', x_pm1, W, x_pm1)
    cuts = (W.sum() - xWx) / 4.0

    t_total = time.perf_counter() - t0
    return float(cuts.mean()), t_total, t_opt, result.n_evals


# ---------------------------------------------------------------------------
# RQAOA
# ---------------------------------------------------------------------------

def run_rqaoa(W: np.ndarray, depth: int, n_cutoff: int = 5):
    """
    CUAOA を使った RQAOA を実行し、カット値と時間を返す。
    """
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
    cut_opt = None
    if compute_optimal:
        _, cut_opt = brute_force_maxcut(W)
        cut_opt = float(cut_opt)

    results = {"n": len(W), "cut_opt": cut_opt, "qaoa": {}, "rqaoa": {}}

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
    # (n, n_seeds, compute_optimal)
    (8,  5, True),
    (10, 5, True),
    (12, 5, True),
    (14, 5, True),
    (16, 5, True),
    (20, 3, False),
]

QAOA_DEPTHS  = [1]
RQAOA_DEPTHS = [1]
N_CUTOFF     = 5
N_SHOTS      = 2048


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 80)
    print("Benchmark: QAOA / RQAOA on Random 3-Regular Graphs  (CUAOA backend)")
    print(f"  QAOA depths={QAOA_DEPTHS}   RQAOA depths={RQAOA_DEPTHS}")
    print(f"  n_cutoff={N_CUTOFF}  n_shots={N_SHOTS}")
    print("=" * 80)

    all_summaries = []

    for n, n_seeds, compute_opt in CONFIGS:
        print(f"\n{'='*60}")
        print(f"n={n}  seeds={n_seeds}  brute_force={'yes' if compute_opt else 'no'}")
        print(f"{'='*60}")

        seed_results = []

        for seed in range(n_seeds):
            W = make_3regular(n, seed=seed * 100 + n)
            n_edges = int(np.sum(np.triu(W, 1) > 0))

            r = evaluate_instance(
                W,
                qaoa_depths=QAOA_DEPTHS,
                rqaoa_depths=RQAOA_DEPTHS,
                n_cutoff=N_CUTOFF,
                n_shots=N_SHOTS,
                compute_optimal=compute_opt,
            )
            seed_results.append(r)

            # 1 行サマリ表示
            opt_str = f"opt={r['cut_opt']:.1f}" if r["cut_opt"] else "opt=N/A"

            qaoa_parts = []
            for p in QAOA_DEPTHS:
                q = r["qaoa"][p]
                s = f"p{p}:{q['cut']:.2f}"
                if q["ratio"] is not None:
                    s += f"({q['ratio']:.3f})"
                qaoa_parts.append(s)

            rqaoa_parts = []
            for p in RQAOA_DEPTHS:
                rq = r["rqaoa"][p]
                s = f"p{p}:{rq['cut']:.2f}"
                if rq["ratio"] is not None:
                    s += f"({rq['ratio']:.3f})"
                rqaoa_parts.append(s)

            print(f"  seed={seed} e={n_edges}  {opt_str}"
                  f"  QAOA[{' | '.join(qaoa_parts)}]"
                  f"  RQAOA[{' | '.join(rqaoa_parts)}]")

        # 統計集計
        summary = {"n": n, "n_seeds": n_seeds, "n_edges": n * 3 // 2}

        for p in QAOA_DEPTHS:
            cuts   = [r["qaoa"][p]["cut"] for r in seed_results]
            ratios = [r["qaoa"][p]["ratio"] for r in seed_results
                      if r["qaoa"][p]["ratio"] is not None]
            t_ms   = [r["qaoa"][p]["t_total_ms"] for r in seed_results]
            summary[f"qaoa_p{p}"] = {
                "cut_mean": float(np.mean(cuts)),
                "cut_std":  float(np.std(cuts)),
                "ratio_mean": float(np.mean(ratios)) if ratios else None,
                "ratio_std":  float(np.std(ratios))  if ratios else None,
                "t_ms_mean":  float(np.mean(t_ms)),
            }

        for p in RQAOA_DEPTHS:
            cuts   = [r["rqaoa"][p]["cut"] for r in seed_results]
            ratios = [r["rqaoa"][p]["ratio"] for r in seed_results
                      if r["rqaoa"][p]["ratio"] is not None]
            t_ms   = [r["rqaoa"][p]["t_ms"] for r in seed_results]
            summary[f"rqaoa_p{p}"] = {
                "cut_mean": float(np.mean(cuts)),
                "cut_std":  float(np.std(cuts)),
                "ratio_mean": float(np.mean(ratios)) if ratios else None,
                "ratio_std":  float(np.std(ratios))  if ratios else None,
                "t_ms_mean":  float(np.mean(t_ms)),
            }

        all_summaries.append(summary)

        print(f"\n  [統計]")
        for p in QAOA_DEPTHS:
            s = summary[f"qaoa_p{p}"]
            ratio_str = (f"  ratio={s['ratio_mean']:.3f}±{s['ratio_std']:.3f}"
                         if s["ratio_mean"] is not None else "")
            print(f"    QAOA  p={p}  cut={s['cut_mean']:.2f}±{s['cut_std']:.2f}"
                  f"{ratio_str}  t={s['t_ms_mean']:.0f}ms")
        for p in RQAOA_DEPTHS:
            s = summary[f"rqaoa_p{p}"]
            ratio_str = (f"  ratio={s['ratio_mean']:.3f}±{s['ratio_std']:.3f}"
                         if s["ratio_mean"] is not None else "")
            print(f"    RQAOA p={p}  cut={s['cut_mean']:.2f}±{s['cut_std']:.2f}"
                  f"{ratio_str}  t={s['t_ms_mean']:.0f}ms")

    # ---------------------------------------------------------------------------
    # 全体サマリテーブル
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Summary Table  (近似比の平均±標準偏差 / 時間)")
    print("  ratio = cut / optimal_cut  (optimal = brute-force, n<=16 のみ)")
    print("=" * 80)

    col_w = 22

    header = f"{'n':>4}"
    for p in QAOA_DEPTHS:
        header += f"  {'QAOA p='+str(p):^{col_w}}"
    for p in RQAOA_DEPTHS:
        header += f"  {'RQAOA p='+str(p):^{col_w}}"
    print(header)
    print("-" * (4 + (col_w + 2) * (len(QAOA_DEPTHS) + len(RQAOA_DEPTHS))))

    for s in all_summaries:
        row = f"{s['n']:>4}"
        for p in QAOA_DEPTHS:
            d = s[f"qaoa_p{p}"]
            if d["ratio_mean"] is not None:
                cell = f"{d['ratio_mean']:.3f}±{d['ratio_std']:.3f} {d['t_ms_mean']:.0f}ms"
            else:
                cell = f"cut={d['cut_mean']:.2f} {d['t_ms_mean']:.0f}ms"
            row += f"  {cell:^{col_w}}"
        for p in RQAOA_DEPTHS:
            d = s[f"rqaoa_p{p}"]
            if d["ratio_mean"] is not None:
                cell = f"{d['ratio_mean']:.3f}±{d['ratio_std']:.3f} {d['t_ms_mean']:.0f}ms"
            else:
                cell = f"cut={d['cut_mean']:.2f} {d['t_ms_mean']:.0f}ms"
            row += f"  {cell:^{col_w}}"
        print(row)

    # JSON 保存
    out_path = "../benchmark_3regular_results.json"
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n結果を {out_path} に保存しました。")

    return all_summaries


if __name__ == "__main__":
    run_benchmark()
