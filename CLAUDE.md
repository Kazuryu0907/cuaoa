# CUAOA 開発メモ

## ビルド手順

```bash
# 1. conda 環境をそのまま使う (custatevec, lbfgs が入っている)
# 2. maturin と patchelf をインストール
uv tool install maturin
pip install patchelf  # または uv tool install patchelf

# 3. ビルド (CONDA_PREFIX を明示的に指定)
CONDA_PREFIX=/root/miniconda3/envs/conda_env \
  maturin build --release --interpreter /root/miniconda3/envs/conda_env/bin/python3.11

# 4. インストール
/root/miniconda3/envs/conda_env/bin/pip install \
  target/wheels/pycuaoa-0.1.0-cp311-cp311-manylinux_2_39_x86_64.whl --force-reinstall

# 5. 実行
/root/miniconda3/envs/conda_env/bin/python qaoa_example.py
```

## 注意点・既知の問題

### Python バージョン
- システムの Python は 3.14 だが PyO3 0.23.5 は 3.13 までしか対応していない
- **conda 環境の Python 3.11 を使うこと**: `/root/miniconda3/envs/conda_env/bin/python3.11`
- `conda run` は uv 管理の Python に対して pip が使えないので直接パスを指定する

### CUDA 13.1 との互換性
- `cuaoa/internal/src/wrapper/device_info.cpp` の `cudaGetDeviceProperties_v2` が CUDA 13.1 で未定義エラー
- **修正済み**: `cudaGetDeviceProperties` に変更 (L28)

### Python API の実際の型・属性名

`Parameters` オブジェクト (`.get_parameters()` の戻り値):
- `params.betas` — プロパティ (list), `betas()` と呼ぶとエラー
- `params.gammas` — プロパティ (list)

`OptimizeResult` オブジェクト (`.optimize()` の戻り値):
- `result.iteration` — 反復回数 (int)
- `result.n_evals` — 関数評価回数 (int)
- `result.parameters` — 最適パラメータ (Parameters)
- ~~`result.fx_hist`~~ — **存在しない** (README には記載あるが実装なし)

`PyCudaDevice` オブジェクト (`utils.get_cuda_devices_info()` の戻り値):
- `dev.id` — デバイス番号 (int)
- `dev.name` — デバイス名 (str)
- `dev.memory` — 総メモリ bytes (int)
- ~~`dev.device`~~ / ~~`dev.total_memory`~~ / ~~`dev.free_memory`~~ — **存在しない**

`SampleSet` の `energies()` と `samples()` は **メソッド呼び出し** (属性ではない):
- `sample_set.energies()` → numpy array
- `sample_set.samples()` → 2D numpy bool array

`pycuaoa.utils` のインポート:
```python
# NG: from pycuaoa.utils import get_cuda_devices_info
# OK:
from pycuaoa import utils as pycuaoa_utils
pycuaoa_utils.get_cuda_devices_info()
```

## 動作確認済み環境

| 項目 | バージョン |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 (8 GB) |
| CUDA | 13.1 |
| custatevec | 1.12.0 |
| Python | 3.11 (conda_env) |
| PyO3 | 0.23.5 |
| maturin | 1.12.6 |

## ベンチマーク結果 (RTX 4060 8GB, ランダム重み付き MaxCut, 密度~50%)

| n  | p | 状態ベクトル       | 最適化時間 | 反復 | 初期EV   | 最適EV   | 近似比 |
|----|---|--------------------|-----------|------|----------|----------|--------|
|  8 | 2 | 2^8  = 256         |    30 ms  |  14  |  -5.59   |  -6.54   | 0.859  |
| 10 | 2 | 2^10 = 1,024       |    49 ms  |  15  |  -7.79   |  -8.71   | 0.848  |
| 12 | 3 | 2^12 = 4,096       |   275 ms  |  22  | -13.41   | -14.56   | 0.923  |
| 14 | 3 | 2^14 = 16,384      |   219 ms  |  20  | -15.53   | -16.70   | 0.914  |
| 16 | 3 | 2^16 = 65,536      |   406 ms  |  17  | -22.59   | -24.08   | 0.911  |
| 18 | 4 | 2^18 = 262,144     |   1.2 s   |  27  | -22.09   | -24.12   |   —    |
| 20 | 4 | 2^20 = 1,048,576   |   1.6 s   |  21  | -35.69   | -37.62   |   —    |
| 22 | 4 | 2^22 = 4,194,304   |    23 s   |  25  | -40.48   | -42.62   |   —    |
| 24 | 5 | 2^24 = 16,777,216  |   121 s   |  24  | -44.22   | -47.49   |   —    |

全サイズで確率の合計 = 1.00000000 (GPU 演算の正確性 OK)

**スケーリングの傾向:**
- n=20 まで数秒以内、実用的な速度
- n=22 で状態ベクトルが 64 MB → 最適化が急増 (~23s)
- n=24 で 256 MB、完走可能だが最適化に ~2 分
- p を上げると近似比が向上 (p=2: ~0.85 → p=3: ~0.91)

各操作の単体レイテンシ (n=20, p=4):
- 期待値計算: ~11 ms
- サンプリング 1024 shots: ~10 ms
- 状態ベクトル取得: ~35 ms

## RTX 4060 (8 GB) での最大 qubit 数

実測による上限 (線形チェーングラフ, depth=1):

| n  | 状態ベクトルメモリ | `expectation_value()` | `statevector()` |
|----|-------------------|----------------------|-----------------|
| 27 | 1 GB              | OK (~676 ms)         | OK              |
| 28 | 2 GB              | OK (~1340 ms), ev=0.0 が出ることあり (精度不明) | **フリーズ** (2GB GPU→CPU 転送) |
| 29 | 4 GB              | **segfault** (OOM)   | —               |

**実用上限:**
- `expectation_value` のみ使う場合: **n=28** (ただし結果の信頼性要確認)
- `statevector()` も使う場合: **n=27**
- `statevector()` を n=28 以上で呼ぶとプロセスがフリーズするので絶対に呼ばない

期待値計算の単体レイテンシ (線形チェーン, depth=1):
- n=24: ~85 ms (128 MB)
- n=25: ~169 ms (256 MB)
- n=26: ~328 ms (512 MB)
- n=27: ~676 ms (1 GB)
- n=28: ~1340 ms (2 GB) — expval のみ

## RQAOA 実装 (rqaoa_cuaoa.py)

CUAOA を使った RQAOA (Recursive QAOA) の独自実装。

### アルゴリズム
1. 現在のグラフ W で CUAOA を実行し最適パラメータを取得
2. `sim.statevector(h)` → `compute_expectations()` で <Z_i> と <Z_i Z_j> を計算
3. |<Z_i Z_j>| が最大のエッジ (i*, j*) を選択
4. Z_i* = c × Z_j* に固定 (c = sign(<Z_i* Z_j*>))
5. グラフを縮約: W'[ref, k] += c × W[elim, k] でノード elim を除去
6. n ≤ n_cutoff になるまで繰り返す
7. 残った小問題をブルートフォースで解き、後ろ向き代入で全解を復元

### QAOA vs RQAOA 性能比較 (RTX 4060, ランダム重み付き MaxCut, 密度50%)

| n  | depth | QAOA 近似比 | RQAOA 近似比 | QAOA 時間 | RQAOA 時間 |
|----|-------|------------|-------------|----------|-----------|
|  8 |  2    | 1.000      | 0.867       |  56 ms   |  140 ms   |
| 10 |  2    | 1.000      | 0.861       |  31 ms   |  163 ms   |
| 12 |  3    | 1.000      | 0.826       |  81 ms   |  428 ms   |
| 14 |  3    | 1.000      | 0.895       | 101 ms   |  639 ms   |
| 16 |  3    | 1.000      | 0.909       | 264 ms   |  928 ms   |

**考察:**
- QAOA (exact statevector + 1024 shots) はこのサイズで常に最適解を発見
- RQAOA は平均近似比 0.83〜0.91 で QAOA より質が下がる傾向
- RQAOA は QAOA より 3〜5 倍遅い (各ステップで QAOA を繰り返すため)
- RQAOA のメリットは大規模問題 (n > 27) への拡張性: 小問題に再帰的に縮約するため理論上は n に制限がない

### 使い方
```python
from rqaoa_cuaoa import rqaoa_cuaoa
import numpy as np

W = ...  # (n, n) 対称重み行列
x, cut_value, history = rqaoa_cuaoa(W, depth=2, n_cutoff=5, verbose=True)
```

戻り値:
- `x` : ±1 割り当て (np.ndarray)
- `cut_value` : MaxCut 値
- `history` : 各ステップの詳細 (dict のリスト)

### 注意点
- `compute_expectations()` は `statevector()` 経由なので n ≤ 27 が安全上限
- n_cutoff は 3〜6 推奨 (大きくするとブルートフォースが遅くなる)

## cuTensorNet 実装との比較 (../cuquantum_test/)

`../cuquantum_test/` に cuQuantum NetworkState API を使った QAOA/RQAOA 実装あり。

| 実装 | ファイル | Python | バックエンド |
|---|---|---|---|
| CUAOA (statevector) | `rqaoa_cuaoa.py` | conda_env 3.11 | custatevec |
| cuTensorNet (TN 収縮) | `../cuquantum_test/rqaoa_cutn.py` | base 3.13 | NetworkState |

**n=10 RQAOA 速度比較:**
- CUAOA: **~166 ms** (約 500 倍速い)
- cuTN: **~79 s** (各評価ごとに TN 再構築のオーバーヘッド)

cuTN の `update_tensor_operator` は収縮プランのキャッシュを試みるが、実測では fresh build より **1.6 倍遅い**（per-call オーバーヘッドが構築節約分 ~9ms を大幅に上回るため）。使わないこと。
cuTN の L1-RQAOA ライトコーン最適化版（`rqaoa_l1_tn.py`）では n=10 で 17s、n=40 で 464s（3-regular）まで実用的。

## rqaoa_maxkcut（MAX-k-CUT 実装）

`rqaoa_maxkcut/` ディレクトリに Bravyi et al. 2020 に基づく MAX-k-CUT 用 L1-RQAOA を実装。

- k=2（MaxCut）と k=3 に対応
- `rqaoa_maxkcut/core/rqaoa.py`: メインの RQAOA ループ（`n_grid` パラメータで γ グリッド点数を指定）
- `experiment_ngrid_k2.py`, `experiment_ngrid_k3.py`: n_grid スイープ実験（推奨 n_grid=20）
- `benchmark_3regular.py`, `benchmark_complete_graph.py`: 3-regular / 完全グラフでの近似比ベンチマーク

**注意:** k=3 で `n_grid` を `optimize_gamma` に渡すには `rqaoa.py` の修正が必要（渡さないと無視される）。

## サンプルスクリプト

`qaoa_example.py` — 5 ノード MaxCut の基本デモ:
- GPU デバッグ情報表示
- 期待値・状態ベクトル・勾配・最適化・サンプリングの全操作
- ブルートフォース解法との比較、`from_map` API のデモ

`qaoa_large.py` — 8〜24 ノードのスケーリングベンチマーク:
- ランダム重み付き MaxCut を複数サイズで実行
- 16 ノード以下はブルートフォースとの近似比を算出

## 重要: bit ordering バグ修正 (2026-05-04)

`rqaoa_cuaoa.compute_expectations` に LSB/MSB 取り違えバグがあり修正した。

- **症状**: `compute_expectations(sv, n)` 内のビット行列が
  `bit_matrix[s, k] = (s >> (n-1-k)) & 1` (MSB-first) で構築されていたが、
  CUAOA (custatevec) の `sim.statevector(h)` は **LSB-first** SV を返す。
  結果として qubit インデックスが反転され (`corr_zz[i, j]` が実質
  `<Z_{n-1-i} Z_{n-1-j}>`)、RQAOA の最大相関エッジ選択が誤エッジを指していた。
- **影響**: 過去の CUAOA p=1 / p=2 RQAOA ベンチ結果はすべて影響あり
  (`benchmark_3regular.py`, `benchmark_p2_density.py`,
  `benchmark_complete_graph.py`, `benchmark_qaoa_vs_rqaoa.py` 等)。
  bit-reversal 対称性のあるグラフ (K_{n,n}, 完全グラフ) では偶然マスクされる。
- **修正**: 1 行 — `np.arange(n - 1, -1, -1)` → `np.arange(n)` (LSB-first)。
- **検証**: n=4 重み非対称グラフで cuquantum_test の qaoa_statevector_np と
  ZZ 相関子 6 ペアすべて一致 (誤差 < 1e-15)。
- **影響範囲の数値例** (n=18, d=3-7, 5 seeds):
  - benchmark_p2_density: ratio 0.79-0.87 → **0.99-1.00**
  - benchmark_3regular (n=8-16): RQAOA p=1/p=2 が 0.81-0.94 → **1.000**
- **教訓**: cuQuantum / custatevec の SV API は LSB-first 規約。
  自前で bit 取り出しをするときは必ず単純な非対称グラフで sanity check すること。
