# rasx-cluster-analyzer

Rigaku のマップ測定用 **rasX** をまとめて読み込み、XRD 回折パターンを **DBSCAN** でクラスタリングし、**PCA で圧縮した特徴を入力として t-SNE** で 2 次元化します。結果は **Plotly** の HTML で出力します。

**HTML の構成:**

- **メイン列（左）**  
  - **上段**: **PCA（PC1 vs PC2）** と **t-SNE（PCA 後）** の 2 パネル。データ座標の縦横比は **1:1**。点の色は **クラスタ（およびノイズ）ごと**。凡例タイトルは **「クラスタ（散布図）」**。  
  - **余白**（区切り）を挟んで **下段**: まず **All profiles**、その下にクラスタ（ノイズ）ごとの **XRD プロファイル**を **別々の Plotly 図**として表示します。デスクトップでは **2 列レイアウト**、狭い画面では **1 列**に戻ります。横軸 **2theta / degree**、縦軸 **Intensity / a.u.**。  
  - `grid.exclude_ranges` を設定している場合でも、**下段の XRD 図は除外前の全 2θ 範囲**を表示します。除外レンジは **グレーの縦帯**で重ね表示されます。  
  - XRD 図の直上に **全 XRD 図へ一括適用する X/Y 軸フォーム**があります。  
  - 各 XRD 図の legend は **図ごとに独立**していて、Plotly 標準のクリック操作で個別トグルできます。
- **解析メタデータ（右サイドバー）**  
  - 右列に **幅約 320px（狭い画面では最大 32vw）** のパネル。**ビューポート高さいっぱい**（`height: 100vh`）にし、`position: sticky; top: 0` で **縦スクロールしても画面上端に張り付いたまま**表示されます。メタデータが長い場合は **サイドバー内だけ**縦スクロールします。内容は生成日時、入力ディレクトリ、出力パス、スペクトル数、クラスタ／ノイズの点数、`config.toml` に基づく **2θ グリッド・前処理・PCA / t-SNE / DBSCAN** の主要パラメータなどです。

表示用の XRD 強度は **除外前の全 2θ グリッド**に対する **行方向の強度正規化後・列方向 `StandardScaler` の前**の値です。一方、クラスタリング側では必要に応じて `grid.exclude_ranges` が適用されます。

**前処理:** 補間後の強度行列に対し、必要なら **2θ マスク**を適用し、その後 **スペクトル（行）ごと**に規格化（既定 **L2**）、続けて **各 2θ ビン（列）ごと**に `StandardScaler` で標準化してからクラスタリング・次元削減に入ります。

## 必要な環境

- Python 3.13 以上
- [uv](https://github.com/astral-sh/uv)（推奨）

## 依存関係（実行時）

`numpy`・`polars`・`scikit-learn`（PCA / DBSCAN / t-SNE）・`plotly`。表形式処理に **pandas は使いません**。

## セットアップ

```sh
uv sync
```

開発用（テスト・Ruff・ty）:

```sh
uv sync --group dev
```

## 使い方

`.rasx` が入ったディレクトリを第 1 引数に渡します。設定ファイルは `-c` / `--config` で指定し、省略した場合は **カレントディレクトリの `./config.toml`** を読みます。

```sh
uv run rasx-cluster-analyzer --config config.toml /path/to/rasx_dir
uv run rasx-cluster-analyzer /path/to/rasx_dir
```

同等の実行例:

```sh
uv run python -m rasx_cluster_analyzer --config config.toml /path/to/rasx_dir
uv run python src/main.py --config config.toml /path/to/rasx_dir
```

ログを詳しくする場合:

```sh
uv run rasx-cluster-analyzer -v --config config.toml /path/to/rasx_dir
```

通常実行時も、読み込み後に `Intensity normalization: l2`（など）が 1 行 INFO で出ます。

### 出力

- 既定: `<rasx_dir>/cluster_map.html`
- `[paths] output_html` で変更可能（相対パスはカレントディレクトリ基準、絶対パスも可）
- HTML は **埋め込み用 Plotly 図**、**複数の XRD Plotly 図**、および **ビューポート高さいっぱいの sticky 右サイドバー（解析メタデータ）**を 1 ページにまとめたものです。Plotly は **CDN** から読み込みます（オフライン閲覧が必要な場合は別途配布方法を検討してください）。

## 処理の流れ（概要）

1. 各 `.rasx` からスペクトルを読み、`config` の **共通 2θ グリッド**上に線形補間して特徴行列（行＝スペクトル、列＝2θ ビン）を作る。  
2. 表示用のために、**除外前の全 2θ グリッド**で行方向の強度規格化を行う。  
3. `grid.exclude_ranges` があれば、クラスタリング用の特徴量に対してだけその 2θ 範囲の列を落とす。  
4. クラスタリング用の特徴量に対して **行ごとの強度規格化**（`[preprocess]`、後述の「強度の規格化」参照）。既定は **L2**（各行ベクトルのノルムを 1 に）。  
5. **StandardScaler** で **列ごと**（各 2θ 位置で全サンプル横断）に平均 0・分散 1。  
6. **同じ行列**に **PCA** を 1 回フィットし、`n_components` 次元まで射影（次元はデータ形状で上限）。  
7. **埋め込み図**: 左は **常に PC1 vs PC2**、右は PCA 射影を入力とした **t-SNE（2D）**（`init='random'`）。`pca.n_components >= 2` なら、左の PCA 図は基本的に **PC1 / PC2 固定**で、`n_components` は主に右の t-SNE 側へ効く。  
8. **DBSCAN** を設定に応じて **高次元特徴**または **t-SNE 2D 座標**に適用しクラスタラベルを得る。  
9. **XRD 図**: まず **All profiles**、その下に各クラスタごとの **2θ–強度** の折線図を **別々の Plotly 図**として描く。ここで描いている強度は **除外前・行正規化後・列標準化前**。`exclude_ranges` は **グレー帯**で示す。ゼロ近傍・対数軸向けに極小正へクリップして **線が途切れない**ようにしている。  
10. XRD 用フォームから、全 XRD 図に対して **X/Y 範囲**と **Y 軸の linear/log** を一括で変更できる。  
11. メタデータ用 HTML を生成し、右の sticky サイドバーに埋め込む。

## 強度の規格化（`[preprocess]`）

| 値 | 意味 |
|----|------|
| `l2` | **既定**。各行をユークリッドノルムで除算（ゼロ除算回避のため下限あり）。全スペクトルでベクトル長を揃え、相対形状を比較しやすくする。 |
| `max` | 各行を **絶対値最大**で除算。ピーク強度を 1 に揃える。 |
| `none` | 行方向の規格化は行わない（その後の列方向 `StandardScaler` のみ）。 |

- **`[preprocess]` セクションを省略**した `config.toml` でも、強度規格化は **既定で `l2`** と同じ動きになります。  
- 設定例はリポジトリの [`config.toml`](config.toml) を参照。

```toml
[preprocess]
intensity_normalization = "l2"   # または "max" / "none"
```

## rasX ファイル

- rasX は **zip**。`Data0/Profile0.txt` を読みます。  
- TSV はヘッダなし、列は `twotheta`・`intensity_raw`・`attenuator`。強度は `intensity_raw × attenuator`。  
- **ファイル名**は次の形式である必要があります（座標の小数点は `-`）。

  `サンプル名_インデックス(2桁)_X座標_Y座標.rasx`

  例: `sample1_01_-42-000_5-000.rasx` → X = -42.0、Y = 5.0（単位はファイル命名に依存）。

## 設定ファイル（`config.toml`）

| セクション | キー | 説明 |
|------------|------|------|
| `[paths]` | `output_html`（任意） | HTML の出力パス |
| `[grid]` | `theta_min`, `theta_max`, `n_points` | 補間用 2θ の等間隔グリッド（度）。全スペクトルをここに揃える |
| `[grid]` | `exclude_ranges`（任意） | 特徴量から除外する 2θ レンジ。`[[26.4, 27.2], [54.1, 55.0]]` のように指定 |
| `[preprocess]` | `intensity_normalization` | `l2`（既定）・`max`・`none`。**セクション省略時も `l2` と同等** |
| `[pca]` | `n_components` | PCA の主成分数（**2 以上**）。t-SNE への入力次元の上限にもなる。実際は `min(設定値, サンプル数, 特徴次元)` に制限 |
| `[pca]` | `random_state` | PCA の乱数種（再現性） |
| `[tsne]` | `perplexity`, `max_iter`, `random_state` | t-SNE のパラメータ |
| `[dbscan]` | `eps`, `min_samples` | DBSCAN のパラメータ |
| `[dbscan]` | `clustering_space` | `feature`（既定）・`tsne`・`pca2d`。DBSCAN をどの空間にかけるか |
| `[visualize]` | `xrd_min_panel_height_px` | 各 XRD 図の最小高さの基準値（px）。実際の描画高さはこれを下限としてさらに広めに取る |

**制約・注意:**

- `tsne.max_iter` は **250 以上**（scikit-learn の仕様）。  
- グリッドが各ファイルの 2θ 範囲と十分重ならない場合、ログに **カバー率の警告**が出ます。  
- `grid.exclude_ranges` を使うと、指定した 2θ 範囲の列は **クラスタリング用特徴量から**落とされます。下段の XRD 図ではその範囲も表示され、灰色帯で示されます。  
- サンプル数が少ないと PCA の実効次元が `n_components` 未満になります。  
- 左の PCA 図は **PC1 / PC2 固定**です。`pca.n_components` を 2 より大きくしても、左図自体は基本的に変わりません。主に **t-SNE の入力次元**として効きます。  
- t-SNE はサンプル数が大きいと時間がかかります。  
- `dbscan.clustering_space = "feature"` の場合、高次元かつ列標準化後の距離スケールはデータ依存です。**DBSCAN の `eps` は実データで見直す**こと（全点がノイズ `-1` になりやすい場合は `eps` を大きめにする、など）。
- `dbscan.clustering_space = "tsne"` の場合、可視化に使う **t-SNE 2D 座標**に DBSCAN を適用します。図上の塊に近いクラスタを得やすい一方、t-SNE の `perplexity` や乱数種にも影響されます。
- `dbscan.clustering_space = "pca2d"` の場合、**PC1 / PC2** に対して DBSCAN を適用します。t-SNE が不安定なデータで、まず PCA 平面上の分離を見る用途に向きます。このモードでは t-SNE は計算せず、右パネルも PCA ベース表示になります。
- `visualize.xrd_min_panel_height_px` を上げると、各 XRD 図の縦方向が広がって見やすくなります。現在は描画時に **約 1.5 倍**へ拡張して使っています。

リポジトリ付属の [`config.toml`](config.toml) をテンプレートにできます。

## 開発

```sh
uv run ruff format .
uv run ruff check .
uv run ty check src tests
uv run pytest --cov=rasx_cluster_analyzer --cov-report=term-missing
```

コーディング方針・テスト方針の詳細は [`AGENTS.md`](AGENTS.md) を参照してください。
