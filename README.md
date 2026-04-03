# rasx-cluster-analyzer

Rigaku のXYマップ測定用 **rasX** をまとめて読み込み、XRD 回折パターンを **DBSCAN** でクラスタリングし、**PCA** と設定に応じて **t-SNE**・**UMAP** または **PCA 2D** で右パネルを 2 次元表示します。結果は **Plotly** の HTML で出力します。

## HTML の構成

- メイン列（左）
  上段は、左が **常に PCA（PC1 vs PC2）**、右が第 2 埋め込みの 2 つの Plotly 散布図、その右にウェハマップ SVG を並べた構成です。散布図は `scaleanchor` と `constrain: domain` で 1:1 比率に固定しています。右パネルの算法は **`[embedding] method`** で選びます（`tsne` → t-SNE、`umap` → UMAP、いずれも PCA 射影を入力とする。`pca2d` → 右も PC1 vs PC2 の 2D 表示）。
- メイン列（左）
  下段は XRD プロファイル群です。先頭に `All profiles`、その下にクラスタごとの個別 Plotly 図を置いています。デスクトップでは 2 列、狭い画面では 1 列です。`grid.exclude_ranges` を設定していても、表示は除外前の全 2θ 範囲を使い、除外範囲はグレーの縦帯で示します。
- メイン列（左）
  XRD セクションの上には、全 XRD 図へまとめて適用する X/Y 軸フォームがあります。各 XRD 図の legend は独立していて、Plotly 標準操作で個別に表示を切り替えられます。
- 右サイドバー
  幅はおおむね `380px` 基準で、デスクトップでは `max 36vw`、狭い画面では `max 94vw` です。`Analysis metadata` と `Files by cluster` の 2 カードを表示し、内容が長い場合はサイドバー内だけスクロールします。デスクトップでは折りたたみ、狭い画面では開閉ができます。

表示用の XRD 強度は **除外前の全 2θ グリッド**に対する **行方向の強度正規化後・列方向 `StandardScaler` の前**の値です。一方、クラスタリング側では必要に応じて `grid.exclude_ranges` が適用されます。

**前処理:** 補間後の強度行列に対し、必要なら **2θ マスク**を適用し、その後 **スペクトル（行）ごと**に規格化（既定 **L2**）、続けて **各 2θ ビン（列）ごと**に `StandardScaler` で標準化してからクラスタリング・次元削減に入ります。

## 必要な環境

- Python 3.13 以上
- [uv](https://github.com/astral-sh/uv)（推奨）

## 依存関係（実行時）

`numpy`・`polars`・`scikit-learn`（PCA / DBSCAN / t-SNE）・`umap-learn`・`plotly`。

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
- HTML は **埋め込み用 Plotly 図**、**ウェハーマップ SVG パネル**、**複数の XRD Plotly 図**、および **ビューポート高さいっぱいの sticky 右サイドバー（解析メタデータ）**を 1 ページにまとめたものです。Plotly は **CDN** から読み込みます（オフライン閲覧が必要な場合は別途配布方法を検討してください）。

## 処理の流れ（概要）

1. 各 `.rasx` からスペクトルを読み、`config` の **共通 2θ グリッド**上に線形補間して特徴行列（行＝スペクトル、列＝2θ ビン）を作る。  
2. 表示用のために、**除外前の全 2θ グリッド**で行方向の強度規格化を行う。  
3. `grid.exclude_ranges` があれば、クラスタリング用の特徴量に対してだけその 2θ 範囲の列を落とす。  
4. クラスタリング用の特徴量に対して **行ごとの強度規格化**（`[preprocess]`、後述の「強度の規格化」参照）。既定は **L2**（各行ベクトルのノルムを 1 に）。  
5. **StandardScaler** で **列ごと**（各 2θ 位置で全サンプル横断）に平均 0・分散 1。  
6. **同じ行列**に **PCA** を 1 回フィットし、`[pca] n_components` 次元まで射影（次元はデータ形状で上限）。  
7. **埋め込み図**: 左は **常に PC1 vs PC2**。右は **`embedding.method`** に応じて、PCA 射影を入力とした **t-SNE（2D）**（`init='random'`）、**UMAP（2D）**、または **右も PC1 vs PC2（`pca2d`）** のいずれか。`pca.n_components >= 2` なら左図は **PC1 / PC2 固定**で、`pca.n_components` は主に **t-SNE / UMAP への入力次元**として効く。  
8. **DBSCAN** を **`dbscan.clustering_space`** に応じて、列標準化後の特徴（`scaled`）・**PCA 全成分**（`pca`）・**右パネルと同じ 2D 埋め込み**（`embedding`）・**PC1/PC2**（`pca2d`）のいずれかに適用しクラスタラベルを得る。  
9. **XRD 図**: まず **All profiles**、その下に各クラスタごとの **2θ–強度** の折線図を **別々の Plotly 図**として描く。ここで描いている強度は **除外前・行正規化後・列標準化前**。`exclude_ranges` は **グレー帯**で示す。ゼロ近傍・対数軸向けに極小正へクリップして **線が途切れない**ようにしている。  
10. XRD 用フォームから、全 XRD 図に対して **X/Y 範囲**と **Y 軸の linear/log** を一括で変更できる。  
11. ウェハーマップ用 SVG とメタデータ用 HTML を生成し、メイン列と右の sticky サイドバーに埋め込む。

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

## rasX 入力データ

### ディレクトリ

- 引数で渡すディレクトリの **直下** にある `.rasx` だけを対象にします（**サブフォルダは再帰しません**）。  
- 拡張子は **`.rasx`**（大文字小文字は区別せず拾います）。

### ファイル中身（Rigaku 形式）

- rasX は **ZIP アーカイブ**として開き、**`Data0/Profile0.txt`** を読みます。  
- TSV は **ヘッダなし**、列は `twotheta`・`intensity_raw`・`attenuator`。表示・クラスタリングに使う強度は **`intensity_raw × attenuator`** です。

### ファイル名（マップ測定用の命名規則）

ウェハマップやメタデータで **X/Y 座標**を使うため、**ファイル名（拡張子を除くベース名）**は次の形である必要があります。

```
<sampleName>_<index>_<X>_<Y>
```

| 要素 | ルール |
|------|--------|
| `<sampleName>` | 先頭から、**最後の 3 つの `_` で区切られたフィールドの手前まで**がサンプル名。`_` を含んでもよい（例: `my_batch_a_02_1-500_0-0.rasx` → サンプル名は `my_batch_a`）。 |
| `<index>` | **ちょうど 2 桁の数字**（`00`〜`99`。printfの `%02d` 相当）。 |
| `<X>`, `<Y>` | マップ上の座標を表す **1 トークンずつ**。Rigaku 流儀で **小数点の代わりに `-`** を使います。先頭の `-` だけが **負号**、それ以外の `-` は **小数点**に読み替えます。 |

拡張子は **`.rasx`** です（実装はベース名だけをパターン照合します）。

**正しい例**

| ファイル名 | 解釈 |
|------------|------|
| `sample1_01_-42-000_5-000.rasx` | サンプル `sample1`、インデックス `1`、X = **-42.0**、Y = **5.0** |
| `my_batch_a_02_1-500_0-0.rasx` | サンプル `my_batch_a`、インデックス `2`、X = **1.5**、Y = **0.0** |
| `x_03_0-0_0-0.rasx` | サンプル `x`、インデックス `3`、X/Y = **0.0** |

**使えない例**

- 区切りが足りない: `measurement.rasx`  
- インデックスが 2 桁でない・座標トークンが規則外: `no_index_01_x_y.rasx`  

いずれか 1 ファイルでもパターンに合わないと、読み込み時に **エラー**になります（座標が取れないためウェハマップも正しく描けません）。

座標の**物理単位**（mm など）はファイル名には含まれず、ツールは数値としてそのままプロットに使います。

## 設定ファイル（`config.toml`）

| セクション | キー | 説明 |
|------------|------|------|
| `[paths]` | `output_html`（任意） | HTML の出力パス |
| `[grid]` | `theta_min`, `theta_max`, `n_points` | 補間用 2θ の等間隔グリッド（度）。全スペクトルをここに揃える |
| `[grid]` | `exclude_ranges`（任意） | 特徴量から除外する 2θ レンジ。`[[26.4, 27.2], [54.1, 55.0]]` のように指定 |
| `[preprocess]` | `intensity_normalization` | `l2`（既定）・`max`・`none`。**セクション省略時も `l2` と同等** |
| `[pca]` | `n_components` | PCA の主成分数（**2 以上**）。t-SNE / UMAP への入力次元の上限にもなる。実際は `min(設定値, サンプル数, 特徴次元)` に制限 |
| `[pca]` | `random_state` | PCA の乱数種（再現性） |
| `[embedding]` | `method` | 右パネルの 2D 埋め込み: `tsne`・`umap`・`pca2d` |
| `[embedding]` | `n_components` | 現状 **2 のみ**（HTML は 2D 固定） |
| `[embedding.tsne]` | `perplexity`, `learning_rate`, `random_state`, `max_iter`（任意） | t-SNE。`learning_rate` は `"auto"` または正の数。`max_iter` 省略時は既定 1000 |
| `[embedding.umap]` | `n_neighbors`, `min_dist`, `metric`, `random_state` | UMAP（`method = "umap"` のときに使用） |
| `[dbscan]` | `eps`, `min_samples` | DBSCAN のパラメータ |
| `[dbscan]` | `clustering_space` | `scaled`（既定）・`pca`・`embedding`・`pca2d`。DBSCAN をどの空間にかけるか |
| `[visualize]` | `xrd_min_panel_height_px` | 各 XRD 図の最小高さの基準値（px）。実際の描画高さはこれを下限としてさらに広めに取る |

**制約・注意:**

- `embedding.tsne.max_iter` は **250 以上**（scikit-learn の仕様）。省略時の既定は 1000。  
- `embedding.n_components` は **2 のみ**許可（レイアウトが 2D 固定のため）。  
- グリッドが各ファイルの 2θ 範囲と十分重ならない場合、ログに **カバー率の警告**が出ます。  
- `grid.exclude_ranges` を使うと、指定した 2θ 範囲の列は **クラスタリング用特徴量から**落とされます。下段の XRD 図ではその範囲も表示され、灰色帯で示されます。  
- サンプル数が少ないと PCA の実効次元が `n_components` 未満になります。  
- 左の PCA 図は **PC1 / PC2 固定**です。`pca.n_components` を 2 より大きくしても、左図自体は基本的に変わりません。主に **t-SNE / UMAP の入力次元**として効きます。  
- t-SNE・UMAP はサンプル数が大きいと時間がかかります。`method = "pca2d"` のときは右パネル用に追加の多様体計算は行いません。  
- ウェハーマップは **ファイル名から読んだ X/Y 座標**に依存します。命名規則から外れたファイルは前段でエラーになります。  
- `dbscan.clustering_space = "scaled"` の場合、**列標準化後の高次元特徴**でクラスタリングします。距離スケールはデータ依存なので、**DBSCAN の `eps` は実データで見直す**こと（全点がノイズ `-1` になりやすい場合は `eps` を大きめにする、など）。
- `dbscan.clustering_space = "pca"` の場合、**PCA 射影の全成分**（`pca.n_components` 次元、データ形状で上限）に DBSCAN をかけます。
- `dbscan.clustering_space = "embedding"` の場合、**右パネルと同じ 2D 座標**（`embedding.method` が `tsne` / `umap` / `pca2d` のいずれかに対応）に DBSCAN をかけます。図上の塊に近い結果を得やすい一方、多様体側のパラメータや乱数種にも影響されます。
- `dbscan.clustering_space = "pca2d"` の場合、**PC1 / PC2** のみに DBSCAN をかけます。PCA 平面上の分離を素直に見たいとき向きです（右パネルの見た目は `embedding.method` に依存します）。
- `visualize.xrd_min_panel_height_px` を上げると、各 XRD 図の縦方向が広がって見やすくなります。現在は描画時に **約 1.5 倍**へ拡張して使っています。

**設定例（抜粋）:**

```toml
[embedding]
method = "tsne"   # "tsne" | "umap" | "pca2d"
n_components = 2

[embedding.tsne]
perplexity = 30.0
learning_rate = "auto"
random_state = 42

[embedding.umap]
n_neighbors = 15
min_dist = 0.1
metric = "euclidean"
random_state = 42

[dbscan]
clustering_space = "scaled"   # "scaled" | "pca" | "embedding" | "pca2d"
```

以前の設定ではトップレベル **`[tsne]`** や **`dbscan.clustering_space = "feature"` / `"tsne"`** を使っていました。現在は **`[embedding]`** と **`[embedding.tsne]` / `[embedding.umap]`**、および **`scaled` / `embedding`** に置き換えてください。

リポジトリ付属の [`config.toml`](config.toml) をテンプレートにできます。

## 開発

```sh
uv run ruff format .
uv run ruff check .
uv run ty check src tests
uv run pytest --cov=rasx_cluster_analyzer --cov-report=term-missing
```

コーディング方針・テスト方針の詳細は [`AGENTS.md`](AGENTS.md) を参照してください。
