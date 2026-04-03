# rasx-cluster-analyzer
## 概要
- XRD回折パターンをクラスタリングし次元削減した2次元マップで可視化するプログラム
- rasxファイルを含むディレクトリを引数とし、config.tomlの設定ファイル読み込んで可視化する。

## 開発環境&依存関係
- uvのpythonを使用
	- pythonのバージョンは3.13
- ty
- ruff
- plotly
- dbscan
- sklearn
	- PCA
	- DBSCAN
	- t-SNE
- numpy
- polars
	- pandasは使わない

## 仕様

## 使い方
```sh
uv run src/main.py -- --config <config file path> <rasX dir path> 
```


### RigakuのフォーマットのrasXファイル
#### rasX の読み込み方
- 仮想ファイルとしてzip解凍し、内部のData0/Profile0.txtを読み込み、twotheta, intensityを取得する。
- Profile0.txtは、twotheta,intensity_raw,attenuatorのヘッダーレスのtsv形式
- intensityはintensity_raw * attenuator算出する。
#### rasXファイル名の命名規則
```

<sampleName>_<index:Int(%02d)>_<Xcoordination:Float(小数点-)>_<Ycoordination:Float(小数点-)>.rasx

例) sample1_01_-42-000_5-000.rasx

```

### config.toml
クラスタリングのパラメータをまとめたファイル。意味論ごとに変数をまとめる。
デフォルトではプロジェクト直下のファイルを読み込む。オプション引数で別ファイルも指定できる
```
--config (or -c ) <config file path>
```

### クラスタリング
DBSCANを用いる。
次元削減にはPCAとt-SNEを用いて2次元にする

### 可視化
plotlyを用いてクラスタリングマップを埋め込んだhtmlを作る
plotlyのmapは高IPの学術論文誌にも適用できるスタイルにする。

## 開発手法
- テスト駆動開発(TDD)で実装
	- pytestを使用
	- 古典学派スタイルのTDD
		- モックは極力使わない
	- イテレーション後に必ず実施
	- 努力目標値はカバレッジ80%以上
- 型アノーテーションを必ず入れる
- tyで型チェックを実施
	- イテレーション後に必ず実施
- Ruffでフォーマットを実施
	- イテレーション後に必ず実施
- コードは意味論に基づきコンポーネント単位に細かく分割
