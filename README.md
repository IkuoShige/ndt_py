# 2D NDT Localization

2次元の点群データに対するNormal Distribution Transform（NDT）を実装した自己位置推定システムです。このプロジェクトでは、占有格子地図上でのLiDARシミュレーションとNDTマッチングによる自己位置推定を行います。

## 機能

- 2D占有格子地図（.pgm, .yaml形式）からの地図読み込み
- LiDARセンサのシミュレーション
- NDT（Normal Distribution Transform）による点群マッチング
- Rerunを使用したリアルタイム可視化

## 環境構築

このプロジェクトでは、Python仮想環境マネージャ「uv」を使用します。以下のコマンドで環境をセットアップしてください：

```bash
# セットアップスクリプトを実行
./setup.sh

# または手動でセットアップ
# uvのインストール
curl -LsSf https://astral.sh/uv/install.sh | sh

# 仮想環境の作成と有効化
uv venv
source .venv/bin/activate

# 依存関係のインストール
uv pip install -r requirements.txt
```

## 使い方

```bash
# 仮想環境のアクティベーション
source .venv/bin/activate

# プログラムの実行
python ndt_localization.py --map_dir map --rerun --steps 100
```

### コマンドラインオプション

- `--map_dir`: マップファイル(.pgmと.yaml)が格納されているディレクトリ
- `--visualize`: matplotlibで結果を可視化
- `--steps`: シミュレーションのステップ数
- `--rerun`: Rerunを使用して可視化（デフォルトはオン）
- `--newton`: ニュートン法を使用してNDTマッチングを実行（デフォルトは勾配降下法）
- `--damping`: ニュートン法のダンピング係数（デフォルト: 0.1）
- `--trajectory`: 生成する軌道の種類（'random', 'circular', 'spiral'）
- `--radius`: 円軌道またはらせん軌道の半径（デフォルト: 5.0m）
- `--end_radius`: らせん軌道の終了半径（デフォルト: 1.0m）

### 軌道の種類

このシミュレータでは、以下の3種類の軌道を選択できます：

#### ランダム軌道
- 各ステップでランダムな方向変化を加えながら前進する自然な動き
- `--trajectory random`（デフォルト）

#### 円軌道
- 指定した中心点と半径で円を描く軌道
- 一定の曲率での性能評価に適している
- `--trajectory circular --radius 5.0`

#### らせん軌道
- 中心から外側または内側に向かってらせん状に動く軌道
- 半径が徐々に変化する場合のロバスト性評価に適している
- `--trajectory spiral --radius 5.0 --end_radius 1.0`

例：
```bash
# 円軌道を使った実行例
python ndt_localization.py --map_dir map --trajectory circular --radius 3.0

# 中心に向かうらせん軌道を使った実行例
python ndt_localization.py --map_dir map --trajectory spiral --radius 5.0 --end_radius 1.0
```

## アルゴリズムの詳細

### NDT (Normal Distribution Transform)

NDTは点群マッチングのアルゴリズムの一つで、以下のステップで構成されます：

1. リファレンスの点群を格子状のセルに分割
2. 各セルごとに点群の平均と共分散行列を計算して正規分布で表現
3. スキャンデータの点が、この確率分布場の中でどれだけ確からしいかをスコア化
4. スコアを最大化するように変換パラメータを最適化

### 最適化アルゴリズム

NDTマッチングでは、以下の2つの最適化アルゴリズムを選択できます：

#### 勾配降下法（デフォルト）
- スコア関数の勾配方向に一定のステップサイズで進む単純な方法
- 安定性が高いが、収束が遅い場合がある

#### ニュートン法
- 勾配（一階微分）とヘッセ行列（二階微分）の両方を利用
- より高速に収束する傾向がある
- 数値的に不安定になる場合があるため、ダンピング係数によって安定化

### 位置推定プロセス

1. 真の位置からLiDARスキャンをシミュレート
2. 前フレームの推定位置と移動量から初期推定値を設定
3. NDTマッチングで変換パラメータを最適化
4. 推定位置を更新
5. Rerunで結果を可視化

## ファイル構成

### コード
- `ndt_localization.py`: メイン実行ファイル
- `src/occupancy_grid_map.py`: 占有格子地図を扱うクラス
- `src/lidar_simulator.py`: LiDARシミュレータ
- `src/ndt.py`: NDTアルゴリズムの実装

### ドキュメント
- `docs/ndt_theory_and_implementation.md`: NDTアルゴリズムの理論と実装の詳細解説
- `docs/implementation_notes.md`: 実装時の知見やベストプラクティス
- `docs/performance_evaluation.md`: 性能評価と分析結果
- `docs/ndt_mathematical_derivation.md`: NDTアルゴリズムの数学的導出
- `docs/ndt_implementation_code.md`: 実装コードの具体例と解説
- `docs/ndt_experimental_results.md`: 実験結果とパラメータ分析

## 依存ライブラリ

- numpy: 科学計算
- matplotlib: データ可視化（オプション）
- pyyaml: YAMLファイルの読み込み
- rerun-sdk: リアルタイム可視化（バージョン0.23以降に対応）
- Pillow: 画像処理（PGMファイルの読み込み）
- scipy: 科学計算（行列演算など）

## 注意事項

- このプロジェクトはRerun SDKバージョン0.23以降に対応しています
- 古いバージョンのRerunでは、時間関連のAPI（`set_time_seconds`など）が異なる場合があります
