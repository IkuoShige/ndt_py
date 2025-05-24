# 2D NDT Localization Simulator

2D環境における**Normal Distribution Transform (NDT)**アルゴリズムを使用したローカライゼーションシミュレーターです。

## 新機能：実際のロボットに近い現実的なシミュレーション

### realistic_ndt_localization.py - 現実的なNDTローカライゼーション

実際のロボットに適応させることを想定した、より現実的なシミュレーション環境を提供します：

#### 主な改良点

1. **現実的なLiDARシミュレーション (`src/realistic_lidar_simulator.py`)**
   - 距離依存ノイズモデル
   - センサーの最小・最大検出距離制限
   - ビーム発散による角度ばらつき
   - 反射失敗とマルチパス反射
   - 反射強度のシミュレーション
   - センサー統計情報の提供

2. **現実的なオドメトリーモデル (`src/robot_odometry.py`)**
   - 車輪滑りとエンコーダー量子化誤差
   - 系統的誤差と累積誤差
   - キャリブレーション誤差（車輪間距離、半径等）
   - ジャイロドリフト
   - 不確実性の共分散行列計算

3. **ROSメッセージ互換性 (`src/ros_compatibility.py`)**
   - sensor_msgs/LaserScan
   - nav_msgs/Odometry
   - geometry_msgs/Pose, Transform
   - nav_msgs/OccupancyGrid
   - 将来的なROS統合を容易にするデータ構造

4. **改良されたローカライゼーションフレームワーク**
   - リアルタイム処理を想定した設計
   - センサーフュージョン（LiDAR + オドメトリー）
   - 不確実性の追跡と可視化
   - 詳細な統計情報とエラー分析

#### 使用方法

```bash
# 基本実行（現実的なノイズモデル有効）
python realistic_ndt_localization.py --map_dir map --steps 100

# ニュートン法を使用
python realistic_ndt_localization.py --map_dir map --newton --steps 100

# 速度制限を調整
python realistic_ndt_localization.py --map_dir map --max_velocity 0.3 --max_angular_velocity 0.5

# 低ノイズモード（従来モード）
python realistic_ndt_localization.py --map_dir map --no-realistic_noise
```

#### 主要パラメータ

- `--realistic_noise`: 現実的なノイズモデルを使用（デフォルト：有効）
- `--max_velocity`: 最大線速度 [m/s]（デフォルト：0.5）
- `--max_angular_velocity`: 最大角速度 [rad/s]（デフォルト：0.8）
- `--newton`: ニュートン法を使用（デフォルト：勾配降下法）

#### 出力情報

シミュレーション終了時に以下の詳細統計が表示されます：

```
=== 最終結果 ===
最終位置誤差: 0.1234 m
最終角度誤差: 2.34 度
平均位置誤差: 0.0567 m
平均角度誤差: 1.23 度
NDT成功率: 95.00%
平均計算時間: 12.34 ms

=== センサー統計 ===
LiDAR検出率: 97.5%
反射失敗率: 2.0%
マルチパス率: 0.5%

=== オドメトリー統計 ===
累積移動距離: 25.67 m
推定位置誤差: 0.0789 m
推定角度誤差: 1.45 度
```

## 従来のシミュレーション

### ndt_localization.py - 基本的なNDTローカライゼーション

#### 使用方法

```bash
# 基本的な実行
python ndt_localization.py --map_dir map

# ニュートン法を使用
python ndt_localization.py --map_dir map --newton

# 可視化を有効にして実行
python ndt_localization.py --map_dir map --visualize

# 軌道の種類を指定
python ndt_localization.py --map_dir map --trajectory circular --radius 5.0
python ndt_localization.py --map_dir map --trajectory spiral --radius 5.0 --end_radius 1.0
```

#### コマンドラインオプション

- `--map_dir`: マップファイル(.pgmと.yaml)が格納されているディレクトリ
- `--visualize`: matplotlibで結果を可視化
- `--steps`: シミュレーションのステップ数（デフォルト: 100）
- `--rerun`: Rerunを使用して可視化（デフォルト: オン）
- `--newton`: ニュートン法を使用してNDTマッチングを実行
- `--damping`: ニュートン法のダンピング係数（デフォルト: 0.1）
- `--trajectory`: 軌道の種類（random, circular, spiral）
- `--radius`: 円軌道またはらせん軌道の半径（デフォルト: 5.0）
- `--end_radius`: らせん軌道の終了半径（デフォルト: 1.0）

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

#### 従来のシミュレーション
1. 真の位置からLiDARスキャンをシミュレート
2. 前フレームの推定位置と移動量から初期推定値を設定
3. NDTマッチングで変換パラメータを最適化
4. 推定位置を更新
5. Rerunで結果を可視化

#### 現実的なシミュレーション
1. 制御コマンドから現実的なオドメトリー誤差を計算
2. 真の位置から現実的なLiDARスキャンをシミュレート
3. オドメトリーとLiDARデータを統合してNDTマッチングの初期値を設定
4. NDTマッチングで姿勢補正
5. 不確実性を考慮した姿勢更新
6. 詳細な統計情報の収集と可視化

## ファイル構成

### 新しい現実的なシミュレーション
- `realistic_ndt_localization.py`: 現実的なメイン実行ファイル
- `src/realistic_lidar_simulator.py`: 現実的なLiDARシミュレーター
- `src/robot_odometry.py`: 現実的なオドメトリーモデル
- `src/ros_compatibility.py`: ROSメッセージ互換データ構造

### 従来のシミュレーション
- `ndt_localization.py`: 基本的なメイン実行ファイル
- `src/occupancy_grid_map.py`: 占有格子地図を扱うクラス
- `src/lidar_simulator.py`: 基本的なLiDARシミュレータ
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

## 実際のロボットへの適応について

現実的なシミュレーション（`realistic_ndt_localization.py`）は、以下の点で実際のロボットシステムとの統合を容易にします：

1. **ROSメッセージ互換性**: 標準的なROSメッセージ形式に対応
2. **現実的なセンサーモデル**: 実際のセンサー特性を考慮
3. **リアルタイム処理設計**: 実用的な計算時間と処理頻度
4. **エラーハンドリング**: センサー故障やマッチング失敗に対する対処
5. **統計情報**: システム性能の監視と調整に必要な情報

将来的には、以下のような拡張が可能です：
- ROSノードとしての実装
- リアルタイムLiDARデータとの統合
- 実際のオドメトリーデータとの統合
- 動的環境や移動物体への対応

## 注意事項

- このプロジェクトはRerun SDKバージョン0.23以降に対応しています
- 古いバージョンのRerunでは、時間関連のAPI（`set_time_seconds`など）が異なる場合があります
- 現実的なシミュレーションでは、計算負荷が高くなる場合があります
