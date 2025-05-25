#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
import time
import argparse
from pathlib import Path
import sys

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.occupancy_grid_map import OccupancyGridMap
from src.lidar_simulator import LidarSimulator
from src.gicp import GICPMatcher

def generate_random_trajectory(start_pose, steps=100, step_size=0.2, angle_noise=0.05, pos_noise=0.02):
    """
    ランダムな軌道を生成する
    
    Args:
        start_pose (numpy.ndarray): 開始位置 [x, y, theta]
        steps (int): ステップ数
        step_size (float): ステップサイズ
        angle_noise (float): 角度のランダム性の大きさ
        pos_noise (float): 位置のランダム性の大きさ
    
    Returns:
        list: 姿勢のリスト [pose1, pose2, ...]
    """
    trajectory = [start_pose.copy()]
    current_pose = start_pose.copy()
    
    for _ in range(steps):
        # 角度にランダムな変化を加える
        current_pose[2] += np.random.uniform(-angle_noise, angle_noise)
        
        # 現在の向きに基づいて前進
        current_pose[0] += step_size * np.cos(current_pose[2]) + np.random.uniform(-pos_noise, pos_noise)
        current_pose[1] += step_size * np.sin(current_pose[2]) + np.random.uniform(-pos_noise, pos_noise)
        
        # 姿勢を保存
        trajectory.append(current_pose.copy())
    
    return trajectory

def generate_circular_trajectory(center, radius, steps=100, start_angle=0, angular_range=2*np.pi, noise=0.02):
    """
    円軌道を生成する
    
    Args:
        center (tuple): 円の中心座標 (x, y)
        radius (float): 円の半径 [m]
        steps (int): ステップ数
        start_angle (float): 開始角度 [rad]
        angular_range (float): 角度範囲 [rad]（2*piで完全な円）
        noise (float): 位置にランダムなノイズを加える量 [m]
    
    Returns:
        list: 姿勢のリスト [pose1, pose2, ...]
    """
    trajectory = []
    
    # 角度のリストを生成
    angles = np.linspace(start_angle, start_angle + angular_range, steps)
    
    for angle in angles:
        # 円上の位置を計算
        x = center[0] + radius * np.cos(angle) + np.random.uniform(-noise, noise)
        y = center[1] + radius * np.sin(angle) + np.random.uniform(-noise, noise)
        
        # 進行方向を計算（円の接線方向）
        theta = angle + np.pi/2  # 半時計回りの場合は +π/2
        
        # 姿勢を保存
        pose = np.array([x, y, theta])
        trajectory.append(pose)
    
    return trajectory

def generate_spiral_trajectory(center, start_radius, end_radius, steps=100, start_angle=0, angular_range=2*np.pi, noise=0.02):
    """
    らせん軌道を生成する
    
    Args:
        center (tuple): らせんの中心座標 (x, y)
        start_radius (float): 開始半径 [m]
        end_radius (float): 終了半径 [m]
        steps (int): ステップ数
        start_angle (float): 開始角度 [rad]
        angular_range (float): 角度範囲 [rad]
        noise (float): 位置にランダムなノイズを加える量 [m]
    
    Returns:
        list: 姿勢のリスト [pose1, pose2, ...]
    """
    trajectory = []
    
    # 角度のリストを生成
    angles = np.linspace(start_angle, start_angle + angular_range, steps)
    
    # 半径の変化を計算
    radii = np.linspace(start_radius, end_radius, steps)
    
    for i, angle in enumerate(angles):
        # 現在の半径
        radius = radii[i]
        
        # らせん上の位置を計算
        x = center[0] + radius * np.cos(angle) + np.random.uniform(-noise, noise)
        y = center[1] + radius * np.sin(angle) + np.random.uniform(-noise, noise)
        
        # 進行方向を計算
        # らせんの接線方向（半径方向の成分と円周方向の成分の合成）
        dr = (end_radius - start_radius) / steps  # 1ステップあたりの半径変化
        dtheta = angular_range / steps  # 1ステップあたりの角度変化
        
        # 接線方向の角度（アークタンジェント）
        tangent_angle = angle + np.pi/2
        if dr != 0:
            tangent_angle = np.arctan2(radius * dtheta, dr)
            if end_radius < start_radius:  # 縮小らせんの場合
                tangent_angle += np.pi
        
        # 姿勢を保存
        pose = np.array([x, y, tangent_angle])
        trajectory.append(pose)
    
    return trajectory

def create_target_map(occupancy_map, resolution=0.5):
    """
    占有格子地図から参照用の点群マップを生成
    
    Args:
        occupancy_map: 占有格子地図オブジェクト
        resolution (float): サンプリング解像度
    
    Returns:
        numpy.ndarray: 点群のリスト [[x1, y1], [x2, y2], ...]
    """
    binary_map = occupancy_map.get_occupancy_grid()
    height, width = binary_map.shape
    
    # 占有セルの座標を抽出
    occupied_indices = np.where(binary_map > 0)
    occupied_coords = list(zip(occupied_indices[1], occupied_indices[0]))
    
    # サブサンプリング（点の数を減らす）
    sampled_indices = np.random.choice(len(occupied_coords), 
                                       size=min(10000, len(occupied_coords)), 
                                       replace=False)
    sampled_coords = [occupied_coords[i] for i in sampled_indices]
    
    # ピクセル座標をワールド座標に変換
    world_points = []
    for px, py in sampled_coords:
        x, y = occupancy_map.pixel_to_world(px, py)
        world_points.append([x, y])
    
    return np.array(world_points)

def run_multiscale_gicp(gicp_matcher, scan_points, initial_guess, damping=0.2, scales=[1.0, 0.5, 0.25]):
    """マルチスケールGICPアプローチを実行
    
    段階的に精度を上げながらマッチングを行う
    """
    current_params = initial_guess.copy()
    best_score = float('inf')
    best_params = current_params.copy()
    total_iterations = 0
    
    # 元の設定を保存
    original_threshold = gicp_matcher.dist_threshold
    original_max_iter = gicp_matcher.max_iterations
    
    for scale in scales:
        # スケールに応じた距離閾値の調整
        current_threshold = original_threshold * scale * 2
        gicp_matcher.dist_threshold = max(0.2, current_threshold)
        
        # スケールに応じたダンピング係数の調整
        current_damping = damping * scale
        
        # 反復回数も調整
        current_max_iter = int(original_max_iter * (0.5 + 0.5 * scale))
        gicp_matcher.max_iterations = current_max_iter
        
        print(f"スケール {scale:.2f} でマッチング: 閾値={gicp_matcher.dist_threshold:.3f}m, ダンピング={current_damping:.3f}")
        
        # このスケールでのGICPマッチング
        params, score, iterations = gicp_matcher.align_newton(scan_points, current_params, damping=current_damping)
        total_iterations += iterations
        
        # 良い結果が得られたらそれを次のスケールの初期値に
        if score < best_score:
            best_score = score
            best_params = params.copy()
            
        # 次のスケールの初期値として現在の結果を使用
        current_params = best_params.copy()
    
    # 元の設定を復元
    gicp_matcher.dist_threshold = original_threshold
    gicp_matcher.max_iterations = original_max_iter
    
    return best_params, best_score, total_iterations

def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description='2D GICP Localization Simulator')
    parser.add_argument('--map_dir', type=str, default='map',
                        help='マップファイル(.pgmと.yaml)が格納されているディレクトリ')
    parser.add_argument('--visualize', action='store_true',
                        help='matplotlibで結果を可視化')
    parser.add_argument('--steps', type=int, default=100,
                        help='シミュレーションのステップ数')
    parser.add_argument('--rerun', action='store_true', default=True,
                        help='Rerunを使用して可視化（デフォルトはオン）')
    parser.add_argument('--newton', action='store_true',
                        help='ニュートン法を使用してGICPマッチングを実行')
    parser.add_argument('--damping', type=float, default=0.2,
                        help='ニュートン法のダンピング係数（デフォルト: 0.2）')
    parser.add_argument('--trajectory', type=str, default='random', choices=['random', 'circular', 'spiral'],
                        help='生成する軌道の種類（random: ランダム軌道、circular: 円軌道、spiral: らせん軌道）')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='円軌道またはらせん軌道の半径（デフォルト: 5.0）')
    parser.add_argument('--end_radius', type=float, default=1.0,
                        help='らせん軌道の終了半径（デフォルト: 1.0）')
    parser.add_argument('--neighbors', type=int, default=30,
                        help='GICPの近傍点数（デフォルト: 30）')
    parser.add_argument('--correspondence_threshold', type=float, default=0.5,
                        help='対応点とみなす最大距離（dist_threshold、デフォルト: 0.5m）')
    args = parser.parse_args()
    
    # 地図ファイルのパスを構築
    map_dir = Path(args.map_dir)
    pgm_files = list(map_dir.glob('*.pgm'))
    if not pgm_files:
        print(f"Error: No .pgm files found in {map_dir}")
        return
    
    pgm_path = pgm_files[0]
    yaml_path = pgm_path.with_suffix('.yaml')
    
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {yaml_path}")
        return
    
    # Rerunのセットアップ
    if args.rerun:
        rr.init("2D GICP Localization", spawn=True)
        # 新しいAPI（Rerun 0.23以降）を使用
        # rr.set_time_seconds("frame", 0.0)
        rr.set_time(timestamp=0.0, timeline="frame")
    
    # 占有格子地図を読み込み
    occupancy_map = OccupancyGridMap(str(pgm_path), str(yaml_path))
    
    # LiDARシミュレーターの初期化
    lidar_sim = LidarSimulator(occupancy_map, num_beams=360, max_range=10.0)
    
    # 参照用の点群マップを生成
    target_points = create_target_map(occupancy_map)
    
    # GICPマッチャーの初期化（パラメータを最適化）
    gicp_matcher = GICPMatcher(target_points, 
                             max_neighbors=40,  # 近傍点数を増加
                             max_iterations=50,  # 反復回数を増加
                             dist_threshold=0.3,  # 対応点距離閾値を厳しく
                             epsilon=1e-8,  # 収束判定をより厳しく
                             sigma=0.01)  # 共分散行列のばらつきを減少

    # 初期位置を空きスペースからランダムに選択
    start_position = occupancy_map.get_empty_positions(1)[0]
    start_pose = np.array([start_position[0], start_position[1], 0.0])  # [x, y, theta]
    
    # 軌道の生成
    if args.trajectory == 'random':
        # ランダムな軌道を生成
        print(f"ランダム軌道を生成: {args.steps}ステップ")
        true_trajectory = generate_random_trajectory(start_pose, steps=args.steps)
    elif args.trajectory == 'circular':
        # 円軌道を生成
        print(f"円軌道を生成: 半径={args.radius}m, {args.steps}ステップ")
        center = start_position  # 開始位置を中心とする
        true_trajectory = generate_circular_trajectory(center, args.radius, steps=args.steps)
    elif args.trajectory == 'spiral':
        # らせん軌道を生成
        print(f"らせん軌道を生成: 開始半径={args.radius}m, 終了半径={args.end_radius}m, {args.steps}ステップ")
        center = start_position  # 開始位置を中心とする
        true_trajectory = generate_spiral_trajectory(center, args.radius, args.end_radius, steps=args.steps)
    
    estimated_trajectory = [true_trajectory[0].copy()]  # 初期位置は真の位置と同じ
    
    # 現在の推定位置
    current_estimate = true_trajectory[0].copy()
    
    # スコア追跡用の変数を初期化
    prev_score = 0.0
    
    # シミュレーション開始
    print("シミュレーション開始...")
    print(f"アルゴリズム: GICP ({'ニュートン法' if args.newton else '標準法'})")
    print(f"近傍点数: {args.neighbors}, 対応点距離閾値: {args.correspondence_threshold}m")
    
    # トラッキング性能のメトリクス
    position_errors = []
    angle_errors = []
    confidences = []
    total_position_errors = 0.0
    total_angle_errors = 0.0
    max_position_error = 0.0
    max_angle_error = 0.0
    true_path_length = 0.0
    estimated_path_length = 0.0
    
    for i in range(1, len(true_trajectory)):
        true_pose = true_trajectory[i]
        print(f"ステップ {i}: 真の位置: {true_pose}")
        
        # 真の位置からのLiDARスキャンをシミュレート
        _, scan_points = lidar_sim.simulate_scan(true_pose)
        
        if len(scan_points) < 5:  # スキャンポイントが少なすぎる場合はスキップ
            print(f"警告: ステップ {i} でスキャンポイントが不足しています。前回の推定値を使用します。")
            estimated_trajectory.append(current_estimate.copy())
            continue
        
        # オドメトリ的なアプローチ: より精密な初期値推定
        if i == 1:
            # 最初のステップは正確な初期位置が重要なので、少量のノイズのみ
            delta_pose = true_trajectory[i] - true_trajectory[i-1]  # 真の移動量（オドメトリ）
            noisy_delta = delta_pose.copy()
            noisy_delta[0] += np.random.normal(0, 0.005)  # xノイズを半減
            noisy_delta[1] += np.random.normal(0, 0.005)  # yノイズを半減
            noisy_delta[2] += np.random.normal(0, 0.002)  # 角度ノイズを半減
        else:
            # 2回目以降は前回の推定値との差分を利用
            delta_pose_true = true_trajectory[i] - true_trajectory[i-1]  # 真の移動量
            delta_theta = delta_pose_true[2]
            
            # 角度の正規化
            while delta_theta > np.pi:
                delta_theta -= 2*np.pi
            while delta_theta < -np.pi:
                delta_theta += 2*np.pi
            
            # 移動距離を真の値に近づける（オドメトリ的な情報）
            move_dist = np.linalg.norm(delta_pose_true[:2])
            move_dir = delta_pose_true[:2] / (move_dist + 1e-10)
            
            # 前回の推定誤差を考慮した初期値
            noisy_delta = np.zeros(3)
            noisy_delta[0] = move_dist * move_dir[0]  # 真の移動距離・方向
            noisy_delta[1] = move_dist * move_dir[1]  # 真の移動距離・方向
            noisy_delta[2] = delta_theta
            
            # より現実的なノイズを加える
            noise_scale = min(0.02, move_dist * 0.1)  # 移動距離に応じたノイズスケール
            noisy_delta[0] += np.random.normal(0, noise_scale)
            noisy_delta[1] += np.random.normal(0, noise_scale)
            noisy_delta[2] += np.random.normal(0, 0.002)
        
        # オドメトリを初期値に使用
        initial_guess = np.array([noisy_delta[2], noisy_delta[0], noisy_delta[1]])  # [theta, x, y]
        
        # 測定値に基づいてGICPマッチングを実行（ニュートン法またはマルチスケール）
        start_time = time.time()
        if args.newton:
            # ニュートン法のダンピング係数を適応的に調整
            # 前回との位置変化が大きい場合や対応点が少ない場合には慎重に調整
            adaptive_damping = args.damping
            
            if i > 1:
                # 前ステップからの移動量を計算
                delta_est = np.linalg.norm(true_trajectory[i][:2] - true_trajectory[i-1][:2])
                
                if delta_est > 0.3:  # 移動量が大きい場合はダンピングを増加
                    adaptive_damping = min(args.damping * 1.5, 0.5)
                elif prev_score > 0.05:  # 前回のスコアが悪い場合はダンピングを増加
                    adaptive_damping = min(args.damping * 2.0, 0.5)
            
            # マルチスケールアプローチを使用
            params, score, iterations = run_multiscale_gicp(gicp_matcher, scan_points, initial_guess, damping=adaptive_damping)
            print(f"マルチスケールGICP: 反復回数={iterations}")
        else:
            params, score, iterations = gicp_matcher.align(scan_points, initial_guess)
        elapsed_time = time.time() - start_time
        
        # 前回のスコアを保存
        prev_score = score
        
        # 対応点の数を取得
        correspondences, _ = gicp_matcher._find_correspondences(scan_points, params)
        correspondences_ratio = len(correspondences) / max(1, len(scan_points))
        print(f"対応点数: {len(correspondences)}/{len(scan_points)} ({correspondences_ratio:.2f}), スコア: {score:.4f}")
        
        # 推定位置を更新（正しい座標変換を適用）
        # GICPの返り値paramsは [theta, tx, ty] 形式のローカル座標系での変換
        # それをグローバル座標系に変換する必要がある
        local_theta = params[0]  # スキャン座標系での回転角
        local_tx = params[1]     # スキャン座標系でのx方向移動
        local_ty = params[2]     # スキャン座標系でのy方向移動
        
        # 現在の推定位置の姿勢（回転）を取得
        current_theta = current_estimate[2]
        
        # ローカル変換をグローバル座標系に変換
        # 回転行列R(current_theta)を使ってローカル並進をグローバルに変換
        c, s = np.cos(current_theta), np.sin(current_theta)
        delta_x_global = c * local_tx - s * local_ty
        delta_y_global = s * local_tx + c * local_ty
        
        # 回転角度も累積
        theta = local_theta
        
        # スコアと対応点数に基づく信頼度係数を計算
        # 対応点比率はそのまま使用（対応点が多いほど信頼度が高い）
        score_factor = 0.0
        if score < float('inf'):
            # スコアが小さいほど高い信頼度 (0 < score < 1 の範囲にマッピング)
            max_acceptable_score = 0.1  # より厳しい閾値
            min_acceptable_score = 0.005  # より厳しい閾値
            
            if score < min_acceptable_score:
                score_factor = 1.0
            elif score > max_acceptable_score:
                score_factor = 0.0
            else:
                # 変更: 指数関数的減衰を使用（より適切な評価）
                score_factor = np.exp(-(score - min_acceptable_score) / 
                                    (max_acceptable_score - min_acceptable_score) * 5)
            
        # 対応点の品質評価を考慮した総合信頼度
        # 1. 対応点の分布均一性評価（中央値と四分位範囲を利用）
        distribution_factor = 1.0
        if len(correspondences) > 10:
            # 対応点の角度分布を計算
            angles = []
            center = np.mean([scan_points[src_idx] for src_idx, _, _ in correspondences], axis=0)
            for src_idx, _, _ in correspondences:
                point = scan_points[src_idx]
                angle = np.arctan2(point[1] - center[1], point[0] - center[0])
                angles.append(angle)
            
            # 角度の分布を評価（均一な分布が望ましい）
            hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
            filled_bins = np.sum(hist > 0)
            distribution_factor = filled_bins / 8.0  # 8方向すべてに対応点があると1.0
        
        # 2. 前回の推定との整合性（急激な変化がないか）
        consistency_factor = 1.0
        if i > 1:
            # 前回のステップからの移動距離
            prev_est = estimated_trajectory[-2][:2]
            expected_est = prev_est + np.array([delta_x_global, delta_y_global])
            expected_dist = np.linalg.norm(expected_est - prev_est)
            
            # オドメトリからの予測移動量
            odom_dx = true_trajectory[i][0] - true_trajectory[i-1][0]
            odom_dy = true_trajectory[i][1] - true_trajectory[i-1][1]
            odom_dist = np.sqrt(odom_dx**2 + odom_dy**2)
            
            # 移動量の比率（GICPとオドメトリ）
            if odom_dist > 0.02:  # 十分な移動がある場合のみ評価
                dist_ratio = expected_dist / max(0.01, odom_dist)
                # 比率が極端に大きいまたは小さい場合は不自然
                if dist_ratio > 2.0 or dist_ratio < 0.3:
                    consistency_factor = 0.5
                elif dist_ratio > 1.5 or dist_ratio < 0.5:
                    consistency_factor = 0.7
        
        # 最終的な信頼度計算（各要素を重み付けして統合）
        # 基本の重みは対応点比率を使用（主要な指標）
        confidence_base = correspondences_ratio
        
        # 対応点数が少なすぎる場合の信頼度調整
        min_correspondence_ratio = 0.4  # 最低限必要な対応点の割合
        if correspondences_ratio < min_correspondence_ratio:
            # 対応点が少なすぎる場合、信頼度を低減
            confidence_base *= correspondences_ratio / min_correspondence_ratio
        
        # 対応点比率とスコアは非常に重要（高めに重み付け）
        confidence = 0.5 * confidence_base + 0.3 * score_factor + 0.1 * distribution_factor + 0.1 * consistency_factor
        confidence = max(0.0, min(1.0, confidence))
        
        print(f"信頼度: {confidence:.3f} (対応点: {confidence_base:.3f}, スコア: {score_factor:.3f}, 分布: {distribution_factor:.3f}, 一貫性: {consistency_factor:.3f})")
        
        # 信頼度が低いか、誤差が閾値を超える場合はオドメトリを使用
        position_error = np.sqrt(delta_x_global**2 + delta_y_global**2)
        
        # オドメトリの移動量を取得
        odom_dx = true_trajectory[i][0] - true_trajectory[i-1][0]
        odom_dy = true_trajectory[i][1] - true_trajectory[i-1][1]
        odom_dtheta = true_trajectory[i][2] - true_trajectory[i-1][2]
        
        # 角度の連続性を保つためにアンラップ処理
        while odom_dtheta > np.pi:
            odom_dtheta -= 2*np.pi
        while odom_dtheta < -np.pi:
            odom_dtheta += 2*np.pi
        
        # オドメトリをグローバル座標系に変換
        odom_dx_global = c * odom_dx - s * odom_dy
        odom_dy_global = s * odom_dx + c * odom_dy
        
        # オドメトリの移動量を確認
        odom_dist = np.sqrt(odom_dx_global**2 + odom_dy_global**2)
        print(f"オドメトリ移動量: {odom_dist:.3f}m, GICP移動量: {position_error:.3f}m")
        
        # 移動量や信頼度に基づいて適応的にオドメトリと推定値をブレンド
        # 基本的な信頼度でブレンド係数を決定
        use_odom_factor = 0.0
        
        # 信頼度の階層的判定基準
        CONFIDENCE_THRESHOLD_HIGH = 0.8   # 高信頼度（オドメトリほぼ不要）
        CONFIDENCE_THRESHOLD_MID = 0.6    # 中信頼度（軽いブレンド）
        CONFIDENCE_THRESHOLD_LOW = 0.4    # 低信頼度（ブレンドに依存）
        CONFIDENCE_THRESHOLD_CRITICAL = 0.2  # 危険な信頼度（オドメトリ主体）
        
        # 動きの不自然さの判定
        movement_ratio = position_error / max(odom_dist, 0.01)
        is_movement_unnatural = movement_ratio > 3.0 or movement_ratio < 0.2
        
        # 基本的なブレンド係数の決定
        if is_movement_unnatural:
            print(f"警告: 移動量が不自然 (GICP:オドメトリ = {movement_ratio:.2f})")
            use_odom_factor = 0.8  # 動きが不自然な場合は強くオドメトリを信頼
        elif confidence < CONFIDENCE_THRESHOLD_CRITICAL:
            print(f"警告: 推定の信頼度が非常に低い (信頼度:{confidence:.3f})")
            use_odom_factor = 0.85  # ほぼオドメトリに依存
        elif confidence < CONFIDENCE_THRESHOLD_LOW:
            print(f"警告: 推定の信頼度が低い (信頼度:{confidence:.3f})")
            use_odom_factor = 0.7  # オドメトリにかなり依存
        elif confidence < CONFIDENCE_THRESHOLD_MID:
            print(f"注意: 推定の信頼度がやや低い (信頼度:{confidence:.3f})")
            use_odom_factor = 0.5  # オドメトリと均等にブレンド
        elif confidence < CONFIDENCE_THRESHOLD_HIGH:
            use_odom_factor = 0.3  # ほぼGICPに依存するが少しブレンド
        else:
            use_odom_factor = 0.1  # 高信頼度時は最小限のブレンド（安定性のため）
        
        # 特殊ケース：オドメトリが極端に小さい場合（静止状態など）
        if odom_dist < 0.01:
            if position_error > 0.1:
                # オドメトリは静止しているのにGICPが大きく動く場合は不審
                print(f"注意: 静止時の大きな動き (GICP移動量: {position_error:.3f}m)")
                use_odom_factor = min(0.9, use_odom_factor + 0.2)  # より強くオドメトリを使用
        
        # 信頼度とオドメトリの使用率に基づいて、推定値とオドメトリを組み合わせる
        if use_odom_factor > 0.0:
            print(f"オドメトリブレンド率: {use_odom_factor:.3f}")
            delta_x_global = (1 - use_odom_factor) * delta_x_global + use_odom_factor * odom_dx_global
            delta_y_global = (1 - use_odom_factor) * delta_y_global + use_odom_factor * odom_dy_global
            theta = (1 - use_odom_factor) * theta + use_odom_factor * odom_dtheta
        
        # 現在の推定位置を更新
        current_estimate[0] += delta_x_global
        current_estimate[1] += delta_y_global
        current_estimate[2] += theta

        # 正規化の前の値を保存（デバッグ用）
        theta_before_norm = current_estimate[2]
        
        # グローバルな軌道の方向を考慮した角度の更新
        if i > 1:
            # 前回の推定位置から現在の位置への方向ベクトル
            prev_est = estimated_trajectory[-2][:2]
            current_est = current_estimate[:2].copy()
            direction_vector = current_est - prev_est
            direction_norm = np.linalg.norm(direction_vector)
            
            # 前回の角度
            prev_theta = estimated_trajectory[-2][2]
            current_theta = current_estimate[2]
            
            # 角度差の計算（-πからπの範囲）
            angle_diff = ((current_theta - prev_theta + np.pi) % (2*np.pi)) - np.pi
            
            # 大きな角度変化がある場合は、前後の角度と移動方向から最も合理的な値を選択
            if abs(angle_diff) > np.pi/3 and direction_norm > 0.05:  # 60度以上の急激な変化かつ十分な移動量
                # 移動方向から推定される角度
                dir_theta = np.arctan2(direction_vector[1], direction_vector[0])
                
                # 移動方向を向いている場合の角度（前向き）
                forward_theta = dir_theta
                # 移動方向と反対を向いている場合の角度（後ろ向き）
                backward_theta = np.mod(dir_theta + np.pi, 2*np.pi)
                if backward_theta > np.pi:
                    backward_theta -= 2*np.pi
                
                # 前回の角度に近い方向を選択
                candidates = [forward_theta, backward_theta]
                diffs = [abs(((c - prev_theta + np.pi) % (2*np.pi) - np.pi)) for c in candidates]
                best_idx = np.argmin(diffs)
                
                # 移動方向に基づく最も合理的な角度を高い重みで採用
                blend_factor = 0.8
                current_estimate[2] = (1 - blend_factor) * current_theta + blend_factor * candidates[best_idx]
                print(f"角度を移動方向に基づいて修正: {np.degrees(current_theta):.1f}° -> {np.degrees(current_estimate[2]):.1f}°")
            
            # 連続性を保つためのアンラップ処理
            # 角度の急激な変化を防ぐため、2πの倍数を調整
            angle_diff = current_estimate[2] - prev_theta
            
            # -πからπの範囲に正規化
            while angle_diff > np.pi:
                angle_diff -= 2.0 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2.0 * np.pi
            
            # 急激な変化は不自然なので、抑制する
            if abs(angle_diff) > np.pi/2:  # 90度以上の変化
                print(f"警告: 角度の急激な変化を検出 ({np.degrees(angle_diff):.1f}度)")
                # 変化を抑制（最大でも45度程度の変化に制限）
                max_change = np.pi/4 * np.sign(angle_diff)
                current_estimate[2] = prev_theta + max_change
                print(f"角度変化を制限: {np.degrees(angle_diff):.1f}度 -> {np.degrees(max_change):.1f}度")
            
            # 方向ベクトルからの推定角度（移動方向を向いていると仮定）
            if direction_norm > 0.05:  # 十分な移動量がある場合のみ方向を考慮
                estimated_theta = np.arctan2(direction_vector[1], direction_vector[0])
                
                # 現在の推定角度と移動方向から計算した角度の差
                # -π～πの範囲に正規化
                theta_diff = ((current_estimate[2] - estimated_theta + np.pi) % (2*np.pi)) - np.pi
                
                # 角度と移動方向の整合性を評価（90度以上の差は不自然）
                if abs(theta_diff) > np.pi/2:
                    print(f"警告: 角度が移動方向と大きく不整合 ({np.degrees(theta_diff):.1f}度)")
                    # 移動量が大きいほど方向ベクトルの信頼性が高い
                    direction_confidence = min(1.0, direction_norm / 0.2)  # 0.2m以上の移動で最大値
                    
                    # 移動方向と角度を適応的にブレンド
                    blend_factor = 0.6 * direction_confidence
                    adjusted_theta = (1 - blend_factor) * current_estimate[2] + blend_factor * estimated_theta
                    current_estimate[2] = adjusted_theta
                    print(f"角度補正: {np.degrees(current_estimate[2] - adjusted_theta):.1f}度, ブレンド係数: {blend_factor:.2f}")
                # 45〜90度の差は部分的に補正
                elif abs(theta_diff) > np.pi/4:
                    blend_factor = 0.3 * min(1.0, direction_norm / 0.1)
                    adjusted_theta = (1 - blend_factor) * current_estimate[2] + blend_factor * estimated_theta
                    current_estimate[2] = adjusted_theta
        
        # 最終的な正規化（-πからπの範囲に）
        current_estimate[2] = np.arctan2(np.sin(current_estimate[2]), np.cos(current_estimate[2]))
        
        print(f"角度正規化: {np.degrees(theta_before_norm):.1f}度 -> {np.degrees(current_estimate[2]):.1f}度")
        
        # デバッグ情報と性能メトリクスの記録
        position_error = np.linalg.norm(true_pose[:2] - current_estimate[:2])
        angle_error = abs(true_pose[2] - current_estimate[2])
        angle_error = min(angle_error, 2*np.pi - angle_error)  # [0,π]の範囲に正規化
        
        print(f"推定位置: [{current_estimate[0]:.3f}, {current_estimate[1]:.3f}, {np.degrees(current_estimate[2]):.1f}°]")
        print(f"真の位置: [{true_pose[0]:.3f}, {true_pose[1]:.3f}, {np.degrees(true_pose[2]):.1f}°]")
        print(f"位置誤差: {position_error:.3f}m, 角度誤差: {np.degrees(angle_error):.1f}度")
        
        # 過去の平均誤差と最大誤差を更新
        total_position_errors += position_error
        total_angle_errors += angle_error
        max_position_error = max(max_position_error, position_error)
        max_angle_error = max(max_angle_error, angle_error)
        
        # 状態の記録
        position_errors.append(position_error)
        angle_errors.append(angle_error)
        confidences.append(confidence)
        
        # 移動軌跡の長さを更新
        # if i > 0:
        #     true_path_length += np.linalg.norm(true_trajectory[i][:2] - true_trajectory[i-1][:2])
        #     estimated_path_length += np.linalg.norm(estimated_trajectory[-1][:2] - estimated_trajectory[-2][:2])
        # estimated_trajectory.append(current_estimate.copy())
        # 移動軌跡の長さを更新
        if i > 0:
            true_path_length += np.linalg.norm(true_trajectory[i][:2] - true_trajectory[i-1][:2])
            # リストの長さをチェックしてからアクセス
            if len(estimated_trajectory) >= 2:
                estimated_path_length += np.linalg.norm(estimated_trajectory[-1][:2] - estimated_trajectory[-2][:2])
        estimated_trajectory.append(current_estimate.copy())
        
        # 進捗表示と性能メトリクス
        if i % 10 == 0 or i == len(true_trajectory) - 1:
            avg_position_error = total_position_errors / (i + 1)
            avg_angle_error = np.degrees(total_angle_errors / (i + 1))
            
            print(f"\n===== ステップ {i}/{len(true_trajectory)-1} =====")
            print(f"現在の誤差: {position_error:.3f}m / {np.degrees(angle_error):.1f}°")
            print(f"平均誤差: {avg_position_error:.3f}m / {avg_angle_error:.1f}°")
            print(f"最大誤差: {max_position_error:.3f}m / {np.degrees(max_angle_error):.1f}°")
            
            # 直近10ステップの誤差傾向
            if len(position_errors) >= 10:
                recent_pos_err = np.mean(position_errors[-10:])
                recent_ang_err = np.degrees(np.mean(angle_errors[-10:]))
                recent_conf = np.mean(confidences[-10:])
                print(f"直近10ステップ: 位置誤差={recent_pos_err:.3f}m, 角度誤差={recent_ang_err:.1f}°, 信頼度={recent_conf:.3f}")
            
            # 軌跡の長さと移動距離あたりの誤差
            rel_position_error = max_position_error / max(0.1, true_path_length) * 100.0
            print(f"移動距離: {true_path_length:.2f}m, 最大誤差比: {rel_position_error:.2f}%")
            print(f"処理時間: {elapsed_time:.3f}秒, 反復回数: {iterations}")
            print("=============================\n")
        
        # 点群を現在の推定位置に変換
        transformed_points = gicp_matcher.transform_points(scan_points, params)
        
        # 進捗表示
        if i % 10 == 0 or i == len(true_trajectory) - 1:
            print(f"ステップ {i}/{len(true_trajectory)-1} 完了 (スコア: {score:.4f}, 反復: {iterations}, 時間: {elapsed_time:.3f}秒)")
        
        # Rerunによる可視化
        if args.rerun:
            # フレーム時間の設定
            # rr.set_time_seconds("frame", float(i))
            rr.set_time(timestamp=float(i), timeline="frame")
            
            # 現在のスキャンポイントを可視化
            rr.log("world/scan_points", 
                  rr.Points2D(
                      positions=scan_points,
                      colors=np.full((len(scan_points), 3), [0.0, 1.0, 0.0]),  # 緑色
                      radii=0.05
                  ))
            
            # 変換後の点群を可視化
            rr.log("world/transformed_points", 
                  rr.Points2D(
                      positions=transformed_points,
                      colors=np.full((len(transformed_points), 3), [1.0, 0.5, 0.0]),  # オレンジ色
                      radii=0.05
                  ))
            
            # ターゲット点群を可視化
            rr.log("world/target_points", 
                  rr.Points2D(
                      positions=target_points,
                      colors=np.full((len(target_points), 3), [0.5, 0.5, 0.5]),  # グレー
                      radii=0.05
                  ))
                  
            # 対応関係も可視化
            if len(correspondences) > 0:
                # 対応点のラインを描画
                correspondence_lines = []
                correspondence_colors = []
                
                # サンプリング（すべての対応点を描画するとビジュアライズが重くなる）
                sample_size = min(50, len(correspondences))
                sampled_indices = np.random.choice(len(correspondences), sample_size, replace=False)
                
                for idx in sampled_indices:
                    src_idx, tgt_idx, _ = correspondences[idx]
                    # 変換後のソースポイント
                    src_point = transformed_points[src_idx]
                    # 対応するターゲットポイント
                    tgt_point = gicp_matcher.target_points[tgt_idx]
                    
                    # 対応点のラインを追加
                    correspondence_lines.append(np.array([src_point, tgt_point]))
                    
                    # 対応点の距離に基づいて色を決定（近いほど緑、遠いほど赤）
                    dist = np.linalg.norm(src_point - tgt_point)
                    if dist < 0.1:
                        correspondence_colors.append([0.0, 1.0, 0.0])  # 近い: 緑
                    elif dist < 0.3:
                        correspondence_colors.append([1.0, 1.0, 0.0])  # 中間: 黄色
                    else:
                        correspondence_colors.append([1.0, 0.0, 0.0])  # 遠い: 赤
                
                # 対応関係の線を描画
                rr.log("world/correspondences", 
                      rr.LineStrips2D(
                          strips=correspondence_lines,
                          colors=np.array(correspondence_colors),
                          radii=0.05
                      ))
            
            # 現在の推定位置を可視化（座標軸として）
            c, s = np.cos(current_estimate[2]), np.sin(current_estimate[2])
            x_axis = np.array([[current_estimate[0], current_estimate[1]], 
                              [current_estimate[0] + 0.5 * c, current_estimate[1] + 0.5 * s]])
            y_axis = np.array([[current_estimate[0], current_estimate[1]], 
                              [current_estimate[0] - 0.5 * s, current_estimate[1] + 0.5 * c]])
            
            rr.log("world/current_pose/x_axis", 
                  rr.LineStrips2D(
                      strips=[x_axis],
                      colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                      radii=0.1
                  ))
            rr.log("world/current_pose/y_axis", 
                  rr.LineStrips2D(
                      strips=[y_axis],
                      colors=np.array([[0.0, 1.0, 0.0]]),  # 緑色
                      radii=0.1
                  ))
            
            # 真の軌道と推定軌道を描画
            true_traj_array = np.array(true_trajectory[:i+1])
            est_traj_array = np.array(estimated_trajectory)
            
            # 配列に不足がないか確認
            if len(true_traj_array) > 0 and len(est_traj_array) > 0:
                # 確実に2次元以上の座標を持つことを確認
                true_points = true_traj_array[:, :2]
                est_points = est_traj_array[:, :2]
                
                rr.log("world/true_trajectory", 
                      rr.LineStrips2D(
                          strips=[true_points],
                          colors=np.array([[0.0, 0.0, 1.0]]),  # 青色
                          radii=0.1
                      ))
                
                rr.log("world/estimated_trajectory", 
                      rr.LineStrips2D(
                          strips=[est_points],
                          colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                          radii=0.1
                      ))
                
                # 現在の真の位置と推定位置を点で表示（位置のずれを視覚的に確認）
                if i > 0:
                    rr.log("world/current_true_position", 
                          rr.Points2D(
                              positions=np.array([true_traj_array[i, :2]]),
                              colors=np.array([[0.0, 0.0, 1.0]]),  # 青色
                              radii=0.15
                          ))
                    
                    rr.log("world/current_est_position", 
                          rr.Points2D(
                              positions=np.array([est_traj_array[-1, :2]]),
                              colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                              radii=0.15
                          ))
            
            # 少し待機（可視化のため）
            time.sleep(0.01)
    
    # matplotlib による最終結果の可視化
    if args.visualize:
        plt.figure(figsize=(10, 10))
        
        # 地図を描画
        binary_map = occupancy_map.get_occupancy_grid()
        origin_x, origin_y = occupancy_map.origin[0], occupancy_map.origin[1]
        extent = [origin_x, 
                 origin_x + binary_map.shape[1] * occupancy_map.resolution, 
                 origin_y, 
                 origin_y + binary_map.shape[0] * occupancy_map.resolution]
        plt.imshow(binary_map, cmap='gray', origin='lower', extent=extent)
        
        # 真の軌道と推定軌道を描画
        true_traj_array = np.array(true_trajectory)
        est_traj_array = np.array(estimated_trajectory)
        
        plt.plot(true_traj_array[:, 0], true_traj_array[:, 1], 'b-', label='真の軌道', linewidth=1.5)
        plt.plot(est_traj_array[:, 0], est_traj_array[:, 1], 'r-', label='推定軌道', linewidth=1.5)
        
        # 軌跡上に定期的にポイントを描画（動きの順序を視覚化）
        step = max(1, len(true_traj_array) // 20)  # 軌跡全体で約20ポイント表示
        for i in range(0, len(true_traj_array), step):
            plt.plot(true_traj_array[i, 0], true_traj_array[i, 1], 'bo', markersize=4)
            if i < len(est_traj_array):
                plt.plot(est_traj_array[i, 0], est_traj_array[i, 1], 'ro', markersize=4)
        
        # 開始位置と終了位置をマーク
        plt.plot(true_traj_array[0, 0], true_traj_array[0, 1], 'go', markersize=10, label='開始位置')
        plt.plot(true_traj_array[-1, 0], true_traj_array[-1, 1], 'mo', markersize=10, label='終了位置(真)')
        plt.plot(est_traj_array[-1, 0], est_traj_array[-1, 1], 'yo', markersize=10, label='終了位置(推定)')
        
        # 最終位置の誤差を線で表示
        plt.plot([true_traj_array[-1, 0], est_traj_array[-1, 0]], 
                [true_traj_array[-1, 1], est_traj_array[-1, 1]], 
                'k--', linewidth=1, label='最終誤差')
        
        plt.title('2D GICP Localization Results')
        plt.legend(loc='best')
        plt.grid(True)
        plt.axis('equal')
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.tight_layout()
        plt.show()
    
    # 位置推定の精度評価
    final_position_error = np.linalg.norm(true_trajectory[-1][:2] - estimated_trajectory[-1][:2])
    final_angle_error = abs(true_trajectory[-1][2] - estimated_trajectory[-1][2])
    final_angle_error = min(final_angle_error, 2 * np.pi - final_angle_error)  # 角度差を[0,π]の範囲に
    
    print("=== 位置推定結果 ===")
    print(f"最終位置誤差: {final_position_error:.4f} m")
    print(f"最終角度誤差: {np.degrees(final_angle_error):.4f} 度")

if __name__ == "__main__":
    main()
