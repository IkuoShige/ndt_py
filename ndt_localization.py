#!/usr/bin/env python3
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
from src.ndt import NDTMatcher

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

def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description='2D NDT Localization Simulator')
    parser.add_argument('--map_dir', type=str, default='map',
                        help='マップファイル(.pgmと.yaml)が格納されているディレクトリ')
    parser.add_argument('--visualize', action='store_true',
                        help='matplotlibで結果を可視化')
    parser.add_argument('--steps', type=int, default=100,
                        help='シミュレーションのステップ数')
    parser.add_argument('--rerun', action='store_true', default=True,
                        help='Rerunを使用して可視化（デフォルトはオン）')
    parser.add_argument('--newton', action='store_true',
                        help='ニュートン法を使用してNDTマッチングを実行')
    parser.add_argument('--damping', type=float, default=0.1,
                        help='ニュートン法のダンピング係数（デフォルト: 0.1）')
    parser.add_argument('--trajectory', type=str, default='random', choices=['random', 'circular', 'spiral'],
                        help='生成する軌道の種類（random: ランダム軌道、circular: 円軌道、spiral: らせん軌道）')
    parser.add_argument('--radius', type=float, default=5.0,
                        help='円軌道またはらせん軌道の半径（デフォルト: 5.0）')
    parser.add_argument('--end_radius', type=float, default=1.0,
                        help='らせん軌道の終了半径（デフォルト: 1.0）')
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
        rr.init("2D NDT Localization", spawn=True)
        # 新しいAPI（Rerun 0.23以降）を使用
        rr.set_time(timestamp=0.0, timeline="frame")
    
    # 占有格子地図を読み込み
    occupancy_map = OccupancyGridMap(str(pgm_path), str(yaml_path))
    
    # LiDARシミュレーターの初期化
    lidar_sim = LidarSimulator(occupancy_map, num_beams=360, max_range=10.0)
    
    # 参照用の点群マップを生成
    target_points = create_target_map(occupancy_map)
    
    # NDTマッチャーの初期化
    ndt_matcher = NDTMatcher(target_points, cell_size=0.5, max_iterations=60, tolerance=1e-2, step_size=0.1)
    
    # 初期位置を空きスペースからランダムに選択
    # start_position = occupancy_map.get_empty_positions(1)[0]
    start_position = [0.0, 0.0]  # デフォルトの開始位置
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
    print(f"初期位置: {current_estimate}")
    
    # シミュレーション開始
    print("シミュレーション開始...")
    print(f"アルゴリズム: {'ニュートン法' if args.newton else '勾配降下法'}")

    for i in range(1, len(true_trajectory)):
        true_pose = true_trajectory[i]
        
        # 真の位置からのLiDARスキャンをシミュレート
        _, scan_points = lidar_sim.simulate_scan(true_pose)
        
        if len(scan_points) < 5:  # スキャンポイントが少なすぎる場合はスキップ
            print(f"警告: ステップ {i} でスキャンポイントが不足しています。前回の推定値を使用します。")
            estimated_trajectory.append(current_estimate.copy())
            continue
        
        # 前回の推定位置を初期推定値として使用
        delta_pose = true_trajectory[i] - true_trajectory[i-1]  # 真の移動量
        initial_guess = np.array([delta_pose[2], delta_pose[0], delta_pose[1]])  # [theta, x, y]
        
        # 測定値に基づいてNDTマッチングを実行（ニュートン法または勾配降下法）
        if args.newton:
            params, score, iterations = ndt_matcher.align_newton(scan_points, initial_guess, damping=args.damping)
        else:
            params, score, iterations = ndt_matcher.align_gradient_descent(scan_points, initial_guess)
        
        # 推定位置を更新
        current_estimate[0] += params[1]  # x方向の更新
        current_estimate[1] += params[2]  # y方向の更新
        current_estimate[2] += params[0]  # 角度の更新
        
        # 角度を-piからpiの範囲に正規化
        current_estimate[2] = np.arctan2(np.sin(current_estimate[2]), np.cos(current_estimate[2]))
        
        # 推定軌道に追加
        estimated_trajectory.append(current_estimate.copy())
        
        # 点群を現在の推定位置に変換
        transformed_points = ndt_matcher.transform_points(scan_points, params)
        
        # 進捗表示
        if i % 10 == 0 or i == len(true_trajectory) - 1:
            print(f"ステップ {i}/{len(true_trajectory)-1} 完了 (スコア: {score:.4f}, 反復: {iterations})")
        
        # Rerunによる可視化
        if args.rerun:
            rr.set_time(timestamp=float(i), timeline="frame")

            lidar_sim.visualize_with_rerun(
                current_estimate, scan_points, 
                target_points=target_points, 
                transformed_points=transformed_points
            )
            
            # 真の軌道と推定軌道を描画
            true_traj_array = np.array(true_trajectory[:i+1])
            est_traj_array = np.array(estimated_trajectory)
            
            rr.log("world/true_trajectory", 
                  rr.LineStrips2D(
                      strips=[true_traj_array[:, :2]],
                      colors=np.array([[0.0, 0.0, 1.0]]),  # 青色
                      radii=0.1
                  ))
            
            rr.log("world/estimated_trajectory", 
                  rr.LineStrips2D(
                      strips=[est_traj_array[:, :2]],
                      colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                      radii=0.1
                  ))
            
            true_x, true_y, true_theta = true_pose
            rr.log("world/true_pose",
              rr.Points2D(
                  positions=np.array([[true_x, true_y]]),
                  colors=np.array([[0.0, 0.0, 1.0]]),  # 青色
                  radii=0.25
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
        
        plt.plot(true_traj_array[:, 0], true_traj_array[:, 1], 'b-', label='真の軌道')
        plt.plot(est_traj_array[:, 0], est_traj_array[:, 1], 'r-', label='推定軌道')
        
        # 開始位置と終了位置をマーク
        plt.plot(true_traj_array[0, 0], true_traj_array[0, 1], 'go', markersize=10, label='開始位置')
        plt.plot(true_traj_array[-1, 0], true_traj_array[-1, 1], 'mo', markersize=10, label='終了位置(真)')
        plt.plot(est_traj_array[-1, 0], est_traj_array[-1, 1], 'yo', markersize=10, label='終了位置(推定)')
        
        plt.title('2D NDT Localization Results')
        plt.legend()
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
