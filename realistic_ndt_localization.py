#!/usr/bin/env python3
"""
実際のロボットに近い現実的な2D NDTローカライゼーションシミュレーター

このシミュレーターは以下の特徴を持ちます：
- 現実的なLiDARノイズモデル（距離依存、反射失敗、マルチパス）
- 現実的なオドメトリー誤差（車輪滑り、エンコーダー誤差、累積誤差）
- ROSメッセージとの互換性
- リアルタイム処理の制約
- 実際のロボット制御への応用を想定した設計
"""

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
from src.realistic_lidar_simulator import RealisticLidarSimulator
from src.robot_odometry import RobotOdometry, OdometryIntegrator
from src.ndt import NDTMatcher
from src.ros_compatibility import (
    ROSMessageConverter, LaserScan, Odometry, NDTMatchingResult
)

def generate_realistic_trajectory(start_pose, steps=100, max_velocity=0.5, max_angular_velocity=0.8):
    """
    実際のロボットに近い軌道を生成する（速度制限とスムースな動きを考慮）
    
    Args:
        start_pose (numpy.ndarray): 開始位置 [x, y, theta]
        steps (int): ステップ数
        max_velocity (float): 最大線速度 [m/s]
        max_angular_velocity (float): 最大角速度 [rad/s]
    
    Returns:
        tuple: (軌道のリスト, 制御コマンドのリスト)
    """
    trajectory = [start_pose.copy()]
    control_commands = []  # [(linear_vel, angular_vel), ...]
    current_pose = start_pose.copy()
    
    dt = 0.1  # 100ms刻みでの制御
    
    for i in range(steps):
        # スムースな速度変化を生成
        if i < steps // 3:
            # 加速段階
            linear_vel = min(max_velocity, 0.1 + 0.4 * i / (steps // 3))
        elif i < 2 * steps // 3:
            # 定速段階
            linear_vel = max_velocity
        else:
            # 減速段階
            remaining = steps - i
            linear_vel = max(0.1, max_velocity * remaining / (steps // 3))
        
        # スムースな方向変化
        phase = 2 * np.pi * i / steps
        angular_vel = max_angular_velocity * 0.3 * np.sin(phase)
        
        # 制御コマンドを記録
        control_commands.append((linear_vel, angular_vel))
        
        # 理想的な運動学で次の姿勢を計算
        if abs(angular_vel) < 1e-6:
            # 直線運動
            dx = linear_vel * dt * np.cos(current_pose[2])
            dy = linear_vel * dt * np.sin(current_pose[2])
            dtheta = 0.0
        else:
            # 曲線運動
            radius = linear_vel / angular_vel
            dtheta = angular_vel * dt
            dx = radius * (np.sin(current_pose[2] + dtheta) - np.sin(current_pose[2]))
            dy = radius * (-np.cos(current_pose[2] + dtheta) + np.cos(current_pose[2]))
        
        current_pose[0] += dx
        current_pose[1] += dy
        current_pose[2] += dtheta
        current_pose[2] = np.arctan2(np.sin(current_pose[2]), np.cos(current_pose[2]))
        
        trajectory.append(current_pose.copy())
    
    return trajectory, control_commands

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

class RealisticNDTLocalizer:
    """
    実際のロボットに近い現実的なNDTローカライザークラス
    """
    def __init__(self, occupancy_map, config=None):
        """
        初期化関数
        
        Args:
            occupancy_map: 占有格子地図オブジェクト
            config (dict): 設定パラメータ
        """
        if config is None:
            config = {}
        
        self.occupancy_map = occupancy_map
        self.config = config
        
        # 現実的なLiDARシミュレーターの初期化
        self.lidar_sim = RealisticLidarSimulator(
            occupancy_map,
            num_beams=config.get('num_beams', 360),
            max_range=config.get('max_range', 10.0),
            min_range=config.get('min_range', 0.1),
            base_noise_sigma=config.get('base_noise_sigma', 0.01),
            distance_noise_factor=config.get('distance_noise_factor', 0.001),
            beam_divergence=config.get('beam_divergence', 0.003),
            reflection_prob=config.get('reflection_prob', 0.98),
            multipath_prob=config.get('multipath_prob', 0.02)
        )
        
        # オドメトリーシステムの初期化
        self.odometry = RobotOdometry(
            wheel_base=config.get('wheel_base', 0.5),
            systematic_error_factor=config.get('odom_systematic_error', 0.02),
            random_error_factor=config.get('odom_random_error', 0.01),
            slip_factor=config.get('slip_factor', 0.05)
        )
        
        # 参照用の点群マップを生成
        self.target_points = create_target_map(occupancy_map)
        
        # NDTマッチャーの初期化
        self.ndt_matcher = NDTMatcher(
            self.target_points,
            cell_size=config.get('cell_size', 0.5),
            max_iterations=config.get('max_iterations', 60),
            tolerance=config.get('tolerance', 1e-3),
            step_size=config.get('step_size', 0.1)
        )
        
        # ローカライゼーション状態
        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.pose_covariance = np.eye(3) * 0.1  # 初期の不確実性
        self.last_update_time = time.time()
        
        # 統計情報
        self.stats = {
            'successful_matches': 0,
            'failed_matches': 0,
            'total_computation_time': 0.0,
            'average_iterations': 0.0,
            'position_errors': [],
            'angle_errors': []
        }
        
        # ROSメッセージコンバーター
        self.ros_converter = ROSMessageConverter()
    
    def initialize_pose(self, initial_pose):
        """
        初期姿勢を設定
        
        Args:
            initial_pose (numpy.ndarray): 初期姿勢 [x, y, theta]
        """
        self.current_pose = initial_pose.copy()
        self.odometry.set_pose(initial_pose)
        
    def process_scan_and_odometry(self, true_pose, control_command, dt):
        """
        LiDARスキャンとオドメトリーデータを処理してローカライゼーションを実行
        
        Args:
            true_pose (numpy.ndarray): 真の姿勢（シミュレーション用）
            control_command (tuple): 制御コマンド (linear_vel, angular_vel)
            dt (float): 時間間隔
        
        Returns:
            dict: ローカライゼーション結果
        """
        linear_vel, angular_vel = control_command
        current_time = time.time()
        
        # オドメトリーを更新
        delta_odom, odom_uncertainty = self.odometry.update_from_motion_command(
            linear_vel, angular_vel, dt
        )
        
        # 現実的なLiDARスキャンをシミュレート
        scan_data = self.lidar_sim.simulate_scan(true_pose, current_time)
        
        # ROSメッセージ形式に変換
        laser_scan_msg = self.ros_converter.scan_data_to_ros(scan_data)
        odometry_msg = Odometry.from_pose_and_velocity(
            self.odometry.get_pose(),
            np.array([linear_vel, angular_vel]),
            odom_uncertainty
        )
        
        # NDTマッチングを実行
        ndt_result = self._perform_ndt_matching(scan_data, delta_odom)
        
        # 姿勢を更新
        if ndt_result['success']:
            self._update_pose_with_ndt(ndt_result, odom_uncertainty)
        else:
            # NDTマッチングが失敗した場合はオドメトリーのみで更新
            self.current_pose = self.odometry.get_pose()
        
        # 統計情報を更新
        self._update_statistics(true_pose, ndt_result)
        
        # 結果を返す
        return {
            'estimated_pose': self.current_pose.copy(),
            'pose_covariance': self.pose_covariance.copy(),
            'laser_scan': laser_scan_msg,
            'odometry': odometry_msg,
            'ndt_result': ndt_result,
            'scan_data': scan_data,
            'sensor_stats': self.lidar_sim.get_statistics()
        }
    
    def _perform_ndt_matching(self, scan_data, odom_delta):
        """
        NDTマッチングを実行
        
        Args:
            scan_data (dict): スキャンデータ
            odom_delta (numpy.ndarray): オドメトリー移動量
        
        Returns:
            dict: NDTマッチング結果
        """
        scan_points = scan_data['points']
        
        if len(scan_points) < 5:
            return {
                'success': False,
                'params': np.zeros(3),
                'score': 0.0,
                'iterations': 0,
                'computation_time': 0.0
            }
        
        # 初期推定値をオドメトリーから設定
        initial_guess = np.array([odom_delta[2], odom_delta[0], odom_delta[1]])  # [theta, x, y]
        
        # NDTマッチング実行
        start_time = time.time()
        
        if self.config.get('use_newton', False):
            params, score, iterations = self.ndt_matcher.align_newton(
                scan_points, initial_guess, 
                damping=self.config.get('damping', 0.1)
            )
        else:
            params, score, iterations = self.ndt_matcher.align_gradient_descent(
                scan_points, initial_guess
            )
        
        computation_time = (time.time() - start_time) * 1000  # ミリ秒
        
        # 収束判定
        success = iterations < self.ndt_matcher.max_iterations and score > 0.01
        
        return {
            'success': success,
            'params': params,
            'score': score,
            'iterations': iterations,
            'computation_time': computation_time
        }
    
    def _update_pose_with_ndt(self, ndt_result, odom_uncertainty):
        """
        NDTマッチング結果で姿勢を更新
        
        Args:
            ndt_result (dict): NDTマッチング結果
            odom_uncertainty (numpy.ndarray): オドメトリーの不確実性
        """
        params = ndt_result['params']
        
        # 姿勢を更新
        self.current_pose[0] += params[1]  # x
        self.current_pose[1] += params[2]  # y
        self.current_pose[2] += params[0]  # theta
        self.current_pose[2] = np.arctan2(np.sin(self.current_pose[2]), np.cos(self.current_pose[2]))
        
        # 共分散行列を更新（簡略化されたモデル）
        ndt_uncertainty = np.eye(3) * (1.0 / max(0.01, ndt_result['score']))
        
        # カルマンフィルターのような更新（簡単な重み付け平均）
        weight = min(0.8, ndt_result['score'] / 10.0)
        self.pose_covariance = (1 - weight) * odom_uncertainty + weight * ndt_uncertainty
    
    def _update_statistics(self, true_pose, ndt_result):
        """
        統計情報を更新
        
        Args:
            true_pose (numpy.ndarray): 真の姿勢
            ndt_result (dict): NDTマッチング結果
        """
        if ndt_result['success']:
            self.stats['successful_matches'] += 1
        else:
            self.stats['failed_matches'] += 1
        
        self.stats['total_computation_time'] += ndt_result['computation_time']
        
        # 位置誤差と角度誤差を計算
        pos_error = np.linalg.norm(true_pose[:2] - self.current_pose[:2])
        angle_diff = abs(true_pose[2] - self.current_pose[2])
        angle_error = min(angle_diff, 2*np.pi - angle_diff)
        
        self.stats['position_errors'].append(pos_error)
        self.stats['angle_errors'].append(angle_error)
        
        # 平均反復回数を更新
        total_matches = self.stats['successful_matches'] + self.stats['failed_matches']
        if total_matches > 0:
            self.stats['average_iterations'] = (
                self.stats['average_iterations'] * (total_matches - 1) + ndt_result['iterations']
            ) / total_matches
    
    def get_statistics(self):
        """統計情報を取得"""
        total_matches = self.stats['successful_matches'] + self.stats['failed_matches']
        if total_matches == 0:
            return self.stats
        
        return {
            **self.stats,
            'success_rate': self.stats['successful_matches'] / total_matches,
            'average_computation_time': self.stats['total_computation_time'] / total_matches,
            'final_position_error': self.stats['position_errors'][-1] if self.stats['position_errors'] else 0.0,
            'final_angle_error': self.stats['angle_errors'][-1] if self.stats['angle_errors'] else 0.0,
            'mean_position_error': np.mean(self.stats['position_errors']) if self.stats['position_errors'] else 0.0,
            'mean_angle_error': np.mean(self.stats['angle_errors']) if self.stats['angle_errors'] else 0.0
        }

def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(description='現実的な2D NDTローカライゼーションシミュレーター')
    parser.add_argument('--map_dir', type=str, default='map',
                        help='マップファイル(.pgmと.yaml)が格納されているディレクトリ')
    parser.add_argument('--steps', type=int, default=100,
                        help='シミュレーションのステップ数')
    parser.add_argument('--rerun', action='store_true', default=True,
                        help='Rerunを使用して可視化')
    parser.add_argument('--newton', action='store_true',
                        help='ニュートン法を使用してNDTマッチングを実行')
    parser.add_argument('--realistic_noise', action='store_true', default=True,
                        help='現実的なノイズモデルを使用')
    parser.add_argument('--max_velocity', type=float, default=0.5,
                        help='最大線速度 [m/s]')
    parser.add_argument('--max_angular_velocity', type=float, default=0.8,
                        help='最大角速度 [rad/s]')
    args = parser.parse_args()
    
    # 地図ファイルのパスを構築
    map_dir = Path(args.map_dir)
    pgm_files = list(map_dir.glob('*.pgm'))
    if not pgm_files:
        print(f"エラー: {map_dir} に .pgm ファイルが見つかりません")
        return
    
    pgm_path = pgm_files[0]
    yaml_path = pgm_path.with_suffix('.yaml')
    
    if not yaml_path.exists():
        print(f"エラー: YAML ファイルが見つかりません: {yaml_path}")
        return
    
    # Rerunのセットアップ
    if args.rerun:
        rr.init("現実的な2D NDTローカライゼーション", spawn=True)
        rr.set_time(timestamp=0.0, timeline="frame")
    
    # 占有格子地図を読み込み
    occupancy_map = OccupancyGridMap(str(pgm_path), str(yaml_path))
    
    # 設定パラメータ
    config = {
        'num_beams': 360,
        'max_range': 10.0,
        'min_range': 0.1,
        'base_noise_sigma': 0.02 if args.realistic_noise else 0.01,
        'distance_noise_factor': 0.002 if args.realistic_noise else 0.001,
        'beam_divergence': 0.005 if args.realistic_noise else 0.001,
        'reflection_prob': 0.95 if args.realistic_noise else 0.98,
        'multipath_prob': 0.05 if args.realistic_noise else 0.02,
        'use_newton': args.newton,
        'cell_size': 0.5,
        'max_iterations': 30,
        'wheel_base': 0.5,
        'odom_systematic_error': 0.03 if args.realistic_noise else 0.01,
        'odom_random_error': 0.02 if args.realistic_noise else 0.005,
        'slip_factor': 0.08 if args.realistic_noise else 0.02
    }
    
    # ローカライザーの初期化
    localizer = RealisticNDTLocalizer(occupancy_map, config)
    
    # 初期位置を設定
    start_position = [0.0, 0.0]
    start_pose = np.array([start_position[0], start_position[1], 0.0])
    localizer.initialize_pose(start_pose)
    
    # 現実的な軌道を生成
    print(f"現実的な軌道を生成: {args.steps}ステップ")
    true_trajectory, control_commands = generate_realistic_trajectory(
        start_pose, args.steps, args.max_velocity, args.max_angular_velocity
    )
    
    # 軌道履歴
    estimated_trajectory = [start_pose.copy()]
    
    print("シミュレーション開始...")
    print(f"アルゴリズム: {'ニュートン法' if args.newton else '勾配降下法'}")
    print(f"ノイズレベル: {'現実的' if args.realistic_noise else '低'}")
    
    dt = 0.1  # 100ms刻み
    
    # シミュレーション実行
    for i in range(len(control_commands)):
        true_pose = true_trajectory[i + 1]
        control_command = control_commands[i]
        
        # ローカライゼーション処理
        result = localizer.process_scan_and_odometry(true_pose, control_command, dt)
        
        # 軌道を記録
        estimated_trajectory.append(result['estimated_pose'])
        
        # 進捗表示
        if i % 10 == 0 or i == len(control_commands) - 1:
            print(f"ステップ {i+1}/{len(control_commands)} 完了 "
                  f"(成功: {result['ndt_result']['success']}, "
                  f"スコア: {result['ndt_result']['score']:.4f})")
        
        # Rerunによる可視化
        if args.rerun:
            rr.set_time(timestamp=i * dt, timeline="frame")
            
            # 現実的なLiDARデータで可視化
            localizer.lidar_sim.visualize_scan_with_rerun(
                result['estimated_pose'],
                result['scan_data'],
                target_points=localizer.target_points
            )
            
            # 軌道を描画
            true_traj_array = np.array(true_trajectory[:i+2])
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
            
            # 真の位置と推定位置の比較
            true_x, true_y, true_theta = true_pose
            rr.log("world/true_pose",
                  rr.Points2D(
                      positions=np.array([[true_x, true_y]]),
                      colors=np.array([[0.0, 0.0, 1.0]]),  # 青色
                      radii=0.25
                  ))
            
            # 不確実性楕円（簡略化）
            est_x, est_y = result['estimated_pose'][:2]
            cov = result['pose_covariance']
            uncertainty_radius = np.sqrt(cov[0, 0] + cov[1, 1])
            
            rr.log("world/uncertainty",
                  rr.Points2D(
                      positions=np.array([[est_x, est_y]]),
                      colors=np.array([[1.0, 1.0, 0.0]]),  # 黄色
                      radii=uncertainty_radius
                  ))
            
            time.sleep(0.05)  # リアルタイム可視化のための適度な待機
    
    # 最終統計情報を表示
    stats = localizer.get_statistics()
    sensor_stats = localizer.lidar_sim.get_statistics()
    odom_stats = localizer.odometry.get_error_statistics()
    
    print("\n=== 最終結果 ===")
    print(f"最終位置誤差: {stats['final_position_error']:.4f} m")
    print(f"最終角度誤差: {np.degrees(stats['final_angle_error']):.4f} 度")
    print(f"平均位置誤差: {stats['mean_position_error']:.4f} m")
    print(f"平均角度誤差: {np.degrees(stats['mean_angle_error']):.4f} 度")
    print(f"NDT成功率: {stats['success_rate']:.2%}")
    print(f"平均計算時間: {stats['average_computation_time']:.2f} ms")
    
    print("\n=== センサー統計 ===")
    print(f"LiDAR検出率: {sensor_stats['detection_rate']:.2%}")
    print(f"反射失敗率: {sensor_stats['failure_rate']:.2%}")
    print(f"マルチパス率: {sensor_stats['multipath_rate']:.2%}")
    
    print("\n=== オドメトリー統計 ===")
    print(f"累積移動距離: {odom_stats['cumulative_distance']:.2f} m")
    print(f"推定位置誤差: {odom_stats['position_error_estimate']:.4f} m")
    print(f"推定角度誤差: {np.degrees(odom_stats['angle_error_estimate']):.4f} 度")

if __name__ == "__main__":
    main() 