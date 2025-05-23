#!/usr/bin/env python3
"""
NDT位置推定の性能評価とパラメータ分析のための実験データ収集スクリプト

このスクリプトは様々なパラメータ設定でNDT位置推定を実行し、性能データを収集します。
結果はCSVファイルとして保存され、後で分析できるようになっています。
"""

import os
import sys
import csv
import time
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import traceback

try:
    from tqdm import tqdm
except ImportError:
    print("tqdmモジュールがインストールされていません。進捗バーは表示されません。")
    # tqdmの簡易代替関数
    def tqdm(iterable, **kwargs):
        if 'desc' in kwargs:
            print(f"{kwargs['desc']}...")
        return iterable

import multiprocessing as mp
from itertools import product

# 自作モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ndt_localization import (
    OccupancyGridMap, LidarSimulator, NDTMatcher,
    generate_random_trajectory, generate_circular_trajectory, generate_spiral_trajectory,
    create_target_map
)

# 結果を保存するディレクトリ
RESULTS_DIR = "experiment_results"

class ExperimentRunner:
    """NDT位置推定の実験を実行し、データを収集するクラス"""
    
    def __init__(self, map_path, yaml_path, results_dir=RESULTS_DIR):
        """
        初期化
        
        Args:
            map_path (str): 占有格子地図のパス (.pgm)
            yaml_path (str): 地図メタデータのパス (.yaml)
            results_dir (str): 結果を保存するディレクトリのパス
        """
        self.map_path = map_path
        self.yaml_path = yaml_path
        self.results_dir = results_dir
        
        # 結果保存ディレクトリが存在しない場合は作成
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 占有格子地図を読み込み
        self.occupancy_map = OccupancyGridMap(map_path, yaml_path)
        
    def _generate_trajectory(self, trajectory_type, start_pose, steps=100, **kwargs):
        """
        指定された種類の軌道を生成
        
        Args:
            trajectory_type (str): 軌道の種類 ('random', 'circular', 'spiral')
            start_pose (numpy.ndarray): 開始位置 [x, y, theta]
            steps (int): ステップ数
            **kwargs: 軌道生成関数に渡す追加のパラメータ
            
        Returns:
            list: 軌道のリスト（姿勢のリスト）
        """
        if trajectory_type == 'random':
            return generate_random_trajectory(start_pose, steps=steps, **kwargs)
        elif trajectory_type == 'circular':
            center = kwargs.get('center', start_pose[:2])
            radius = kwargs.get('radius', 5.0)
            return generate_circular_trajectory(center, radius, steps=steps)
        elif trajectory_type == 'spiral':
            center = kwargs.get('center', start_pose[:2])
            start_radius = kwargs.get('start_radius', 5.0)
            end_radius = kwargs.get('end_radius', 1.0)
            return generate_spiral_trajectory(center, start_radius, end_radius, steps=steps)
        else:
            raise ValueError(f"不明な軌道タイプ: {trajectory_type}")
    
    def run_single_experiment(self, config):
        """
        1つの実験設定で位置推定を実行し、結果を返す
        
        Args:
            config (dict): 実験設定
            
        Returns:
            dict: 実験結果
        """
        # パラメータの展開
        cell_size = config.get('cell_size', 0.5)
        max_iterations = config.get('max_iterations', 50)
        use_newton = config.get('use_newton', False)
        damping = config.get('damping', 0.1)
        num_beams = config.get('num_beams', 360)
        max_range = config.get('max_range', 10.0)
        noise_sigma = config.get('noise_sigma', 0.05)
        trajectory_type = config.get('trajectory_type', 'random')
        steps = config.get('steps', 100)
        initial_offset_distance = config.get('initial_offset_distance', 0.0)
        initial_offset_angle = config.get('initial_offset_angle', 0.0)
        sampling_rate = config.get('sampling_rate', 1.0)  # サンプリング率（1.0 = 全点使用）
        
        # 結果を格納する辞書
        result = config.copy()  # 設定をコピー
        result['experiment_id'] = config.get('experiment_id', 'unknown')
        
        # 開始時間
        start_time = time.time()
        
        # 参照用の点群マップを生成
        target_points = create_target_map(self.occupancy_map)
        
        # NDTマッチャーの初期化
        ndt_matcher = NDTMatcher(target_points, cell_size=cell_size, max_iterations=max_iterations)
        
        # LiDARシミュレーターの初期化
        lidar_sim = LidarSimulator(
            self.occupancy_map, 
            num_beams=num_beams,
            max_range=max_range,
            noise_sigma=noise_sigma
        )
        
        # 初期位置を空きスペースからランダムに選択
        start_position = self.occupancy_map.get_empty_positions(1)[0]
        start_pose = np.array([start_position[0], start_position[1], 0.0])  # [x, y, theta]
        
        # 軌道の生成
        if trajectory_type == 'circular':
            kwargs = {'center': start_position, 'radius': 5.0}
        elif trajectory_type == 'spiral':
            kwargs = {'center': start_position, 'start_radius': 5.0, 'end_radius': 1.0}
        else:
            kwargs = {}
            
        true_trajectory = self._generate_trajectory(
            trajectory_type, start_pose, steps=steps, **kwargs)
        
        # 初期推定値に意図的にオフセットを追加（初期値の影響を調査）
        initial_pose = true_trajectory[0].copy()
        if initial_offset_distance > 0:
            offset_angle = np.random.uniform(0, 2*np.pi)
            initial_pose[0] += initial_offset_distance * np.cos(offset_angle)
            initial_pose[1] += initial_offset_distance * np.sin(offset_angle)
            
        if initial_offset_angle > 0:
            angle_offset = np.random.uniform(-initial_offset_angle, initial_offset_angle)
            initial_pose[2] += angle_offset
            
        estimated_trajectory = [initial_pose.copy()]
        current_estimate = initial_pose.copy()
        
        # 評価指標の初期化
        position_errors = [0.0]  # 位置誤差
        angle_errors = [0.0]     # 角度誤差
        process_times = []       # 処理時間
        iteration_counts = []    # 反復回数
        scores = []              # スコア値
        success_count = 0        # 収束成功回数
        
        # シミュレーション実行
        for i in range(1, len(true_trajectory)):
            true_pose = true_trajectory[i]
            
            # 真の位置からのLiDARスキャンをシミュレート
            scan_start_time = time.time()
            _, scan_points = lidar_sim.simulate_scan(true_pose)
            
            # 点群のサブサンプリング（必要に応じて）
            if sampling_rate < 1.0 and len(scan_points) > 10:
                sample_count = max(5, int(len(scan_points) * sampling_rate))
                indices = np.random.choice(len(scan_points), sample_count, replace=False)
                scan_points = [scan_points[idx] for idx in indices]
            
            if len(scan_points) < 5:  # スキャンポイントが少なすぎる場合はスキップ
                estimated_trajectory.append(current_estimate.copy())
                
                # 位置誤差と角度誤差を計算して記録
                pos_error = np.linalg.norm(true_pose[:2] - current_estimate[:2])
                angle_diff = abs(true_pose[2] - current_estimate[2])
                angle_error = min(angle_diff, 2*np.pi - angle_diff)
                
                position_errors.append(pos_error)
                angle_errors.append(angle_error)
                continue
            
            # 前回の推定位置を初期推定値として使用
            # オドメトリの代わりに真の移動量から初期推定値を計算
            delta_pose = true_trajectory[i] - true_trajectory[i-1]  # 真の移動量
            initial_guess = np.array([delta_pose[2], delta_pose[0], delta_pose[1]])  # [theta, x, y]
            
            # NDTマッチング実行前の時間
            match_start_time = time.time()
            
            # 測定値に基づいてNDTマッチングを実行
            if use_newton:
                params, score, iterations = ndt_matcher.align_newton(scan_points, initial_guess, damping=damping)
            else:
                params, score, iterations = ndt_matcher.align(scan_points, initial_guess)
                
            # 処理時間を記録
            process_time = (time.time() - match_start_time) * 1000  # ミリ秒単位
            process_times.append(process_time)
            
            # 反復回数とスコアを記録
            iteration_counts.append(iterations)
            scores.append(score)
            
            # 最大反復回数に達していなければ収束成功とみなす
            if iterations < max_iterations:
                success_count += 1
            
            # 推定位置を更新
            current_estimate[0] += params[1]  # x方向の更新
            current_estimate[1] += params[2]  # y方向の更新
            current_estimate[2] += params[0]  # 角度の更新
            
            # 角度を-piからpiの範囲に正規化
            current_estimate[2] = np.arctan2(np.sin(current_estimate[2]), np.cos(current_estimate[2]))
            
            # 推定軌道に追加
            estimated_trajectory.append(current_estimate.copy())
            
            # 位置誤差と角度誤差を計算して記録
            pos_error = np.linalg.norm(true_pose[:2] - current_estimate[:2])
            angle_diff = abs(true_pose[2] - current_estimate[2])
            angle_error = min(angle_diff, 2*np.pi - angle_diff)
            
            position_errors.append(pos_error)
            angle_errors.append(angle_error)
        
        # 総処理時間
        total_time = time.time() - start_time
        
        # 結果を辞書に格納
        result.update({
            'avg_position_error': np.mean(position_errors[1:]),  # 平均位置誤差（初期位置を除く）
            'max_position_error': np.max(position_errors[1:]),   # 最大位置誤差
            'final_position_error': position_errors[-1],         # 最終位置誤差
            'avg_angle_error_deg': np.mean(angle_errors[1:]) * 180/np.pi,  # 平均角度誤差（度）
            'max_angle_error_deg': np.max(angle_errors[1:]) * 180/np.pi,   # 最大角度誤差（度）
            'final_angle_error_deg': angle_errors[-1] * 180/np.pi,         # 最終角度誤差（度）
            'avg_process_time_ms': np.mean(process_times),       # 平均処理時間（ミリ秒）
            'total_time_s': total_time,                         # 総処理時間（秒）
            'avg_iterations': np.mean(iteration_counts),         # 平均反復回数
            'max_iterations': np.max(iteration_counts),          # 最大反復回数
            'avg_score': np.mean(scores),                        # 平均スコア
            'convergence_rate': success_count / (len(true_trajectory) - 1),  # 収束成功率
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # タイムスタンプ
        })
        
        return result
    
    def run_cell_size_experiment(self, num_experiments=5):
        """
        セルサイズの影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/cell_size_experiment_{timestamp}.csv"
        
        # セルサイズのリスト
        cell_sizes = [0.2, 0.5, 1.0, 2.0, 5.0]
        
        # ヘッダー行
        header = [
            'experiment_id', 'cell_size', 'use_newton', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        for cell_size in cell_sizes:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'cell_size_{cell_size}_{exp_id}',
                    'cell_size': cell_size,
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50  # ステップ数を短くして実験時間を短縮
                }
                configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="セルサイズ実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_optimization_method_experiment(self, num_experiments=5):
        """
        最適化アルゴリズム（勾配降下法とニュートン法）の比較実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/optimization_method_experiment_{timestamp}.csv"
        
        # ヘッダー行
        header = [
            'experiment_id', 'use_newton', 'damping', 'trajectory_type',
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        # 軌道タイプ
        trajectory_types = ['random', 'circular', 'spiral']
        
        configs = []
        for use_newton in [False, True]:
            for traj_type in trajectory_types:
                for exp_id in range(num_experiments):
                    # 実験設定
                    config = {
                        'experiment_id': f'opt_{use_newton}_{traj_type}_{exp_id}',
                        'cell_size': 0.5,  # 固定セルサイズ
                        'use_newton': use_newton,
                        'damping': 0.1 if use_newton else 0.0,
                        'trajectory_type': traj_type,
                        'steps': 50  # ステップ数を短くして実験時間を短縮
                    }
                    configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="最適化手法実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_damping_experiment(self, num_experiments=5):
        """
        ダンピング係数の影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/damping_experiment_{timestamp}.csv"
        
        # ダンピング係数のリスト
        damping_values = [0.001, 0.01, 0.1, 1.0]
        
        # ヘッダー行
        header = [
            'experiment_id', 'damping', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp', 'numerical_instability'
        ]
        
        configs = []
        for damping in damping_values:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'damping_{damping}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': damping,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50  # ステップ数を短くして実験時間を短縮
                }
                configs.append(config)
        
        # 実験を実行（数値的不安定性の発生を追跡）
        results = []
        for config in tqdm(configs, desc="ダンピング係数実験"):
            try:
                result = self.run_single_experiment(config)
                result['numerical_instability'] = 0  # 問題なし
                results.append(result)
            except Exception as e:
                # 数値的不安定性による例外を捕捉
                print(f"数値的不安定性を検出: {str(e)}")
                config_copy = config.copy()
                config_copy['numerical_instability'] = 1  # 問題あり
                # 最低限のデータを記録
                config_copy.update({
                    'avg_position_error': float('nan'),
                    'max_position_error': float('nan'),
                    'final_position_error': float('nan'),
                    'avg_angle_error_deg': float('nan'),
                    'max_angle_error_deg': float('nan'),
                    'final_angle_error_deg': float('nan'),
                    'avg_process_time_ms': float('nan'),
                    'total_time_s': float('nan'),
                    'avg_iterations': float('nan'),
                    'max_iterations': float('nan'),
                    'avg_score': float('nan'),
                    'convergence_rate': 0.0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                results.append(config_copy)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_noise_experiment(self, num_experiments=5):
        """
        ノイズレベルの影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/noise_experiment_{timestamp}.csv"
        
        # ノイズレベルのリスト
        noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
        
        # ヘッダー行
        header = [
            'experiment_id', 'noise_sigma', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        for noise_sigma in noise_levels:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'noise_{noise_sigma}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'noise_sigma': noise_sigma,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50  # ステップ数を短くして実験時間を短縮
                }
                configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="ノイズレベル実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_beam_count_experiment(self, num_experiments=5):
        """
        ビーム数（点群密度）の影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/beam_count_experiment_{timestamp}.csv"
        
        # ビーム数のリスト
        beam_counts = [60, 120, 240, 360, 720]
        
        # ヘッダー行
        header = [
            'experiment_id', 'num_beams', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        for num_beams in beam_counts:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'beams_{num_beams}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'num_beams': num_beams,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50  # ステップ数を短くして実験時間を短縮
                }
                configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="ビーム数実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_trajectory_experiment(self, num_experiments=5):
        """
        軌道パターンによる性能評価実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/trajectory_experiment_{timestamp}.csv"
        
        # 軌道タイプ
        trajectory_types = ['random', 'circular', 'spiral']
        
        # ヘッダー行
        header = [
            'experiment_id', 'trajectory_type', 'use_newton',
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        for traj_type in trajectory_types:
            for use_newton in [False, True]:
                for exp_id in range(num_experiments):
                    # 実験設定
                    config = {
                        'experiment_id': f'trajectory_{traj_type}_{use_newton}_{exp_id}',
                        'cell_size': 0.5,  # 固定セルサイズ
                        'use_newton': use_newton,
                        'damping': 0.1 if use_newton else 0.0,
                        'trajectory_type': traj_type,
                        'steps': 100  # フル軌道
                    }
                    configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="軌道パターン実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_initial_offset_experiment(self, num_experiments=5):
        """
        初期位置と角度のオフセットの影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/initial_offset_experiment_{timestamp}.csv"
        
        # 位置オフセットのリスト
        distance_offsets = [0.1, 0.5, 1.0, 2.0, 5.0]
        # 角度オフセットのリスト（ラジアン）
        angle_offsets = [np.radians(5), np.radians(15), np.radians(30), np.radians(45), np.radians(90)]
        
        # ヘッダー行
        header = [
            'experiment_id', 'initial_offset_distance', 'initial_offset_angle_deg', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        
        # 位置オフセット実験
        for offset in distance_offsets:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'pos_offset_{offset}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50,  # ステップ数を短くして実験時間を短縮
                    'initial_offset_distance': offset,
                    'initial_offset_angle': 0.0
                }
                configs.append(config)
        
        # 角度オフセット実験
        for offset in angle_offsets:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'angle_offset_{np.degrees(offset)}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50,  # ステップ数を短くして実験時間を短縮
                    'initial_offset_distance': 0.0,
                    'initial_offset_angle': offset,
                    'initial_offset_angle_deg': np.degrees(offset)  # 読みやすさのために度数も記録
                }
                configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="初期値オフセット実験"):
            result = self.run_single_experiment(config)
            if 'initial_offset_angle_deg' not in result and 'initial_offset_angle' in result:
                result['initial_offset_angle_deg'] = np.degrees(result['initial_offset_angle'])
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_sampling_rate_experiment(self, num_experiments=5):
        """
        点群サンプリング率の影響を評価する実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            str: 結果ファイルのパス
        """
        # CSVファイルの設定
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_filename = f"{self.results_dir}/sampling_rate_experiment_{timestamp}.csv"
        
        # サンプリング率のリスト
        sampling_rates = [1.0, 0.75, 0.5, 0.25, 0.1]
        
        # ヘッダー行
        header = [
            'experiment_id', 'sampling_rate', 
            'avg_position_error', 'max_position_error', 'final_position_error',
            'avg_angle_error_deg', 'max_angle_error_deg', 'final_angle_error_deg',
            'avg_process_time_ms', 'total_time_s',
            'avg_iterations', 'max_iterations', 'avg_score', 'convergence_rate',
            'timestamp'
        ]
        
        configs = []
        for rate in sampling_rates:
            for exp_id in range(num_experiments):
                # 実験設定
                config = {
                    'experiment_id': f'sampling_{rate}_{exp_id}',
                    'cell_size': 0.5,  # 固定セルサイズ
                    'use_newton': True,  # ニュートン法を使用
                    'damping': 0.1,
                    'num_beams': 360,  # 高分解能スキャン
                    'trajectory_type': 'random',  # ランダム軌道
                    'steps': 50,  # ステップ数を短くして実験時間を短縮
                    'sampling_rate': rate
                }
                configs.append(config)
        
        # 実験を実行
        results = []
        for config in tqdm(configs, desc="サンプリング率実験"):
            result = self.run_single_experiment(config)
            results.append(result)
        
        # 結果をCSVファイルに保存
        self._save_results_to_csv(results, header, csv_filename)
            
        return csv_filename
    
    def run_all_experiments(self, num_experiments=3):
        """
        すべての実験を実行
        
        Args:
            num_experiments (int): 各パラメータ設定で実行する実験の数
            
        Returns:
            list: 結果ファイルのパスのリスト
        """
        results = []
        
        # 各実験を実行
        results.append(self.run_cell_size_experiment(num_experiments))
        results.append(self.run_optimization_method_experiment(num_experiments))
        results.append(self.run_damping_experiment(num_experiments))
        results.append(self.run_noise_experiment(num_experiments))
        results.append(self.run_beam_count_experiment(num_experiments))
        results.append(self.run_trajectory_experiment(num_experiments))
        results.append(self.run_initial_offset_experiment(num_experiments))
        results.append(self.run_sampling_rate_experiment(num_experiments))
        
        return results
    
    def _save_results_to_csv(self, results, header, csv_filename):
        """
        結果をCSVファイルに保存
        
        Args:
            results (list): 実験結果のリスト
            header (list): CSVヘッダー行
            csv_filename (str): 保存先ファイルパス
        """
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            writer.writeheader()
            for result in results:
                # ヘッダーにない項目を除去
                filtered_result = {key: result[key] for key in header if key in result}
                writer.writerow(filtered_result)
        
        print(f"結果を保存しました: {csv_filename}")

def run_experiment_worker(args):
    """
    並列処理のためのワーカー関数
    
    Args:
        args (tuple): (実験クラスインスタンス, 実験関数名, 実験回数)
        
    Returns:
        str: 結果ファイルのパス
    """
    runner, func_name, num_experiments = args
    func = getattr(runner, func_name)
    return func(num_experiments)

def main():
    """
    メイン関数
    """
    try:
        parser = argparse.ArgumentParser(description='NDT位置推定の性能評価と実験データ収集')
        parser.add_argument('--map_dir', type=str, default='map',
                          help='マップファイル(.pgmと.yaml)が格納されているディレクトリ')
        parser.add_argument('--experiments', type=str, nargs='+',
                          choices=['all', 'cell_size', 'optimization', 'damping', 
                                   'noise', 'beam_count', 'trajectory', 
                                   'initial_offset', 'sampling_rate'],
                          default=['all'],
                          help='実行する実験タイプ')
        parser.add_argument('--num_experiments', type=int, default=3,
                          help='各パラメータ設定で実行する実験の数')
        parser.add_argument('--parallel', action='store_true',
                          help='並列処理を有効にする')
        
        args = parser.parse_args()
        
        print(f"引数: map_dir={args.map_dir}, experiments={args.experiments}, num_experiments={args.num_experiments}")
        
        # 地図ファイルのパスを構築
        map_dir = Path(args.map_dir)
        print(f"マップディレクトリのパス: {map_dir} (絶対パス: {map_dir.absolute()})")
        
        pgm_files = list(map_dir.glob('*.pgm'))
        if not pgm_files:
            print(f"エラー: {map_dir}に.pgmファイルが見つかりません")
            return
        
        print(f"見つかったPGMファイル: {pgm_files}")
        pgm_path = pgm_files[0]
        yaml_path = pgm_path.with_suffix('.yaml')
        
        if not yaml_path.exists():
            print(f"エラー: YAMLファイルが見つかりません: {yaml_path}")
            return
            
        print(f"使用するマップファイル: PGM={pgm_path}, YAML={yaml_path}")
    except Exception as e:
        print(f"初期化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 実験ランナーのインスタンスを作成
    runner = ExperimentRunner(str(pgm_path), str(yaml_path))
    
    experiment_funcs = []
    
    # 実行する実験を選択
    if 'all' in args.experiments or 'cell_size' in args.experiments:
        experiment_funcs.append('run_cell_size_experiment')
    
    if 'all' in args.experiments or 'optimization' in args.experiments:
        experiment_funcs.append('run_optimization_method_experiment')
    
    if 'all' in args.experiments or 'damping' in args.experiments:
        experiment_funcs.append('run_damping_experiment')
    
    if 'all' in args.experiments or 'noise' in args.experiments:
        experiment_funcs.append('run_noise_experiment')
    
    if 'all' in args.experiments or 'beam_count' in args.experiments:
        experiment_funcs.append('run_beam_count_experiment')
    
    if 'all' in args.experiments or 'trajectory' in args.experiments:
        experiment_funcs.append('run_trajectory_experiment')
    
    if 'all' in args.experiments or 'initial_offset' in args.experiments:
        experiment_funcs.append('run_initial_offset_experiment')
    
    if 'all' in args.experiments or 'sampling_rate' in args.experiments:
        experiment_funcs.append('run_sampling_rate_experiment')
    
    if args.parallel and len(experiment_funcs) > 1:
        # 並列処理
        print(f"{len(experiment_funcs)}個の実験を並列実行します...")
        with mp.Pool() as pool:
            work_args = [(runner, func, args.num_experiments) for func in experiment_funcs]
            results = pool.map(run_experiment_worker, work_args)
        print("すべての実験が完了しました")
        for result in results:
            print(f"結果: {result}")
    else:
        # 逐次処理
        results = []
        for func_name in experiment_funcs:
            print(f"実験を実行中: {func_name}")
            func = getattr(runner, func_name)
            result = func(args.num_experiments)
            results.append(result)
        
        print("すべての実験が完了しました")
        for result in results:
            print(f"結果: {result}")

if __name__ == "__main__":
    main()
