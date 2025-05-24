import numpy as np
import matplotlib.pyplot as plt
import rerun as rr
import time
import random

class RealisticLidarSimulator:
    """
    実際のLiDARセンサーに近い特性を持つ2D LiDARシミュレータークラス
    実際のロボットへの適応を想定した、より現実的なノイズモデルとセンサー特性を実装
    """
    def __init__(self, occupancy_map, num_beams=360, max_range=10.0, min_range=0.1,
                 angle_min=-np.pi, angle_max=np.pi, angle_resolution=None,
                 base_noise_sigma=0.01, distance_noise_factor=0.001, 
                 beam_divergence=0.003, reflection_prob=0.98, multipath_prob=0.02,
                 max_intensity=1000.0):
        """
        初期化関数
        
        Args:
            occupancy_map: 占有格子地図オブジェクト
            num_beams (int): LiDARビームの数
            max_range (float): 最大検出範囲 [m]
            min_range (float): 最小検出範囲 [m] (実際のセンサーは近距離で検出できない)
            angle_min (float): 最小角度 [rad]
            angle_max (float): 最大角度 [rad]
            angle_resolution (float): 角度解像度 [rad] (Noneの場合は自動計算)
            base_noise_sigma (float): 基本ガウシアンノイズの標準偏差 [m]
            distance_noise_factor (float): 距離に比例するノイズ係数
            beam_divergence (float): ビーム発散角 [rad]
            reflection_prob (float): 反射成功確率
            multipath_prob (float): マルチパス反射の発生確率
            max_intensity (float): 最大反射強度
        """
        self.occupancy_map = occupancy_map
        self.num_beams = num_beams
        self.max_range = max_range
        self.min_range = min_range
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.base_noise_sigma = base_noise_sigma
        self.distance_noise_factor = distance_noise_factor
        self.beam_divergence = beam_divergence
        self.reflection_prob = reflection_prob
        self.multipath_prob = multipath_prob
        self.max_intensity = max_intensity
        
        # 角度解像度の設定
        if angle_resolution is not None:
            self.angle_resolution = angle_resolution
            # 角度解像度に基づいてビーム数を再計算
            angle_range = angle_max - angle_min
            self.num_beams = int(angle_range / angle_resolution) + 1
        else:
            self.angle_resolution = (angle_max - angle_min) / (num_beams - 1)
        
        # 角度のリストを計算（ビームごと）
        self.angles = np.linspace(angle_min, angle_max, self.num_beams)
        
        # センサー特性パラメータ
        self.scan_frequency = 10.0  # Hz
        self.last_scan_time = 0.0
        
        # 統計データ（デバッグ用）
        self.stats = {
            'total_beams': 0,
            'valid_detections': 0,
            'failed_reflections': 0,
            'multipath_detections': 0,
            'out_of_range': 0
        }
    
    def simulate_scan(self, robot_pose, timestamp=None):
        """
        指定されたロボットの姿勢から現実的なLiDARスキャンをシミュレート
        
        Args:
            robot_pose (numpy.ndarray): ロボットの姿勢 [x, y, theta]
            timestamp (float): タイムスタンプ（リアルタイム処理用）
        
        Returns:
            dict: スキャンデータ
                ranges: 距離のリスト
                intensities: 反射強度のリスト
                points: 点群のリスト [[x1, y1], [x2, y2], ...]
                valid: 有効な測定値のフラグリスト
                timestamp: タイムスタンプ
        """
        robot_x, robot_y, robot_theta = robot_pose
        ranges = np.full(self.num_beams, np.inf)  # 初期値は無限大
        intensities = np.zeros(self.num_beams)
        points = []
        valid_flags = np.zeros(self.num_beams, dtype=bool)
        
        if timestamp is None:
            timestamp = time.time()
        
        self.last_scan_time = timestamp
        
        # 各ビームに対してレイキャスティング
        for i, angle in enumerate(self.angles):
            self.stats['total_beams'] += 1
            
            # ロボットの向きを考慮した絶対角度
            absolute_angle = robot_theta + angle
            
            # ビーム発散を考慮（わずかな角度ばらつき）
            if self.beam_divergence > 0:
                angle_noise = np.random.normal(0, self.beam_divergence)
                absolute_angle += angle_noise
            
            # このビームの最大到達点
            end_x = robot_x + self.max_range * np.cos(absolute_angle)
            end_y = robot_y + self.max_range * np.sin(absolute_angle)
            
            # レイキャスティングで最も近い障害物を検出
            detection_result = self._realistic_ray_casting(robot_x, robot_y, end_x, end_y)
            
            if detection_result is not None:
                range_value, intensity, is_multipath = detection_result
                
                # 最小距離チェック
                if range_value < self.min_range:
                    self.stats['out_of_range'] += 1
                    continue
                
                # 反射確率チェック
                if np.random.random() > self.reflection_prob:
                    self.stats['failed_reflections'] += 1
                    continue
                
                # 距離依存ノイズを追加
                noise_sigma = self.base_noise_sigma + self.distance_noise_factor * range_value
                range_value += np.random.normal(0, noise_sigma)
                range_value = max(self.min_range, range_value)  # 最小距離以下にならないよう制限
                
                # マルチパス反射の処理
                if is_multipath:
                    self.stats['multipath_detections'] += 1
                    # マルチパス反射では距離が長くなる傾向
                    range_value *= np.random.uniform(1.1, 1.5)
                    intensity *= 0.3  # 反射強度が低下
                
                ranges[i] = range_value
                intensities[i] = intensity
                valid_flags[i] = True
                self.stats['valid_detections'] += 1
                
                # 点群データを作成
                if range_value < self.max_range:
                    point_x = robot_x + range_value * np.cos(absolute_angle)
                    point_y = robot_y + range_value * np.sin(absolute_angle)
                    points.append([point_x, point_y])
        
        return {
            'ranges': ranges,
            'intensities': intensities,
            'points': np.array(points) if points else np.empty((0, 2)),
            'valid': valid_flags,
            'timestamp': timestamp,
            'angle_min': self.angle_min,
            'angle_max': self.angle_max,
            'angle_increment': self.angle_resolution,
            'range_min': self.min_range,
            'range_max': self.max_range
        }
    
    def _realistic_ray_casting(self, start_x, start_y, end_x, end_y, step_size=0.05):
        """
        現実的なレイキャスティングを実行して障害物までの距離を計算
        
        Args:
            start_x, start_y (float): 開始点の座標
            end_x, end_y (float): 終点の座標
            step_size (float): レイキャストのステップサイズ
        
        Returns:
            tuple or None: (distance, intensity, is_multipath) or None
        """
        # ベクトルの方向を計算
        dx = end_x - start_x
        dy = end_y - start_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # 単位ベクトル
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # レイキャスティングのステップ数
        num_steps = int(distance / step_size)
        
        # 各ステップで占有状態をチェック
        for step in range(1, num_steps + 1):
            current_distance = step * step_size
            x = start_x + dx * current_distance
            y = start_y + dy * current_distance
            
            # この点が占有されているかをチェック
            if self.occupancy_map.is_occupied(x, y):
                # 反射強度を計算（距離と角度に依存）
                intensity = self._calculate_intensity(current_distance, dx, dy)
                
                # マルチパス反射の判定
                is_multipath = np.random.random() < self.multipath_prob
                
                return current_distance, intensity, is_multipath
        
        # 障害物が見つからなかった場合
        return None
    
    def _calculate_intensity(self, distance, dx, dy):
        """
        反射強度を計算
        
        Args:
            distance (float): 距離
            dx, dy (float): ビーム方向の単位ベクトル
        
        Returns:
            float: 反射強度
        """
        # 距離の逆二乗に比例して強度が減衰
        distance_factor = 1.0 / (distance**2 + 0.1)
        
        # 基本強度にランダム性を追加（表面の材質による反射率の違いをシミュレート）
        base_intensity = np.random.uniform(0.3, 1.0)
        
        # 入射角による強度変化（簡易モデル）
        # 実際には表面の法線ベクトルが必要だが、ここでは簡略化
        angle_factor = np.random.uniform(0.7, 1.0)
        
        intensity = self.max_intensity * base_intensity * distance_factor * angle_factor
        return max(0, min(intensity, self.max_intensity))
    
    def get_scan_rate(self):
        """スキャンレートを取得"""
        return self.scan_frequency
    
    def get_statistics(self):
        """統計情報を取得"""
        total = max(1, self.stats['total_beams'])
        return {
            'detection_rate': self.stats['valid_detections'] / total,
            'failure_rate': self.stats['failed_reflections'] / total,
            'multipath_rate': self.stats['multipath_detections'] / total,
            'out_of_range_rate': self.stats['out_of_range'] / total,
            **self.stats
        }
    
    def reset_statistics(self):
        """統計情報をリセット"""
        for key in self.stats:
            self.stats[key] = 0
    
    def visualize_scan_with_rerun(self, robot_pose, scan_data, target_points=None, transformed_points=None):
        """
        Rerunを使用してロボットの位置とLiDARスキャンを可視化（現実的なデータに対応）
        
        Args:
            robot_pose (numpy.ndarray): ロボットの姿勢 [x, y, theta]
            scan_data (dict): スキャンデータ
            target_points (numpy.ndarray, optional): ターゲット点群（参照地図）
            transformed_points (numpy.ndarray, optional): 変換後の点群
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        # 地図を描画
        binary_map = self.occupancy_map.get_occupancy_grid()
        origin_x, origin_y = self.occupancy_map.origin[0], self.occupancy_map.origin[1]
        
        # 地図データをRerunに送信
        rr.log("world/map", 
              rr.Points2D(
                  positions=np.array([(i * self.occupancy_map.resolution + origin_x, 
                                     j * self.occupancy_map.resolution + origin_y) 
                                    for j, i in np.ndindex(binary_map.shape) if binary_map[j, i] > 0]),
                  colors=np.zeros((np.sum(binary_map > 0), 3)),  # 黒色
                  radii=0.05
              ))
        
        # スキャン点群を描画（有効な点のみ）
        scan_points = scan_data['points']
        if len(scan_points) > 0:
            # 有効な点に対応する強度データのみを取得
            # 点群データが作成される条件と一致するように修正
            ranges = scan_data['ranges']
            intensities = scan_data['intensities']
            valid_flags = scan_data['valid']
            
            # 有効な点の強度を抽出（点群の順序に対応）
            point_intensities = []
            point_idx = 0
            for i in range(len(ranges)):
                if valid_flags[i] and ranges[i] < self.max_range:
                    if point_idx < len(scan_points):
                        point_intensities.append(intensities[i])
                        point_idx += 1
            
            # 点群の数と強度データの数を一致させる
            point_intensities = np.array(point_intensities[:len(scan_points)])
            if len(point_intensities) < len(scan_points):
                # 不足分は平均強度で補完
                mean_intensity = np.mean(point_intensities) if len(point_intensities) > 0 else self.max_intensity * 0.5
                point_intensities = np.pad(point_intensities, (0, len(scan_points) - len(point_intensities)), 
                                         mode='constant', constant_values=mean_intensity)
            
            normalized_intensities = point_intensities / self.max_intensity
            
            # 強度に基づいて色を設定（低強度：青、高強度：赤）
            colors = np.zeros((len(scan_points), 3))
            colors[:, 0] = normalized_intensities  # 赤成分
            colors[:, 2] = 1.0 - normalized_intensities  # 青成分
            
            rr.log("world/scan_points", 
                  rr.Points2D(
                      positions=scan_points,
                      colors=colors,
                      radii=0.05
                  ))
        
        # ターゲット点群を描画（存在する場合）
        if target_points is not None and len(target_points) > 0:
            rr.log("world/target_points", 
                  rr.Points2D(
                      positions=target_points,
                      colors=np.full((len(target_points), 3), [0.5, 0.5, 0.5]),  # グレー
                      radii=0.05
                  ))
        
        # 変換後の点群を描画（存在する場合）
        if transformed_points is not None and len(transformed_points) > 0:
            rr.log("world/transformed_points", 
                  rr.Points2D(
                      positions=transformed_points,
                      colors=np.full((len(transformed_points), 3), [1.0, 0.5, 0.0]),  # オレンジ色
                      radii=0.05
                  ))
        
        # ロボットの位置と向きを描画
        c, s = np.cos(robot_theta), np.sin(robot_theta)
        x_axis = np.array([[robot_x, robot_y], 
                          [robot_x + 0.5 * c, robot_y + 0.5 * s]])
        y_axis = np.array([[robot_x, robot_y], 
                          [robot_x - 0.5 * s, robot_y + 0.5 * c]])
        
        rr.log("world/robot_pose/x_axis", 
              rr.LineStrips2D(
                  strips=[x_axis],
                  colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                  radii=0.1
              ))
        rr.log("world/robot_pose/y_axis", 
              rr.LineStrips2D(
                  strips=[y_axis],
                  colors=np.array([[0.0, 1.0, 0.0]]),  # 緑色
                  radii=0.1
              ))
        
        # ロボットの位置を描画
        rr.log("world/robot", 
              rr.Points2D(
                  positions=np.array([[robot_x, robot_y]]),
                  colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                  radii=0.2
              ))
        
        # センサー統計を表示
        stats = self.get_statistics()
        rr.log("sensors/lidar_stats", 
              rr.TextLog(f"検出率: {stats['detection_rate']:.2f}, "
                        f"反射失敗率: {stats['failure_rate']:.2f}, "
                        f"マルチパス率: {stats['multipath_rate']:.2f}")) 