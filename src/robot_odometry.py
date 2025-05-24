import numpy as np
import time

class RobotOdometry:
    """
    現実的なロボットオドメトリーをシミュレートするクラス
    車輪滑り、エンコーダー誤差、キャリブレーション誤差などを考慮
    """
    def __init__(self, wheel_base=0.5, wheel_radius=0.1, encoder_resolution=1024,
                 systematic_error_factor=0.02, random_error_factor=0.01,
                 slip_factor=0.05, drift_rate=0.001):
        """
        初期化関数
        
        Args:
            wheel_base (float): 車輪間距離 [m]
            wheel_radius (float): 車輪半径 [m] 
            encoder_resolution (int): エンコーダー解像度 [pulse/rev]
            systematic_error_factor (float): 系統的誤差係数
            random_error_factor (float): ランダム誤差係数
            slip_factor (float): 車輪滑り係数
            drift_rate (float): ドリフト率 [rad/s]
        """
        self.wheel_base = wheel_base
        self.wheel_radius = wheel_radius
        self.encoder_resolution = encoder_resolution
        self.systematic_error_factor = systematic_error_factor
        self.random_error_factor = random_error_factor
        self.slip_factor = slip_factor
        self.drift_rate = drift_rate
        
        # オドメトリー状態
        self.pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.last_update_time = time.time()
        
        # エラー累積
        self.cumulative_distance = 0.0
        self.cumulative_rotation = 0.0
        
        # キャリブレーション誤差（実際のロボットではセンサー取り付け位置の誤差）
        self.calibration_error = {
            'wheel_base_error': np.random.normal(0, 0.01),  # 車輪間距離の誤差
            'wheel_radius_error': np.random.normal(0, 0.005),  # 車輪半径の誤差
            'heading_bias': np.random.normal(0, 0.02)  # 方位のバイアス
        }
    
    def update_from_wheel_velocities(self, left_vel, right_vel, dt):
        """
        車輪速度からオドメトリーを更新（差動駆動ロボット用）
        
        Args:
            left_vel (float): 左車輪の速度 [m/s]
            right_vel (float): 右車輪の速度 [m/s]
            dt (float): 時間間隔 [s]
        
        Returns:
            tuple: (移動量 [dx, dy, dtheta], 不確実性)
        """
        # キャリブレーション誤差を適用
        effective_wheel_base = self.wheel_base + self.calibration_error['wheel_base_error']
        
        # 車輪速度にノイズを追加（エンコーダー誤差）
        left_vel_noisy = self._add_encoder_noise(left_vel)
        right_vel_noisy = self._add_encoder_noise(right_vel)
        
        # 滑りの影響を追加
        left_vel_actual = left_vel_noisy * (1.0 - np.random.uniform(0, self.slip_factor))
        right_vel_actual = right_vel_noisy * (1.0 - np.random.uniform(0, self.slip_factor))
        
        # 運動学計算
        linear_vel = (left_vel_actual + right_vel_actual) / 2.0
        angular_vel = (right_vel_actual - left_vel_actual) / effective_wheel_base
        
        # ドリフトの影響（ジャイロドリフトなど）
        angular_vel += np.random.normal(0, self.drift_rate)
        
        # 移動量の計算
        if abs(angular_vel) < 1e-6:
            # 直線運動
            dx_local = linear_vel * dt
            dy_local = 0.0
            dtheta = 0.0
        else:
            # 曲線運動
            radius = linear_vel / angular_vel
            dtheta = angular_vel * dt
            dx_local = radius * np.sin(dtheta)
            dy_local = radius * (1.0 - np.cos(dtheta))
        
        # 現在の姿勢を考慮してグローバル座標に変換
        current_theta = self.pose[2]
        cos_theta = np.cos(current_theta)
        sin_theta = np.sin(current_theta)
        
        dx_global = cos_theta * dx_local - sin_theta * dy_local
        dy_global = sin_theta * dx_local + cos_theta * dy_local
        
        # 系統的誤差と累積誤差を追加
        distance_traveled = np.sqrt(dx_global**2 + dy_global**2)
        self.cumulative_distance += distance_traveled
        self.cumulative_rotation += abs(dtheta)
        
        # 距離に比例する誤差
        distance_error_factor = 1.0 + self.systematic_error_factor * self.cumulative_distance
        dx_global *= distance_error_factor
        dy_global *= distance_error_factor
        
        # 回転に比例する角度誤差
        angle_error = self.systematic_error_factor * self.cumulative_rotation
        dtheta += angle_error + self.calibration_error['heading_bias']
        
        # ランダム誤差を追加
        dx_global += np.random.normal(0, self.random_error_factor * max(0.1, distance_traveled))
        dy_global += np.random.normal(0, self.random_error_factor * max(0.1, distance_traveled))
        dtheta += np.random.normal(0, self.random_error_factor * max(0.01, abs(dtheta)))
        
        # 姿勢を更新
        self.pose[0] += dx_global
        self.pose[1] += dy_global
        self.pose[2] += dtheta
        
        # 角度を正規化
        self.pose[2] = np.arctan2(np.sin(self.pose[2]), np.cos(self.pose[2]))
        
        # 不確実性（共分散行列）を計算
        uncertainty = self._calculate_uncertainty(distance_traveled, abs(dtheta), dt)
        
        return np.array([dx_global, dy_global, dtheta]), uncertainty
    
    def update_from_motion_command(self, linear_vel, angular_vel, dt):
        """
        運動コマンドからオドメトリーを更新
        
        Args:
            linear_vel (float): 線速度 [m/s]
            angular_vel (float): 角速度 [rad/s]
            dt (float): 時間間隔 [s]
        
        Returns:
            tuple: (移動量 [dx, dy, dtheta], 不確実性)
        """
        # 速度コマンドから車輪速度を計算
        left_vel = linear_vel - angular_vel * self.wheel_base / 2.0
        right_vel = linear_vel + angular_vel * self.wheel_base / 2.0
        
        return self.update_from_wheel_velocities(left_vel, right_vel, dt)
    
    def _add_encoder_noise(self, velocity):
        """
        エンコーダーノイズを追加
        
        Args:
            velocity (float): 理想的な速度
        
        Returns:
            float: ノイズが追加された速度
        """
        # エンコーダー解像度による量子化誤差
        wheel_circumference = 2 * np.pi * self.wheel_radius
        distance_per_pulse = wheel_circumference / self.encoder_resolution
        
        # 速度の量子化
        pulses_per_second = velocity / distance_per_pulse
        quantized_pulses = round(pulses_per_second)
        quantized_velocity = quantized_pulses * distance_per_pulse
        
        # エンコーダーの読み取り誤差
        encoder_error = np.random.normal(0, distance_per_pulse * 0.1)
        
        return quantized_velocity + encoder_error
    
    def _calculate_uncertainty(self, distance, rotation, dt):
        """
        オドメトリーの不確実性（共分散行列）を計算
        
        Args:
            distance (float): 移動距離
            rotation (float): 回転角度
            dt (float): 時間間隔
        
        Returns:
            numpy.ndarray: 3x3の共分散行列
        """
        # 基本の不確実性
        base_pos_var = (self.random_error_factor * max(0.01, distance))**2
        base_ang_var = (self.random_error_factor * max(0.001, rotation))**2
        
        # 累積誤差による不確実性の増加
        cumulative_factor = 1.0 + 0.1 * self.cumulative_distance
        
        # 3x3共分散行列
        covariance = np.zeros((3, 3))
        covariance[0, 0] = base_pos_var * cumulative_factor  # x位置の分散
        covariance[1, 1] = base_pos_var * cumulative_factor  # y位置の分散
        covariance[2, 2] = base_ang_var * cumulative_factor  # 角度の分散
        
        # 相関項（位置と角度の相関）
        pos_ang_correlation = 0.1 * np.sqrt(base_pos_var * base_ang_var)
        covariance[0, 2] = covariance[2, 0] = pos_ang_correlation
        covariance[1, 2] = covariance[2, 1] = pos_ang_correlation
        
        return covariance
    
    def get_pose(self):
        """現在の姿勢を取得"""
        return self.pose.copy()
    
    def set_pose(self, pose):
        """姿勢を設定（初期化時やリセット時に使用）"""
        self.pose = np.array(pose)
    
    def reset_errors(self):
        """累積誤差をリセット"""
        self.cumulative_distance = 0.0
        self.cumulative_rotation = 0.0
    
    def get_error_statistics(self):
        """エラー統計を取得"""
        return {
            'cumulative_distance': self.cumulative_distance,
            'cumulative_rotation': self.cumulative_rotation,
            'position_error_estimate': self.systematic_error_factor * self.cumulative_distance,
            'angle_error_estimate': self.systematic_error_factor * self.cumulative_rotation
        }


class OdometryIntegrator:
    """
    オドメトリーデータを統合し、真の軌道との比較を行うクラス
    """
    def __init__(self, initial_pose=None):
        """
        初期化関数
        
        Args:
            initial_pose (numpy.ndarray): 初期姿勢 [x, y, theta]
        """
        if initial_pose is None:
            initial_pose = np.array([0.0, 0.0, 0.0])
        
        self.true_pose = initial_pose.copy()
        self.odometry_pose = initial_pose.copy()
        self.odometry = RobotOdometry()
        self.odometry.set_pose(initial_pose)
        
        # 軌道履歴
        self.true_trajectory = [initial_pose.copy()]
        self.odometry_trajectory = [initial_pose.copy()]
        
        # エラー履歴
        self.position_errors = [0.0]
        self.angle_errors = [0.0]
    
    def update(self, true_motion, dt):
        """
        真の運動とオドメトリーを更新
        
        Args:
            true_motion (numpy.ndarray): 真の運動 [linear_vel, angular_vel]
            dt (float): 時間間隔
        """
        linear_vel, angular_vel = true_motion
        
        # 真の姿勢を更新（理想的な運動学モデル）
        if abs(angular_vel) < 1e-6:
            # 直線運動
            dx = linear_vel * dt * np.cos(self.true_pose[2])
            dy = linear_vel * dt * np.sin(self.true_pose[2])
            dtheta = 0.0
        else:
            # 曲線運動
            radius = linear_vel / angular_vel
            dtheta = angular_vel * dt
            dx = radius * (np.sin(self.true_pose[2] + dtheta) - np.sin(self.true_pose[2]))
            dy = radius * (-np.cos(self.true_pose[2] + dtheta) + np.cos(self.true_pose[2]))
        
        self.true_pose[0] += dx
        self.true_pose[1] += dy
        self.true_pose[2] += dtheta
        self.true_pose[2] = np.arctan2(np.sin(self.true_pose[2]), np.cos(self.true_pose[2]))
        
        # オドメトリーを更新
        delta_odom, uncertainty = self.odometry.update_from_motion_command(linear_vel, angular_vel, dt)
        
        # 軌道履歴を更新
        self.true_trajectory.append(self.true_pose.copy())
        self.odometry_trajectory.append(self.odometry.get_pose())
        
        # エラーを計算
        pos_error = np.linalg.norm(self.true_pose[:2] - self.odometry.get_pose()[:2])
        angle_diff = abs(self.true_pose[2] - self.odometry.get_pose()[2])
        angle_error = min(angle_diff, 2*np.pi - angle_diff)
        
        self.position_errors.append(pos_error)
        self.angle_errors.append(angle_error)
        
        return delta_odom, uncertainty
    
    def get_trajectories(self):
        """軌道データを取得"""
        return {
            'true': np.array(self.true_trajectory),
            'odometry': np.array(self.odometry_trajectory)
        }
    
    def get_errors(self):
        """エラーデータを取得"""
        return {
            'position_errors': np.array(self.position_errors),
            'angle_errors': np.array(self.angle_errors)
        }
    
    def get_final_error(self):
        """最終的なエラーを取得"""
        if len(self.position_errors) > 0:
            return {
                'final_position_error': self.position_errors[-1],
                'final_angle_error': self.angle_errors[-1],
                'mean_position_error': np.mean(self.position_errors),
                'mean_angle_error': np.mean(self.angle_errors)
            }
        return None 