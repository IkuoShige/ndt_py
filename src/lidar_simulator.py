import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import rerun as rr
import time

class LidarSimulator:
    """
    2D LiDARシミュレータークラス
    占有格子地図上で、ロボットの位置から仮想的なLiDARスキャンを生成する
    """
    def __init__(self, occupancy_map, num_beams=360, max_range=10.0, angle_min=-np.pi, angle_max=np.pi,
                 noise_sigma=0.01):
        """
        初期化関数
        
        Args:
            occupancy_map: 占有格子地図オブジェクト
            num_beams (int): LiDARビームの数
            max_range (float): 最大検出範囲 [m]
            angle_min (float): 最小角度 [rad]
            angle_max (float): 最大角度 [rad]
            noise_sigma (float): ガウシアンノイズの標準偏差 [m]
        """
        self.occupancy_map = occupancy_map
        self.num_beams = num_beams
        self.max_range = max_range
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.noise_sigma = noise_sigma
        
        # 角度のリストを計算（ビームごと）
        self.angles = np.linspace(angle_min, angle_max, num_beams)
    
    def simulate_scan(self, robot_pose):
        """
        指定されたロボットの姿勢からLiDARスキャンをシミュレート
        
        Args:
            robot_pose (numpy.ndarray): ロボットの姿勢 [x, y, theta]
        
        Returns:
            tuple: (ranges, points)
                ranges: 距離のリスト
                points: 点群のリスト [[x1, y1], [x2, y2], ...]
        """
        robot_x, robot_y, robot_theta = robot_pose
        ranges = np.zeros(self.num_beams)
        points = []
        
        # 各ビームに対してレイキャスティング
        for i, angle in enumerate(self.angles):
            # ロボットの向きを考慮した絶対角度
            absolute_angle = robot_theta + angle
            
            # このビームの最大到達点
            end_x = robot_x + self.max_range * np.cos(absolute_angle)
            end_y = robot_y + self.max_range * np.sin(absolute_angle)
            
            # レイキャスティングで最も近い障害物を検出
            range_value = self._ray_casting(robot_x, robot_y, end_x, end_y)
            
            # ノイズを追加
            if range_value < self.max_range:
                range_value += np.random.normal(0, self.noise_sigma)
                range_value = max(0, range_value)  # 負の距離を防ぐ
            
            ranges[i] = range_value
            
            # 点群データを作成
            if range_value < self.max_range:
                point_x = robot_x + range_value * np.cos(absolute_angle)
                point_y = robot_y + range_value * np.sin(absolute_angle)
                points.append([point_x, point_y])
        
        return ranges, np.array(points) if points else np.empty((0, 2))
    
    def _ray_casting(self, start_x, start_y, end_x, end_y, step_size=0.1):
        """
        レイキャスティングを実行して障害物までの距離を計算
        
        Args:
            start_x, start_y (float): 開始点の座標
            end_x, end_y (float): 終点の座標
            step_size (float): レイキャストのステップサイズ
        
        Returns:
            float: 障害物までの距離。障害物がない場合は最大距離を返す。
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
                return current_distance
        
        # 障害物が見つからなかった場合は最大距離を返す
        return self.max_range
    
    def visualize_scan(self, robot_pose, ranges):
        """
        ロボットの位置とLiDARスキャンを可視化
        
        Args:
            robot_pose (numpy.ndarray): ロボットの姿勢 [x, y, theta]
            ranges (numpy.ndarray): レーザースキャンの距離データ
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        # 地図の二値化データを取得
        binary_map = self.occupancy_map.get_occupancy_grid()
        
        # 地図を描画
        plt.figure(figsize=(10, 10))
        
        # 地図の左下の座標を取得
        origin_x, origin_y = self.occupancy_map.origin[0], self.occupancy_map.origin[1]
        
        # 地図の表示範囲を設定
        extent = [origin_x, 
                 origin_x + binary_map.shape[1] * self.occupancy_map.resolution, 
                 origin_y, 
                 origin_y + binary_map.shape[0] * self.occupancy_map.resolution]
        
        # 地図の描画
        plt.imshow(binary_map, cmap='gray', origin='lower', extent=extent)
        
        # ロボットを描画
        robot_radius = 0.3  # ロボットの半径（表示用）
        robot_circle = plt.Circle((robot_x, robot_y), robot_radius, color='r')
        plt.gca().add_patch(robot_circle)
        
        # ロボットの向きを示す線
        direction_length = robot_radius * 2
        plt.arrow(robot_x, robot_y, 
                 direction_length * np.cos(robot_theta), 
                 direction_length * np.sin(robot_theta), 
                 head_width=0.2, head_length=0.3, fc='r', ec='r')
        
        # スキャンデータを描画
        for i, r in enumerate(ranges):
            angle = self.angles[i] + robot_theta
            if r < self.max_range:
                end_x = robot_x + r * np.cos(angle)
                end_y = robot_y + r * np.sin(angle)
                plt.plot([robot_x, end_x], [robot_y, end_y], 'g-', alpha=0.2)
                plt.plot(end_x, end_y, 'go', markersize=2)
        
        plt.title('LiDAR Simulation')
        plt.axis('equal')
        plt.grid(True)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.tight_layout()
        plt.show()
    
    def visualize_with_rerun(self, robot_pose, scan_points, target_points=None, transformed_points=None):
        """
        Rerunを使用してロボットの位置とLiDARスキャンを可視化
        
        Args:
            robot_pose (numpy.ndarray): ロボットの姿勢 [x, y, theta]
            scan_points (numpy.ndarray): LiDARスキャンの点群データ
            target_points (numpy.ndarray, optional): ターゲット点群（参照地図）
            transformed_points (numpy.ndarray, optional): 変換後の点群
            frame_index (int, optional): フレームインデックス（タイムライン用）
        """
        robot_x, robot_y, robot_theta = robot_pose
        
        # 地図を描画
        binary_map = self.occupancy_map.get_occupancy_grid()
        
        # 地図の左下の座標を計算
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
        
        # タイムライン設定
        
        # スキャン点群を描画
        if len(scan_points) > 0:
            rr.log("world/scan_points", 
                  rr.Points2D(
                      positions=scan_points,
                      colors=np.full((len(scan_points), 3), [0.0, 1.0, 0.0]),  # 緑色
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
        
        # 現在の位置を可視化（座標軸として）
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
        
        # ロボットの向きを描画
        direction_length = 0.6
        rr.log("world/robot_direction",
              rr.LineStrips2D(
                  strips=[np.array([[robot_x, robot_y], 
                                  [robot_x + direction_length * np.cos(robot_theta), 
                                   robot_y + direction_length * np.sin(robot_theta)]])],
                  colors=np.array([[1.0, 0.0, 0.0]]),  # 赤色
                  radii=0.1
              ))


