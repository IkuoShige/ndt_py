"""
ROS 2 Humble対応の互換性レイヤー
実際のROS 2メッセージ型とユーティリティを提供します
"""

import numpy as np
import time
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass, field

# ROS 2のメッセージ型をインポート（利用可能な場合）
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, QoSPresetProfiles
    from rclpy.time import Time as RclTime, Duration as RclDuration
    from rclpy.clock import Clock
    
    # 標準メッセージ型
    from std_msgs.msg import Header, String
    from geometry_msgs.msg import Point, Quaternion, Pose, PoseWithCovariance, Twist, TwistWithCovariance, Transform, TransformStamped
    from sensor_msgs.msg import LaserScan, PointCloud2, PointField
    from nav_msgs.msg import Odometry, OccupancyGrid, MapMetaData
    from tf2_msgs.msg import TFMessage
    
    ROS2_AVAILABLE = True
    print("ROS 2 Humbleメッセージライブラリが利用可能です")
    
except ImportError as e:
    print(f"ROS 2ライブラリが見つかりません: {e}")
    print("フォールバック：カスタムデータクラスを使用します")
    ROS2_AVAILABLE = False
    
    # フォールバック用のカスタムデータクラス
    @dataclass
    class Header:
        """ROS std_msgs/Header に相当するデータ構造"""
        seq: int = 0
        stamp: float = field(default_factory=time.time)
        frame_id: str = ""

    @dataclass 
    class Point:
        """ROS geometry_msgs/Point に相当するデータ構造"""
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0

    @dataclass
    class Quaternion:
        """ROS geometry_msgs/Quaternion に相当するデータ構造"""
        x: float = 0.0
        y: float = 0.0
        z: float = 0.0
        w: float = 1.0
        
        @classmethod
        def from_euler(cls, roll: float, pitch: float, yaw: float):
            """オイラー角からクォータニオンを生成"""
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            cp = np.cos(pitch * 0.5)
            sp = np.sin(pitch * 0.5)
            cr = np.cos(roll * 0.5)
            sr = np.sin(roll * 0.5)
            
            return cls(
                x=sr * cp * cy - cr * sp * sy,
                y=cr * sp * cy + sr * cp * sy,
                z=cr * cp * sy - sr * sp * cy,
                w=cr * cp * cy + sr * sp * sy
            )
        
        def to_euler(self) -> Tuple[float, float, float]:
            """クォータニオンからオイラー角に変換"""
            # Roll (x-axis rotation)
            sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
            cosr_cosp = 1 - 2 * (self.x * self.x + self.y * self.y)
            roll = np.arctan2(sinr_cosp, cosr_cosp)
            
            # Pitch (y-axis rotation)
            sinp = 2 * (self.w * self.y - self.z * self.x)
            if abs(sinp) >= 1:
                pitch = np.copysign(np.pi / 2, sinp)
            else:
                pitch = np.arcsin(sinp)
            
            # Yaw (z-axis rotation)
            siny_cosp = 2 * (self.w * self.z + self.x * self.y)
            cosy_cosp = 1 - 2 * (self.y * self.y + self.z * self.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            return roll, pitch, yaw

    @dataclass
    class Pose:
        """ROS geometry_msgs/Pose に相当するデータ構造"""
        position: Point = field(default_factory=Point)
        orientation: Quaternion = field(default_factory=Quaternion)
        
        @classmethod
        def from_2d(cls, x: float, y: float, theta: float):
            """2D姿勢から3D Poseを生成"""
            return cls(
                position=Point(x=x, y=y, z=0.0),
                orientation=Quaternion.from_euler(0.0, 0.0, theta)
            )
        
        def to_2d(self) -> Tuple[float, float, float]:
            """3D Poseから2D姿勢に変換"""
            _, _, yaw = self.orientation.to_euler()
            return self.position.x, self.position.y, yaw

    @dataclass
    class PoseWithCovariance:
        """ROS geometry_msgs/PoseWithCovariance に相当するデータ構造"""
        pose: Pose = field(default_factory=Pose)
        covariance: List[float] = field(default_factory=lambda: [0.0] * 36)

    @dataclass
    class Twist:
        """ROS geometry_msgs/Twist に相当するデータ構造"""
        linear: Point = field(default_factory=Point)
        angular: Point = field(default_factory=Point)
        
        @classmethod
        def from_2d(cls, linear_x: float, angular_z: float):
            """2D速度から3D Twistを生成"""
            return cls(
                linear=Point(x=linear_x, y=0.0, z=0.0),
                angular=Point(x=0.0, y=0.0, z=angular_z)
            )

    @dataclass
    class TwistWithCovariance:
        """ROS geometry_msgs/TwistWithCovariance に相当するデータ構造"""
        twist: Twist = field(default_factory=Twist)
        covariance: List[float] = field(default_factory=lambda: [0.0] * 36)

    @dataclass
    class LaserScan:
        """ROS sensor_msgs/LaserScan に相当するデータ構造"""
        header: Header = field(default_factory=Header)
        angle_min: float = 0.0
        angle_max: float = 0.0
        angle_increment: float = 0.0
        time_increment: float = 0.0
        scan_time: float = 0.0
        range_min: float = 0.0
        range_max: float = 0.0
        ranges: List[float] = field(default_factory=list)
        intensities: List[float] = field(default_factory=list)
        
        @classmethod
        def from_simulator_data(cls, scan_data: dict, frame_id: str = "laser"):
            """シミュレーターのスキャンデータからLaserScanメッセージを生成"""
            return cls(
                header=Header(stamp=scan_data.get('timestamp', time.time()), frame_id=frame_id),
                angle_min=scan_data['angle_min'],
                angle_max=scan_data['angle_max'],
                angle_increment=scan_data['angle_increment'],
                range_min=scan_data['range_min'],
                range_max=scan_data['range_max'],
                ranges=scan_data['ranges'].tolist() if isinstance(scan_data['ranges'], np.ndarray) else scan_data['ranges'],
                intensities=scan_data['intensities'].tolist() if isinstance(scan_data['intensities'], np.ndarray) else scan_data['intensities']
            )

    @dataclass
    class PointCloud2:
        """ROS sensor_msgs/PointCloud2 に相当するデータ構造（簡略版）"""
        header: Header = field(default_factory=Header)
        height: int = 1
        width: int = 0
        fields: List[str] = field(default_factory=lambda: ["x", "y", "intensity"])
        is_bigendian: bool = False
        point_step: int = 12
        row_step: int = 0
        data: bytes = b''
        is_dense: bool = True
        
        @classmethod
        def from_points(cls, points: np.ndarray, intensities: Optional[np.ndarray] = None, frame_id: str = "laser"):
            """点群データからPointCloud2メッセージを生成"""
            if points.size == 0:
                return cls(header=Header(frame_id=frame_id))
                
            if intensities is None:
                intensities = np.ones(len(points))
                
            # 簡略化された実装（実際のROSでは複雑なバイナリ形式）
            return cls(
                header=Header(frame_id=frame_id),
                width=len(points),
                row_step=len(points) * 12,
                data=b''  # 実際の実装では点群データをバイナリ化
            )

    @dataclass
    class Odometry:
        """ROS nav_msgs/Odometry に相当するデータ構造"""
        header: Header = field(default_factory=Header)
        child_frame_id: str = "base_link"
        pose: PoseWithCovariance = field(default_factory=PoseWithCovariance)
        twist: TwistWithCovariance = field(default_factory=TwistWithCovariance)
        
        @classmethod
        def from_pose_and_velocity(cls, pose_2d: np.ndarray, velocity_2d: np.ndarray, 
                                  covariance: Optional[np.ndarray] = None, frame_id: str = "odom"):
            """2D姿勢と速度からOdometryメッセージを生成"""
            x, y, theta = pose_2d
            linear_vel, angular_vel = velocity_2d
            
            # 姿勢の共分散行列を設定
            pose_cov = [0.0] * 36
            if covariance is not None and covariance.shape == (3, 3):
                # 2D共分散を6D共分散行列に埋め込み
                pose_cov[0] = float(covariance[0, 0])   # x-x
                pose_cov[1] = float(covariance[0, 1])   # x-y
                pose_cov[5] = float(covariance[0, 2])   # x-yaw
                pose_cov[6] = float(covariance[1, 0])   # y-x
                pose_cov[7] = float(covariance[1, 1])   # y-y
                pose_cov[11] = float(covariance[1, 2])  # y-yaw
                pose_cov[30] = float(covariance[2, 0])  # yaw-x
                pose_cov[31] = float(covariance[2, 1])  # yaw-y
                pose_cov[35] = float(covariance[2, 2])  # yaw-yaw
            
            return cls(
                header=Header(frame_id=frame_id),
                pose=PoseWithCovariance(
                    pose=Pose.from_2d(x, y, theta),
                    covariance=pose_cov
                ),
                twist=TwistWithCovariance(
                    twist=Twist.from_2d(linear_vel, angular_vel)
                )
            )

    @dataclass
    class MapMetaData:
        """ROS nav_msgs/MapMetaData に相当するデータ構造"""
        map_load_time: float = field(default_factory=time.time)
        resolution: float = 0.05
        width: int = 0
        height: int = 0
        origin: Pose = field(default_factory=Pose)

    @dataclass
    class OccupancyGrid:
        """ROS nav_msgs/OccupancyGrid に相当するデータ構造"""
        header: Header = field(default_factory=Header)
        info: MapMetaData = field(default_factory=MapMetaData)
        data: List[int] = field(default_factory=list)
        
        @classmethod
        def from_occupancy_map(cls, occupancy_map, frame_id: str = "map"):
            """占有格子地図からOccupancyGridメッセージを生成"""
            binary_map = occupancy_map.get_occupancy_grid()
            height, width = binary_map.shape
            
            # ROSの占有格子では、0=空き、100=占有、-1=不明
            ros_data = []
            for j in range(height):
                for i in range(width):
                    if binary_map[j, i] > 0:
                        ros_data.append(100)  # 占有
                    else:
                        ros_data.append(0)    # 空き
            
            origin_pose = Pose.from_2d(
                occupancy_map.origin[0], 
                occupancy_map.origin[1], 
                occupancy_map.origin[2] if len(occupancy_map.origin) > 2 else 0.0
            )
            
            return cls(
                header=Header(frame_id=frame_id),
                info=MapMetaData(
                    resolution=occupancy_map.resolution,
                    width=width,
                    height=height,
                    origin=origin_pose
                ),
                data=ros_data
            )

    @dataclass
    class Transform:
        """ROS geometry_msgs/Transform に相当するデータ構造"""
        translation: Point = field(default_factory=Point)
        rotation: Quaternion = field(default_factory=Quaternion)
        
        @classmethod
        def from_2d(cls, x: float, y: float, theta: float):
            """2D変換から3D Transformを生成"""
            return cls(
                translation=Point(x=x, y=y, z=0.0),
                rotation=Quaternion.from_euler(0.0, 0.0, theta)
            )

    @dataclass
    class TransformStamped:
        """ROS geometry_msgs/TransformStamped に相当するデータ構造"""
        header: Header = field(default_factory=Header)
        child_frame_id: str = ""
        transform: Transform = field(default_factory=Transform)

# ROS 2メッセージ変換ユーティリティクラス
class ROS2MessageConverter:
    """
    シミュレーターのデータとROS 2メッセージ形式の間で変換を行うユーティリティクラス
    """
    
    @staticmethod
    def create_header(frame_id: str = "", stamp: Optional[float] = None):
        """ROS 2 Headerメッセージを作成"""
        if ROS2_AVAILABLE:
            from builtin_interfaces.msg import Time as TimeMsg
            header = Header()
            header.frame_id = frame_id
            if stamp is not None:
                # float timestampをTimeメッセージに変換
                sec = int(stamp)
                nanosec = int((stamp - sec) * 1e9)
                header.stamp = TimeMsg(sec=sec, nanosec=nanosec)
            return header
        else:
            # フォールバック
            return Header(
                seq=0,
                stamp=stamp if stamp is not None else time.time(),
                frame_id=frame_id
            )
    
    @staticmethod
    def pose_2d_to_ros(pose_2d: np.ndarray):
        """2D姿勢をROS Poseに変換"""
        x, y, theta = pose_2d
        
        if ROS2_AVAILABLE:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.position.z = 0.0
            
            # クォータニオンに変換
            cy = np.cos(theta * 0.5)
            sy = np.sin(theta * 0.5)
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = float(sy)
            pose.orientation.w = float(cy)
            return pose
        else:
            return Pose.from_2d(x, y, theta)
    
    @staticmethod
    def ros_pose_to_2d(pose) -> np.ndarray:
        """ROS Poseを2D姿勢に変換"""
        if ROS2_AVAILABLE:
            x = pose.position.x
            y = pose.position.y
            # クォータニオンからyaw角を計算
            q = pose.orientation
            yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 
                           1 - 2 * (q.y * q.y + q.z * q.z))
            return np.array([x, y, yaw])
        else:
            return np.array(pose.to_2d())
    
    @staticmethod
    def scan_data_to_ros(scan_data: dict, frame_id: str = "laser"):
        """シミュレーターのスキャンデータをROS 2 LaserScanに変換"""
        if ROS2_AVAILABLE:
            scan = LaserScan()
            scan.header = ROS2MessageConverter.create_header(frame_id, 
                                                           scan_data.get('timestamp'))
            scan.angle_min = float(scan_data['angle_min'])
            scan.angle_max = float(scan_data['angle_max'])
            scan.angle_increment = float(scan_data['angle_increment'])
            scan.range_min = float(scan_data['range_min'])
            scan.range_max = float(scan_data['range_max'])
            scan.ranges = [float(r) for r in scan_data['ranges']]
            scan.intensities = [float(i) for i in scan_data['intensities']]
            return scan
        else:
            # フォールバック：カスタムLaserScanクラスのクラスメソッドを使用
            return LaserScan(
                header=Header(stamp=scan_data.get('timestamp', time.time()), frame_id=frame_id),
                angle_min=scan_data['angle_min'],
                angle_max=scan_data['angle_max'],
                angle_increment=scan_data['angle_increment'],
                range_min=scan_data['range_min'],
                range_max=scan_data['range_max'],
                ranges=scan_data['ranges'].tolist() if isinstance(scan_data['ranges'], np.ndarray) else scan_data['ranges'],
                intensities=scan_data['intensities'].tolist() if isinstance(scan_data['intensities'], np.ndarray) else scan_data['intensities']
            )
    
    @staticmethod
    def points_to_ros(points: np.ndarray, intensities: Optional[np.ndarray] = None, 
                     frame_id: str = "laser"):
        """点群データをROS 2 PointCloud2に変換"""
        if ROS2_AVAILABLE:
            from sensor_msgs_py import point_cloud2
            
            if points.size == 0:
                return PointCloud2()
            
            # PointCloud2の作成（sensor_msgs_pyを使用）
            header = ROS2MessageConverter.create_header(frame_id)
            
            if intensities is not None:
                # x, y, intensityを含む点群
                fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='intensity', offset=8, datatype=PointField.FLOAT32, count=1)
                ]
                points_with_intensity = []
                for i, point in enumerate(points):
                    intensity = intensities[i] if i < len(intensities) else 0.0
                    points_with_intensity.append([point[0], point[1], intensity])
                
                return point_cloud2.create_cloud(header, fields, points_with_intensity)
            else:
                # x, yのみの点群
                return point_cloud2.create_cloud_xyz32(header, points.tolist())
        else:
            # フォールバック：カスタムPointCloud2クラスを使用
            if points.size == 0:
                return PointCloud2(header=Header(frame_id=frame_id))
                
            if intensities is None:
                intensities = np.ones(len(points))
                
            # 簡略化された実装
            return PointCloud2(
                header=Header(frame_id=frame_id),
                width=len(points),
                row_step=len(points) * 12,
                data=b''  # 実際の実装では点群データをバイナリ化
            )
    
    @staticmethod
    def create_odometry(pose_2d: np.ndarray, velocity_2d: np.ndarray, 
                       covariance: Optional[np.ndarray] = None, 
                       frame_id: str = "odom", child_frame_id: str = "base_link"):
        """2D姿勢と速度からROS 2 Odometryメッセージを生成"""
        if ROS2_AVAILABLE:
            odom = Odometry()
            odom.header = ROS2MessageConverter.create_header(frame_id)
            odom.child_frame_id = child_frame_id
            
            # 姿勢設定
            odom.pose.pose = ROS2MessageConverter.pose_2d_to_ros(pose_2d)
            
            # 速度設定
            linear_vel, angular_vel = velocity_2d
            odom.twist.twist.linear.x = float(linear_vel)
            odom.twist.twist.angular.z = float(angular_vel)
            
            # 共分散行列設定
            if covariance is not None and covariance.shape == (3, 3):
                pose_cov = [0.0] * 36
                pose_cov[0] = float(covariance[0, 0])   # x-x
                pose_cov[1] = float(covariance[0, 1])   # x-y
                pose_cov[5] = float(covariance[0, 2])   # x-yaw
                pose_cov[6] = float(covariance[1, 0])   # y-x
                pose_cov[7] = float(covariance[1, 1])   # y-y
                pose_cov[11] = float(covariance[1, 2])  # y-yaw
                pose_cov[30] = float(covariance[2, 0])  # yaw-x
                pose_cov[31] = float(covariance[2, 1])  # yaw-y
                pose_cov[35] = float(covariance[2, 2])  # yaw-yaw
                odom.pose.covariance = pose_cov
            
            return odom
        else:
            # フォールバック：カスタムOdometryクラスを使用
            x, y, theta = pose_2d
            linear_vel, angular_vel = velocity_2d
            
            # 姿勢の共分散行列を設定
            pose_cov = [0.0] * 36
            if covariance is not None and covariance.shape == (3, 3):
                # 2D共分散を6D共分散行列に埋め込み
                pose_cov[0] = float(covariance[0, 0])   # x-x
                pose_cov[1] = float(covariance[0, 1])   # x-y
                pose_cov[5] = float(covariance[0, 2])   # x-yaw
                pose_cov[6] = float(covariance[1, 0])   # y-x
                pose_cov[7] = float(covariance[1, 1])   # y-y
                pose_cov[11] = float(covariance[1, 2])  # y-yaw
                pose_cov[30] = float(covariance[2, 0])  # yaw-x
                pose_cov[31] = float(covariance[2, 1])  # yaw-y
                pose_cov[35] = float(covariance[2, 2])  # yaw-yaw
            
            return Odometry(
                header=Header(frame_id=frame_id),
                child_frame_id=child_frame_id,
                pose=PoseWithCovariance(
                    pose=Pose.from_2d(x, y, theta),
                    covariance=pose_cov
                ),
                twist=TwistWithCovariance(
                    twist=Twist.from_2d(linear_vel, angular_vel)
                )
            )
    
    @staticmethod
    def occupancy_map_to_ros(occupancy_map, frame_id: str = "map"):
        """占有格子地図をROS 2 OccupancyGridに変換"""
        if ROS2_AVAILABLE:
            grid = OccupancyGrid()
            grid.header = ROS2MessageConverter.create_header(frame_id)
            
            binary_map = occupancy_map.get_occupancy_grid()
            height, width = binary_map.shape
            
            # メタデータ設定
            grid.info.resolution = float(occupancy_map.resolution)
            grid.info.width = width
            grid.info.height = height
            grid.info.origin = ROS2MessageConverter.pose_2d_to_ros(
                np.array([occupancy_map.origin[0], 
                         occupancy_map.origin[1], 
                         occupancy_map.origin[2] if len(occupancy_map.origin) > 2 else 0.0])
            )
            
            # データ変換
            ros_data = []
            for j in range(height):
                for i in range(width):
                    if binary_map[j, i] > 0:
                        ros_data.append(100)  # 占有
                    else:
                        ros_data.append(0)    # 空き
            grid.data = ros_data
            
            return grid
        else:
            # フォールバック：カスタムOccupancyGridクラスを使用
            binary_map = occupancy_map.get_occupancy_grid()
            height, width = binary_map.shape
            
            # ROSの占有格子では、0=空き、100=占有、-1=不明
            ros_data = []
            for j in range(height):
                for i in range(width):
                    if binary_map[j, i] > 0:
                        ros_data.append(100)  # 占有
                    else:
                        ros_data.append(0)    # 空き
            
            origin_pose = Pose.from_2d(
                occupancy_map.origin[0], 
                occupancy_map.origin[1], 
                occupancy_map.origin[2] if len(occupancy_map.origin) > 2 else 0.0
            )
            
            return OccupancyGrid(
                header=Header(frame_id=frame_id),
                info=MapMetaData(
                    resolution=occupancy_map.resolution,
                    width=width,
                    height=height,
                    origin=origin_pose
                ),
                data=ros_data
            )
    
    @staticmethod
    def create_transform_2d(from_frame: str, to_frame: str, 
                           translation: Tuple[float, float], rotation: float):
        """2D変換からROS 2 TransformStampedを作成"""
        if ROS2_AVAILABLE:
            t = TransformStamped()
            t.header = ROS2MessageConverter.create_header(from_frame)
            t.child_frame_id = to_frame
            
            x, y = translation
            t.transform.translation.x = float(x)
            t.transform.translation.y = float(y)
            t.transform.translation.z = 0.0
            
            # 回転をクォータニオンに変換
            cy = np.cos(rotation * 0.5)
            sy = np.sin(rotation * 0.5)
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = float(sy)
            t.transform.rotation.w = float(cy)
            
            return t
        else:
            # フォールバック：カスタムTransformStampedクラスを使用
            x, y = translation
            return TransformStamped(
                header=Header(frame_id=from_frame),
                child_frame_id=to_frame,
                transform=Transform.from_2d(x, y, rotation)
            )

# NDTマッチングの結果をROSメッセージ形式で返すためのユーティリティ
@dataclass
class NDTMatchingResult:
    """NDTマッチングの結果を格納するデータ構造"""
    success: bool = False
    transform: Transform = field(default_factory=Transform)
    score: float = 0.0
    iterations: int = 0
    computation_time: float = 0.0
    covariance: List[float] = field(default_factory=lambda: [0.0] * 36)
    
    @classmethod
    def from_ndt_result(cls, params: np.ndarray, score: float, iterations: int, 
                       computation_time: float, covariance: Optional[np.ndarray] = None) -> 'NDTMatchingResult':
        """NDTマッチングの結果からメッセージを生成"""
        theta, x, y = params
        
        # 共分散行列の設定
        cov = [0.0] * 36
        if covariance is not None and covariance.shape == (3, 3):
            # 2D共分散を6D共分散行列に埋め込み
            cov[0] = float(covariance[0, 0])   # x-x
            cov[1] = float(covariance[0, 1])   # x-y
            cov[5] = float(covariance[0, 2])   # x-yaw
            cov[6] = float(covariance[1, 0])   # y-x
            cov[7] = float(covariance[1, 1])   # y-y
            cov[11] = float(covariance[1, 2])  # y-yaw
            cov[30] = float(covariance[2, 0])  # yaw-x
            cov[31] = float(covariance[2, 1])  # yaw-y
            cov[35] = float(covariance[2, 2])  # yaw-yaw
        
        # ROS 2が利用可能な場合は実際のTransformメッセージを作成
        if ROS2_AVAILABLE:
            transform = Transform()
            transform.translation.x = float(x)
            transform.translation.y = float(y)
            transform.translation.z = 0.0
            cy = np.cos(theta * 0.5)
            sy = np.sin(theta * 0.5)
            transform.rotation.x = 0.0
            transform.rotation.y = 0.0
            transform.rotation.z = float(sy)
            transform.rotation.w = float(cy)
        else:
            transform = Transform.from_2d(x, y, theta)
        
        return cls(
            success=True,
            transform=transform,
            score=score,
            iterations=iterations,
            computation_time=computation_time,
            covariance=cov
        )

# ROS 2ノード管理ユーティリティ
class ROS2NodeManager:
    """ROS 2ノードの作成と管理を行うユーティリティクラス"""
    
    def __init__(self, node_name: str = "ndt_localizer"):
        self.node_name = node_name
        self.node = None
        self.initialized = False
        
        if ROS2_AVAILABLE:
            try:
                # rclpyの初期化（既に初期化されている場合は無視）
                if not rclpy.ok():
                    rclpy.init()
                
                # ノードの作成
                self.node = rclpy.create_node(node_name)
                self.initialized = True
                print(f"ROS 2ノード '{node_name}' を作成しました")
                
            except Exception as e:
                print(f"ROS 2ノード作成エラー: {e}")
                self.initialized = False
        else:
            print("ROS 2が利用できません。スタンドアロンモードで動作します。")
    
    def create_publisher(self, msg_type, topic: str, qos_depth: int = 10):
        """パブリッシャーを作成"""
        if self.initialized and self.node:
            return self.node.create_publisher(msg_type, topic, qos_depth)
        return None
    
    def create_subscription(self, msg_type, topic: str, callback, qos_depth: int = 10):
        """サブスクリプションを作成"""
        if self.initialized and self.node:
            return self.node.create_subscription(msg_type, topic, callback, qos_depth)
        return None
    
    def spin_once(self, timeout_sec: float = 0.1):
        """ノードを一度だけスピン"""
        if self.initialized and self.node:
            rclpy.spin_once(self.node, timeout_sec=timeout_sec)
    
    def get_clock(self):
        """ノードのクロックを取得"""
        if self.initialized and self.node:
            return self.node.get_clock()
        return None
    
    def get_logger(self):
        """ノードのロガーを取得"""
        if self.initialized and self.node:
            return self.node.get_logger()
        return None
    
    def shutdown(self):
        """ノードとrclpyをシャットダウン"""
        if self.initialized:
            if self.node:
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
            self.initialized = False
            print("ROS 2ノードをシャットダウンしました")

# レガシー互換性のためのエイリアス
ROSMessageConverter = ROS2MessageConverter  # 後方互換性のため 