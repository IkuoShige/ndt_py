import yaml
import numpy as np
from PIL import Image

class OccupancyGridMap:
    """
    占有格子地図を扱うクラス
    pgmファイルとyamlファイルから地図情報を読み込み、
    座標変換などの機能を提供する
    """
    def __init__(self, pgm_path, yaml_path):
        """
        初期化関数
        
        Args:
            pgm_path (str): PGM画像ファイルへのパス
            yaml_path (str): 地図メタデータYAMLファイルへのパス
        """
        # YAML設定ファイルを読み込み
        with open(yaml_path, 'r') as f:
            self.metadata = yaml.safe_load(f)
        
        # PGM画像を読み込み
        self.map_img = Image.open(pgm_path)
        self.map_data = np.array(self.map_img)
        
        # メタデータから重要なパラメータを取得
        self.resolution = self.metadata.get('resolution', 0.05)  # メートル/ピクセル
        self.origin = self.metadata.get('origin', [-10.0, -10.0, 0.0])  # [x, y, theta]
        
        # 地図のサイズを取得
        self.height, self.width = self.map_data.shape
        
        # 閾値（通常、0~255の値で、これよりも大きい値は空きスペース）
        self.occupancy_threshold = self.metadata.get('occupied_thresh', 0.65) * 255
        
        print(f"地図を読み込みました: {pgm_path}")
        print(f"サイズ: {self.width}x{self.height}ピクセル")
        print(f"解像度: {self.resolution}メートル/ピクセル")
        print(f"原点: {self.origin}")
    
    def is_occupied(self, x, y):
        """
        指定されたワールド座標(x,y)が占有されているかをチェック
        
        Args:
            x (float): ワールド座標のx [m]
            y (float): ワールド座標のy [m]
        
        Returns:
            bool: 占有されている場合はTrue、そうでない場合はFalse
        """
        # ワールド座標からピクセル座標への変換
        px, py = self.world_to_pixel(x, y)
        
        # 地図の範囲外の場合は占有されているとみなす
        if px < 0 or px >= self.width or py < 0 or py >= self.height:
            return True
        
        # 地図データからピクセル値を取得し、閾値と比較
        # PGMでは、通常値が大きいほど空きスペース（白）を表す
        pixel_value = self.map_data[py, px]
        return pixel_value < self.occupancy_threshold
    
    def world_to_pixel(self, x, y):
        """
        ワールド座標をピクセル座標に変換
        
        Args:
            x (float): ワールド座標のx [m]
            y (float): ワールド座標のy [m]
        
        Returns:
            tuple: ピクセル座標(px, py)
        """
        px = int((x - self.origin[0]) / self.resolution)
        py = int((y - self.origin[1]) / self.resolution)
        return px, py
    
    def pixel_to_world(self, px, py):
        """
        ピクセル座標をワールド座標に変換
        
        Args:
            px (int): ピクセル座標のx
            py (int): ピクセル座標のy
        
        Returns:
            tuple: ワールド座標(x, y) [m]
        """
        x = px * self.resolution + self.origin[0]
        y = py * self.resolution + self.origin[1]
        return x, y
    
    def get_occupancy_grid(self):
        """
        占有格子地図データを二値化して返す
        
        Returns:
            numpy.ndarray: 占有格子地図データ（0: 空きスペース、1: 占有）
        """
        binary_map = (self.map_data < self.occupancy_threshold).astype(np.uint8)
        return binary_map
    
    def get_empty_positions(self, num_positions=1):
        """
        地図上の空きスペースからランダムな位置を返す
        
        Args:
            num_positions (int): 返す位置の数
        
        Returns:
            list: (x, y)のタプルのリスト
        """
        # 空きスペースのインデックスを取得
        empty_indices = np.where(self.map_data > self.occupancy_threshold)
        empty_indices = list(zip(empty_indices[1], empty_indices[0]))  # (px, py)のリスト
        
        # ランダムに選択
        if len(empty_indices) < num_positions:
            num_positions = len(empty_indices)
        
        selected_indices = np.random.choice(len(empty_indices), num_positions, replace=False)
        selected_pixels = [empty_indices[i] for i in selected_indices]
        
        # ワールド座標に変換
        world_positions = [self.pixel_to_world(px, py) for px, py in selected_pixels]
        
        return world_positions
