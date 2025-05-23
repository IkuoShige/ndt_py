import numpy as np
from scipy.linalg import inv

class NDTCell:
    """
    NDTセル（Normal Distribution Transform）のクラス
    各セル内の点群を多変量正規分布で表現する
    """
    def __init__(self):
        self.points = []
        self.mean = None
        self.covariance = None
        self.count = 0
        self.is_valid = False
    
    def add_point(self, point):
        """
        セルに点を追加
        
        Args:
            point (numpy.ndarray): 2Dの点 [x, y]
        """
        self.points.append(point)
        self.count += 1
    
    def compute_statistics(self, min_point_count=3):
        """
        セル内の点群から統計情報（平均と共分散行列）を計算
        
        Args:
            min_point_count (int): 有効なセルとみなすための最小点数
        """
        if self.count < min_point_count:
            self.is_valid = False
            return
        
        points_array = np.array(self.points)
        self.mean = np.mean(points_array, axis=0)
        
        # 共分散行列を計算
        centered_points = points_array - self.mean
        self.covariance = np.dot(centered_points.T, centered_points) / self.count
        
        # 数値的安定性のための正則化
        # 共分散行列が特異にならないように対角要素に小さな値を追加
        epsilon = 0.01
        self.covariance[0, 0] += epsilon
        self.covariance[1, 1] += epsilon
        
        self.is_valid = True


class NDTGrid:
    """
    NDTグリッド（2Dセル分割）のクラス
    """
    def __init__(self, cell_size=1.0):
        """
        初期化関数
        
        Args:
            cell_size (float): NDTセルの一辺のサイズ [m]
        """
        self.cell_size = cell_size
        self.cells = {}  # ハッシュマップ: (i, j) -> NDTCell
    
    def clear(self):
        """
        グリッドをクリア
        """
        self.cells.clear()
    
    def get_cell_index(self, point):
        """
        点の座標からセルのインデックスを取得
        
        Args:
            point (numpy.ndarray): 2D点 [x, y]
        
        Returns:
            tuple: セルのインデックス (i, j)
        """
        i = int(np.floor(point[0] / self.cell_size))
        j = int(np.floor(point[1] / self.cell_size))
        return (i, j)
    
    def add_point(self, point):
        """
        点をグリッドに追加
        
        Args:
            point (numpy.ndarray): 2D点 [x, y]
        """
        cell_idx = self.get_cell_index(point)
        if cell_idx not in self.cells:
            self.cells[cell_idx] = NDTCell()
        
        self.cells[cell_idx].add_point(point)
    
    def add_points(self, points):
        """
        複数の点をグリッドに追加
        
        Args:
            points (numpy.ndarray): 点のリスト [[x1, y1], [x2, y2], ...]
        """
        for point in points:
            self.add_point(point)
    
    def compute_all_statistics(self, min_point_count=3):
        """
        全てのセルの統計情報を計算
        
        Args:
            min_point_count (int): 有効なセルとみなすための最小点数
        """
        for cell in self.cells.values():
            cell.compute_statistics(min_point_count)


class NDTMatcher:
    """
    NDTマッチング（スキャンマッチング）のクラス
    """
    def __init__(self, target_points, cell_size=1.0, max_iterations=30, 
                 tolerance=1e-2, step_size=0.1):
        """
        初期化関数
        
        Args:
            target_points (numpy.ndarray): ターゲットの点群（地図上の点群）
            cell_size (float): NDTセルのサイズ
            max_iterations (int): 最大反復回数
            tolerance (float): 収束判定のための閾値
            step_size (float): パラメータ更新のステップサイズ
        """
        self.target_ndt = NDTGrid(cell_size)
        self.target_ndt.add_points(target_points)
        self.target_ndt.compute_all_statistics()
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.step_size = step_size
    
    def transform_point(self, point, params):
        """
        2D変換（回転とオフセット）を点に適用
        
        Args:
            point (numpy.ndarray): 変換する点 [x, y]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
                theta: 回転角度 [rad]
                x, y: 並進 [m]
        
        Returns:
            numpy.ndarray: 変換後の点 [x', y']
        """
        theta, tx, ty = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        x, y = point
        x_transformed = x * cos_theta - y * sin_theta + tx
        y_transformed = x * sin_theta + y * cos_theta + ty
        
        return np.array([x_transformed, y_transformed])
    
    def transform_points(self, points, params):
        """
        点群に変換を適用
        
        Args:
            points (numpy.ndarray): 点群 [[x1, y1], [x2, y2], ...]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
        
        Returns:
            numpy.ndarray: 変換後の点群 [[x1', y1'], [x2', y2'], ...]
        """
        transformed_points = np.zeros_like(points)
        for i, point in enumerate(points):
            transformed_points[i] = self.transform_point(point, params)
        return transformed_points
    
    def compute_score_gradient(self, source_point, params, eps=1e-6):
        """
        1点の変換に対するスコアと勾配を計算
        
        Args:
            source_point (numpy.ndarray): ソース点 [x, y]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
            eps (float): 数値微分のためのステップサイズ
        
        Returns:
            tuple: (score, grad)
                score: 点のスコア
                grad: 勾配 [d_score/d_theta, d_score/d_x, d_score/d_y]
        """
        # 点を変換
        transformed_point = self.transform_point(source_point, params)
        cell_idx = self.target_ndt.get_cell_index(transformed_point)
        
        # セルが存在し、有効かどうか確認
        if cell_idx not in self.target_ndt.cells or not self.target_ndt.cells[cell_idx].is_valid:
            return 0.0, np.zeros(3)
        
        # セルの平均と共分散を取得
        cell = self.target_ndt.cells[cell_idx]
        mean = cell.mean
        cov = cell.covariance
        cov_inv = inv(cov)
        
        # 平均からの差分
        d = transformed_point - mean
        
        # スコア計算（正規分布のマイナス対数尤度に基づく）
        exponent = -0.5 * d.dot(cov_inv).dot(d)
        score = np.exp(exponent)
        
        # 勾配の計算（数値微分）
        grad = np.zeros(3)
        
        # theta に関する勾配
        params_theta_plus = params.copy()
        params_theta_plus[0] += eps
        transformed_point_theta_plus = self.transform_point(source_point, params_theta_plus)
        d_theta_plus = transformed_point_theta_plus - mean
        exponent_theta_plus = -0.5 * d_theta_plus.dot(cov_inv).dot(d_theta_plus)
        score_theta_plus = np.exp(exponent_theta_plus)
        grad[0] = (score_theta_plus - score) / eps
        
        # x に関する勾配
        params_x_plus = params.copy()
        params_x_plus[1] += eps
        transformed_point_x_plus = self.transform_point(source_point, params_x_plus)
        d_x_plus = transformed_point_x_plus - mean
        exponent_x_plus = -0.5 * d_x_plus.dot(cov_inv).dot(d_x_plus)
        score_x_plus = np.exp(exponent_x_plus)
        grad[1] = (score_x_plus - score) / eps
        
        # y に関する勾配
        params_y_plus = params.copy()
        params_y_plus[2] += eps
        transformed_point_y_plus = self.transform_point(source_point, params_y_plus)
        d_y_plus = transformed_point_y_plus - mean
        exponent_y_plus = -0.5 * d_y_plus.dot(cov_inv).dot(d_y_plus)
        score_y_plus = np.exp(exponent_y_plus)
        grad[2] = (score_y_plus - score) / eps
        
        return score, grad
    
    def compute_hessian(self, source_point, params, eps=1e-6):
        """
        1点の変換に対するヘッセ行列（二階微分）を計算
        
        Args:
            source_point (numpy.ndarray): ソース点 [x, y]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
            eps (float): 数値微分のためのステップサイズ
        
        Returns:
            numpy.ndarray: 3x3のヘッセ行列
        """
        # 勾配を計算
        score, grad = self.compute_score_gradient(source_point, params)
        
        # 数値微分でヘッセ行列を計算
        hessian = np.zeros((3, 3))
        
        # 各パラメータについてヘッセ行列の要素を計算
        for i in range(3):
            # パラメータをわずかに変更
            params_plus = params.copy()
            params_plus[i] += eps
            
            # 変更後の勾配を計算
            _, grad_plus = self.compute_score_gradient(source_point, params_plus)
            
            # ヘッセ行列の計算（二階微分）
            hessian[:, i] = (grad_plus - grad) / eps
        
        # 数値的安定性のための正則化
        for i in range(3):
            hessian[i, i] += 1e-6
            
        return hessian
    
    def align_gradient_descent(self, source_points, initial_guess=None):
        """
        ソース点群をターゲットにアラインする（NDTマッチングの実行）
        
        Args:
            source_points (numpy.ndarray): ソース点群 [[x1, y1], [x2, y2], ...]
            initial_guess (numpy.ndarray): 初期推定値 [theta, x, y]
        
        Returns:
            tuple: (最適パラメータ, スコア, 反復回数)
        """
        # 初期推定値が指定されていない場合はゼロで初期化
        if initial_guess is None:
            params = np.zeros(3)  # [theta, x, y]
        else:
            params = initial_guess.copy()
        
        prev_score = -float('inf')
        
        for iteration in range(self.max_iterations):
            total_score = 0.0
            total_grad = np.zeros(3)
            
            # 各点のスコアと勾配を計算して合計
            for point in source_points:
                score, grad = self.compute_score_gradient(point, params)
                total_score += score
                total_grad += grad
            
            # 勾配が小さい場合は収束したとみなす
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm < self.tolerance:
                break
            
            # 勾配の方向にパラメータを更新
            params += self.step_size * (total_grad / len(source_points))
            
            # スコアの変化が小さい場合も収束したとみなす
            score_change = abs(total_score - prev_score)
            if iteration > 0 and score_change < self.tolerance * abs(prev_score):
                break
            
            prev_score = total_score
        
        return params, prev_score, iteration + 1
    
    def align_newton(self, source_points, initial_guess=None, damping=0.1):
        """
        ニュートン法を用いてソース点群をターゲットにアラインする
        
        Args:
            source_points (numpy.ndarray): ソース点群 [[x1, y1], [x2, y2], ...]
            initial_guess (numpy.ndarray): 初期推定値 [theta, x, y]
            damping (float): ダンピング係数（0に近いほど純粋なニュートン法）
        
        Returns:
            tuple: (最適パラメータ, スコア, 反復回数)
        """
        # 初期推定値が指定されていない場合はゼロで初期化
        if initial_guess is None:
            params = np.zeros(3)  # [theta, x, y]
        else:
            params = initial_guess.copy()
        
        prev_score = -float('inf')
        
        for iteration in range(self.max_iterations):
            total_score = 0.0
            total_grad = np.zeros(3)
            total_hessian = np.zeros((3, 3))
            
            # 各点のスコア、勾配、ヘッセ行列を計算して合計
            for point in source_points:
                score, grad = self.compute_score_gradient(point, params)
                hessian = self.compute_hessian(point, params)
                
                total_score += score
                total_grad += grad
                total_hessian += hessian
            
            # 勾配が小さい場合は収束したとみなす
            grad_norm = np.linalg.norm(total_grad)
            if grad_norm < self.tolerance:
                break
            
            # ヘッセ行列を正則化（Levenberg-Marquardt法の要素を追加）
            regularized_hessian = total_hessian + damping * np.eye(3)
            
            try:
                # ヘッセ行列の逆行列を計算
                hessian_inv = inv(regularized_hessian)
                
                # ニュートン法でパラメータを更新
                delta_params = hessian_inv.dot(total_grad) / len(source_points)
                params += delta_params
            except np.linalg.LinAlgError:
                # 逆行列計算が失敗した場合は勾配降下法にフォールバック
                params += self.step_size * (total_grad / len(source_points))
                print("警告: ヘッセ行列の逆行列計算に失敗しました。勾配降下法にフォールバックします。")
            
            # スコアの変化が小さい場合も収束したとみなす
            score_change = abs(total_score - prev_score)
            if iteration > 0 and score_change < self.tolerance * abs(prev_score):
                break
            
            prev_score = total_score
        
        return params, prev_score, iteration + 1
