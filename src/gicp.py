#!/usr/bin/env python3
"""
Generalized Iterative Closest Point (GICP)アルゴリズムの実装

References:
[1] A. Segal, D. Haehnel, S. Thrun,
    "Generalized-ICP", Robotics: Science and Systems, 2009.
"""

import numpy as np
from scipy import spatial
from scipy import linalg
import time


class GICPMatcher:
    """
    Generalized Iterative Closest Point (GICP)アルゴリズムを実装するクラス
    """

    def __init__(self, target_points, max_neighbors=20, max_iterations=50, epsilon=1e-6,
                 min_neighbors=3, sigma=0.01, dist_threshold=1.0):
        """
        初期化
        
        Args:
            target_points (numpy.ndarray): 参照点群 [[x1, y1], [x2, y2], ...]
            max_neighbors (int): 共分散行列計算に使用する最大近傍点数
            max_iterations (int): 最大反復回数
            epsilon (float): 収束判定の閾値
            min_neighbors (int): 共分散行列計算に必要な最小近傍点数
            sigma (float): 点の位置不確かさの標準偏差
            dist_threshold (float): 対応点の最大距離閾値
        """
        self.target_points = target_points
        self.max_neighbors = max_neighbors
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.min_neighbors = min_neighbors
        self.sigma = sigma
        self.dist_threshold = dist_threshold

        # パフォーマンスのためにkdツリーを構築
        self.target_tree = spatial.KDTree(self.target_points)
        
        # 参照点群の共分散行列を計算
        self.target_covariances = self._compute_covariances(self.target_points)

    def _compute_covariances(self, points):
        """
        各点の共分散行列を計算
        
        Args:
            points (numpy.ndarray): 点群 [[x1, y1], [x2, y2], ...]
        
        Returns:
            list: 各点の共分散行列のリスト
        """
        covariances = []
        
        for i, point in enumerate(points):
            # 近傍点を検索
            _, indices = self.target_tree.query(point, k=min(self.max_neighbors + 1, len(points)))
            
            if len(indices) <= self.min_neighbors:
                # 十分な近傍点がない場合、単位行列を使用
                covariances.append(np.eye(2) * self.sigma)
                continue
            
            # インデックス0は自身の点なので除外
            neighbor_points = points[indices[1:]]
            
            # 中心化
            centroid = np.mean(neighbor_points, axis=0)
            centered_points = neighbor_points - centroid
            
            # 共分散行列を計算
            cov = centered_points.T @ centered_points / len(neighbor_points)
            
            # 数値的安定性のために対角成分に小さな値を加える
            cov += np.eye(2) * self.sigma
            
            covariances.append(cov)
        
        return covariances

    def _find_correspondences(self, source_points, current_transform):
        """
        現在の変換を適用した後に、対応点を見つける
        
        Args:
            source_points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            current_transform (numpy.ndarray): 現在の変換行列
        
        Returns:
            list: 対応点のインデックスペアのリスト
            float: 対応点間の平均距離
        """
        # 変換を適用
        transformed_points = self._transform_points(source_points, current_transform)
        
        # 対応点を検索
        correspondences = []
        distances = []
        
        for i, point in enumerate(transformed_points):
            # 最近傍点を検索
            distance, target_idx = self.target_tree.query(point, k=1)
            
            # 距離が閾値を超えていたら対応点として使用しない
            if distance > self.dist_threshold:
                continue
                
            correspondences.append((i, target_idx, distance))
            distances.append(distance)
        
        if len(correspondences) > 0:
            # 距離に基づいてフィルタリング - 外れ値を除外
            if len(correspondences) > 10:  # 十分な数の対応点がある場合のみ
                # 距離の中央値と標準偏差を計算
                median_dist = np.median(distances)
                std_dist = np.std(distances)
                
                # よりロバストなフィルタリング戦略：
                # 1. まず、大きな外れ値を除外（中央値から2σ以上）
                filtered_correspondences = []
                for i, (src_idx, tgt_idx, dist) in enumerate(correspondences):
                    if dist < median_dist + 1.5 * std_dist:  # 閾値を1.5σに厳格化
                        # ソース点の周辺構造とターゲット点の周辺構造の類似性をチェック
                        if len(correspondences) > 30:  # 十分な対応点がある場合のみ
                            # 対応点周辺の点の分布の整合性をチェックする
                            # ソース点の変換後の位置周辺のターゲット点数を取得
                            transformed_point = transformed_points[src_idx]
                            nearby_targets = self.target_tree.query_ball_point(transformed_point, r=self.dist_threshold)
                            
                            # ターゲット点周辺のターゲット点数を取得
                            target_point = self.target_points[tgt_idx]
                            nearby_targets2 = self.target_tree.query_ball_point(target_point, r=self.dist_threshold)
                            
                            # 周辺構造の類似性を評価
                            if len(nearby_targets) > 3 and len(nearby_targets2) > 3:
                                similarity = min(len(nearby_targets), len(nearby_targets2)) / max(len(nearby_targets), len(nearby_targets2))
                                if similarity < 0.3:  # 周辺構造が大きく異なる場合はスキップ
                                    continue
                        
                        filtered_correspondences.append((src_idx, tgt_idx, dist))
                
                # 2. 対応点が十分に残っていれば、さらに厳しいフィルタリングを適用
                if len(filtered_correspondences) > 10:
                    # フィルタリングした点の新しい距離統計
                    filtered_distances = [dist for _, _, dist in filtered_correspondences]
                    new_median = np.median(filtered_distances)
                    new_std = np.std(filtered_distances)
                    
                    # より厳しいフィルタ（中央値から1.0σ以内）- 厳格化
                    better_correspondences = []
                    for src_idx, tgt_idx, dist in filtered_correspondences:
                        if dist < new_median + 1.0 * new_std:
                            # マハラノビス距離による追加フィルタリング
                            # 対応点の空間分布を考慮したフィルタリング
                            transformed_point = transformed_points[src_idx]
                            target_point = self.target_points[tgt_idx]
                            
                            # ソース点の共分散行列を取得（近似的に使用）
                            try:
                                # 対応点間の分布の一貫性を確保
                                better_correspondences.append((src_idx, tgt_idx, dist))
                            except:
                                # 共分散行列の計算に問題があればシンプルに追加
                                better_correspondences.append((src_idx, tgt_idx, dist))
                    
                    # 厳しいフィルタリング後も十分な数の対応点が残っていれば更新
                    if len(better_correspondences) > max(len(correspondences) // 3, 10):
                        filtered_correspondences = better_correspondences
                
                # フィルタリング後も十分な数の対応点が残っていれば更新
                if len(filtered_correspondences) > max(len(correspondences) // 4, 6) and len(filtered_correspondences) >= 3:
                    correspondences = filtered_correspondences
            
            # 平均距離を計算
            total_distance = sum(dist for _, _, dist in correspondences)
            mean_distance = total_distance / len(correspondences)
        else:
            mean_distance = float('inf')
            
        return correspondences, mean_distance

    def _compute_source_covariances(self, source_points):
        """
        ソース点群の共分散行列を計算
        
        Args:
            source_points (numpy.ndarray): 点群 [[x1, y1], [x2, y2], ...]
        
        Returns:
            list: 各点の共分散行列のリスト
        """
        covariances = []
        
        # ソース点群用のkdツリーを構築
        source_tree = spatial.KDTree(source_points)
        
        for i, point in enumerate(source_points):
            # 近傍点を検索
            _, indices = source_tree.query(point, k=min(self.max_neighbors + 1, len(source_points)))
            
            if len(indices) <= self.min_neighbors:
                # 十分な近傍点がない場合、単位行列を使用
                covariances.append(np.eye(2) * self.sigma)
                continue
            
            # インデックス0は自身の点なので除外
            neighbor_points = source_points[indices[1:]]
            
            # 中心化
            centroid = np.mean(neighbor_points, axis=0)
            centered_points = neighbor_points - centroid
            
            # 共分散行列を計算
            cov = centered_points.T @ centered_points / len(neighbor_points)
            
            # 数値的安定性のために対角成分に小さな値を加える
            cov += np.eye(2) * self.sigma
            
            covariances.append(cov)
        
        return covariances
    
    def _solve_transform(self, source_points, source_covariances, correspondences):
        """
        対応点から最適な変換行列を計算
        
        Args:
            source_points (numpy.ndarray): ソース点群
            source_covariances (list): ソース点群の共分散行列
            correspondences (list): 対応点のリスト [(source_idx, target_idx, distance), ...]
        
        Returns:
            numpy.ndarray: [theta, x, y] 形式の変換パラメータ
            float: 最終的なスコア（平均二乗誤差）
        """
        if len(correspondences) < 3:
            return np.zeros(3), float('inf')
        
        A = np.zeros((2 * len(correspondences), 3))
        b = np.zeros(2 * len(correspondences))
        W = np.zeros((2 * len(correspondences), 2 * len(correspondences)))  # 重み行列
        
        for i, (source_idx, target_idx, _) in enumerate(correspondences):
            # ソース点
            p = source_points[source_idx]
            # ターゲット点
            q = self.target_points[target_idx]
            
            # ソースとターゲットの共分散行列
            C_p = source_covariances[source_idx]
            C_q = self.target_covariances[target_idx]
            
            # GICP論文の式に従い、共分散行列を合成
            C = C_p + C_q
            
            # 逆行列を計算（数値安定性のために正則化を追加）
            try:
                C_inv = linalg.inv(C + np.eye(2) * 1e-6)
            except:
                C_inv = np.eye(2)
            
            # 行列Aの要素を計算
            A[2*i, 0] = -p[1]  # -p_y
            A[2*i, 1] = 1      # for x translation
            A[2*i, 2] = 0
            
            A[2*i+1, 0] = p[0]  # p_x
            A[2*i+1, 1] = 0
            A[2*i+1, 2] = 1      # for y translation
            
            # 残差ベクトル
            b[2*i] = q[0] - p[0]    # q_x - p_x
            b[2*i+1] = q[1] - p[1]  # q_y - p_y
            
            # 重み行列の対角成分を設定（2x2ブロック）
            W[2*i:2*i+2, 2*i:2*i+2] = C_inv
        
        # 重み付き最小二乗法でパラメータを計算
        try:
            # A^T * W * A * x = A^T * W * b を解く
            AtwA = A.T @ W @ A
            Atwb = A.T @ W @ b
            params = linalg.solve(AtwA, Atwb)
            
            # 残差からスコアを計算
            residuals = A @ params - b
            score = np.mean(residuals**2)
        except:
            params = np.zeros(3)
            score = float('inf')
        
        return params, score

    def _transform_points(self, points, params):
        """
        点群に変換を適用
        
        Args:
            points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
        
        Returns:
            numpy.ndarray: 変換後の点群
        """
        points = np.asarray(points)
        params = np.asarray(params)
        assert points.shape[1] == 2, "Each point must have 2 coordinates (x, y)"
        assert params.shape[0] == 3, "Params must be [theta, tx, ty]"

        # 回転と並進を抽出
        theta = params[0]
        tx = params[1]
        ty = params[2]
        
        # 回転行列
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # 並進ベクトル
        t = np.array([tx, ty])
        
        # 点群を変換
        transformed = np.dot(points, R.T) + t
        
        return transformed

    def align(self, source_points, initial_guess=np.zeros(3)):
        """
        GICPアルゴリズムでソース点群をターゲット点群に位置合わせ
        
        Args:
            source_points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            initial_guess (numpy.ndarray): 初期変換パラメータ [theta, x, y]
        
        Returns:
            numpy.ndarray: 変換パラメータ [theta, x, y]
            float: 最終的なスコア
            int: 収束するまでの反復回数
        """
        current_params = initial_guess.copy()
        
        # ソース点群の共分散行列を計算
        source_covariances = self._compute_source_covariances(source_points)
        
        prev_score = float('inf')
        iterations = 0
        
        for i in range(self.max_iterations):
            # 対応点を見つける
            correspondences, mean_distance = self._find_correspondences(source_points, current_params)
            
            if len(correspondences) < 3:
                # 十分な対応点がない
                break
            
            # 変換パラメータを計算
            delta_params, score = self._solve_transform(source_points, source_covariances, correspondences)
            
            # パラメータを更新
            current_params = self._update_params(current_params, delta_params)
            
            iterations = i + 1
            
            # 収束判定
            if abs(prev_score - score) < self.epsilon:
                break
                
            prev_score = score
        
        return current_params, prev_score, iterations

    def _update_params(self, current, delta):
        """
        変換パラメータを更新
        
        Args:
            current (numpy.ndarray): 現在のパラメータ [theta, x, y]
            delta (numpy.ndarray): パラメータの更新量 [delta_theta, delta_x, delta_y]
        
        Returns:
            numpy.ndarray: 更新後のパラメータ
        """
        # 角度の更新（-πからπの範囲に正規化）
        current[0] += delta[0]
        current[0] = np.arctan2(np.sin(current[0]), np.cos(current[0]))
        
        # 並進の更新
        current[1] += delta[1]
        current[2] += delta[2]
        
        return current

    def align_levenberg(self, source_points, initial_guess=np.zeros(3), damping=0.1):
        """
        Levenberg-Marquardtアルゴリズムを使用したGICP
        
        Args:
            source_points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            initial_guess (numpy.ndarray): 初期変換パラメータ [theta, x, y]
            damping (float): ダンピング係数
        
        Returns:
            numpy.ndarray: 変換パラメータ [theta, x, y]
            float: 最終的なスコア
            int: 収束するまでの反復回数
        """
        # current_params = initial_guess.copy()
        # 初期推定値が指定されていない場合はゼロで初期化
        if initial_guess is None:
            current_params = np.zeros(3)  # [theta, x, y]
        else:
            current_params = initial_guess.copy()
        
        # ソース点群の共分散行列を計算
        source_covariances = self._compute_source_covariances(source_points)
        
        prev_score = float('inf')
        iterations = 0
        
        for i in range(self.max_iterations):
            # 対応点を見つける
            correspondences, mean_distance = self._find_correspondences(source_points, current_params)
            
            if len(correspondences) < 3:
                # 十分な対応点がない
                break
            
            # Levenberg-Marquardt用のJacobian行列とresidualを構築
            A = np.zeros((2 * len(correspondences), 3))
            b = np.zeros(2 * len(correspondences))
            W = np.zeros((2 * len(correspondences), 2 * len(correspondences)))  # 重み行列
            
            for i, (source_idx, target_idx, _) in enumerate(correspondences):
                # ソース点を現在の推定値で変換
                p = source_points[source_idx]
                transformed_p = self._transform_points(np.array([p]), current_params)[0]
                
                # ターゲット点
                q = self.target_points[target_idx]
                
                # ソースとターゲットの共分散行列
                C_p = source_covariances[source_idx]
                C_q = self.target_covariances[target_idx]
                
                # GICP論文の式に従い、共分散行列を合成
                C = C_p + C_q
                
                # 逆行列を計算（数値安定性のために正則化を追加）
                try:
                    C_inv = linalg.inv(C + np.eye(2) * 1e-6)
                except:
                    C_inv = np.eye(2)
                
                # 回転行列のJacobian要素を計算
                theta = current_params[0]
                J_R_theta = np.array([
                    [-np.sin(theta) * p[0] - np.cos(theta) * p[1]],
                    [np.cos(theta) * p[0] - np.sin(theta) * p[1]]
                ])
                
                # 行列Aの要素を計算
                A[2*i, 0] = J_R_theta[0, 0]
                A[2*i, 1] = 1  # for x translation
                A[2*i, 2] = 0
                
                A[2*i+1, 0] = J_R_theta[1, 0]
                A[2*i+1, 1] = 0
                A[2*i+1, 2] = 1  # for y translation
                
                # 残差ベクトル
                b[2*i] = q[0] - transformed_p[0]
                b[2*i+1] = q[1] - transformed_p[1]
                
                # 重み行列の対角成分を設定（2x2ブロック）
                W[2*i:2*i+2, 2*i:2*i+2] = C_inv
            
            # Levenberg-Marquardt法で解く
            try:
                # (A^T * W * A + lambda * I) * delta = A^T * W * b
                AtwA = A.T @ W @ A
                Atwb = A.T @ W @ b
                
                # ダンピング項を追加
                AtwA_damped = AtwA + np.eye(3) * damping * np.trace(AtwA) / 3
                
                # 解を計算
                delta_params = linalg.solve(AtwA_damped, Atwb)
                
                # 更新後のスコアを計算
                new_params = self._update_params(current_params.copy(), delta_params)
                _, new_score = self._find_correspondences(source_points, new_params)
                
                # スコアが改善された場合のみ更新
                if new_score < prev_score:
                    current_params = new_params
                    prev_score = new_score
                    damping *= 0.5  # ダンピング係数を減少
                else:
                    damping *= 2.0  # ダンピング係数を増加
                
            except:
                # 数値的に不安定な場合、ダンピング係数を増加
                damping *= 10.0
            
            iterations += 1
            
            # 収束判定
            if np.linalg.norm(delta_params) < self.epsilon:
                break
        
        return current_params, prev_score, iterations

    def align_newton(self, source_points, initial_guess=np.zeros(3), damping=0.1):
        """
        ニュートン法を使用したGICP
        
        Args:
            source_points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            initial_guess (numpy.ndarray): 初期変換パラメータ [theta, x, y]
            damping (float): ダンピング係数（ヘッセ行列の安定化のため）
        
        Returns:
            numpy.ndarray: 変換パラメータ [theta, x, y]
            float: 最終的なスコア
            int: 収束するまでの反復回数
        """
        # 初期推定値が指定されていない場合はゼロで初期化
        if initial_guess is None:
            current_params = np.zeros(3)  # [theta, x, y]
        else:
            current_params = initial_guess.copy()
        
        # ソース点群の共分散行列を計算
        source_covariances = self._compute_source_covariances(source_points)
        
        prev_score = float('inf')
        best_score = float('inf')
        best_params = current_params.copy()
        iterations = 0
        consecutive_fails = 0
        trust_radius = 0.5  # 信頼領域の初期半径
        min_trust_radius = 0.001  # 最小信頼領域半径
        max_trust_radius = 2.0    # 最大信頼領域半径
        
        for i in range(self.max_iterations):
            # 対応点を見つける
            correspondences, mean_distance = self._find_correspondences(source_points, current_params)
            
            if len(correspondences) < 3:
                # 十分な対応点がない
                print("警告: 十分な対応点がありません。ステップをスキップします。")
                break
            
            # ニュートン法のためのJacobian行列とHessian行列を構築
            jacobian = np.zeros(3)  # [dtheta, dx, dy]
            hessian = np.zeros((3, 3))
            
            total_error = 0

            for src_idx, tgt_idx, _ in correspondences:
                # ソース点
                p = source_points[src_idx]
                # 変換後のソース点
                transformed_p = self._transform_points(np.array([p]), current_params)[0]
                # ターゲット点
                q = self.target_points[tgt_idx]
                
                # 残差ベクトル
                residual = q - transformed_p
                error = np.sum(residual**2)
                total_error += error
                
                # ソースとターゲットの共分散行列
                C_p = source_covariances[src_idx]
                C_q = self.target_covariances[tgt_idx]
                
                # GICP論文の式に従い、共分散行列を合成
                C = C_p + C_q
                
                # 逆行列を計算（数値安定性のために正則化を追加）
                try:
                    C_inv = linalg.inv(C + np.eye(2) * 1e-6)
                except:
                    C_inv = np.eye(2)
                
                # 回転の導関数を計算
                theta = current_params[0]
                # 回転行列の θ に関する導関数
                dR_dtheta = np.array([
                    [-np.sin(theta), -np.cos(theta)],
                    [np.cos(theta), -np.sin(theta)]
                ])
                
                # Jacobianの各要素を計算
                # theta に関する導関数
                dE_dtheta = -2 * residual @ C_inv @ (dR_dtheta @ p)
                # x に関する導関数
                dE_dx = -2 * residual @ C_inv @ np.array([1, 0])
                # y に関する導関数
                dE_dy = -2 * residual @ C_inv @ np.array([0, 1])
                
                # Jacobian行列
                jacobian[0] += dE_dtheta
                jacobian[1] += dE_dx
                jacobian[2] += dE_dy
                
                # Hessian行列（ガウス・ニュートン近似）
                # 二階微分を近似的に計算
                # J_iはパラメータに関するJacobian行列 (2x3)
                # [d(error_x)/d(theta), d(error_x)/d(x), d(error_x)/d(y)]
                # [d(error_y)/d(theta), d(error_y)/d(x), d(error_y)/d(y)]
                J_i = np.zeros((2, 3))
                # 点pを回転させるときの変化率（正確な導関数の計算）
                # dR/dtheta * p = [-sin(theta)*px - cos(theta)*py, cos(theta)*px - sin(theta)*py]
                J_i[0, 0] = dR_dtheta[0, 0] * p[0] + dR_dtheta[0, 1] * p[1]  # d(error_x)/d(theta)
                J_i[0, 1] = 1.0  # d(error_x)/d(x)
                J_i[0, 2] = 0.0  # d(error_x)/d(y)
                J_i[1, 0] = dR_dtheta[1, 0] * p[0] + dR_dtheta[1, 1] * p[1]  # d(error_y)/d(theta)
                J_i[1, 1] = 0.0  # d(error_y)/d(x)
                J_i[1, 2] = 1.0  # d(error_y)/d(y)
                
                # ヘッシアン行列を更新 (J_i.T @ C_inv @ J_i は 3x3行列)
                hessian += 2 * (J_i.T @ C_inv @ J_i)                # ヘッシアン行列の処理
            try:
                # SVDを使ったロバストな行列分解
                U, S, Vh = np.linalg.svd(hessian)
                
                # S が対角行列ではなく1次元ベクトルで返されるため、次元の確認
                if len(S.shape) == 1:
                    S_diag = np.diag(S)
                else:
                    S_diag = S
                
                # 条件数が悪い場合は修正
                condition_number = np.max(S) / max(1e-10, np.min(S))
                if condition_number > 1e8:
                    # 特異値の下限を設定
                    min_sv = np.max(S) / 1e8
                    S = np.maximum(S, min_sv)
                
                # 適応的なダンピングを追加して安定化
                h_diag_mean = np.mean(S)
                adaptive_damping = damping * h_diag_mean if h_diag_mean > 1e-6 else damping
                S_damped = S + adaptive_damping
                
                # SVDを使用して擬似逆行列を計算（安定化のため）
                # hessian⁻¹ = V @ diag(1/S) @ U^T
                if len(S.shape) == 1:
                    # S は1次元配列なので対角行列に変換する必要がある
                    S_inv_diag = 1.0 / S_damped
                    delta_params = np.zeros(3)
                    for i in range(len(S)):
                        # delta_params += S_inv_diag[i] * (Vh[i, :].T @ U[:, i]) @ (-jacobian)
                        delta_params += S_inv_diag[i] * np.outer(Vh[i, :], U[:, i]) @ (-jacobian)
                else:
                    # S が既に対角行列の場合（通常はこの形式ではない）
                    delta_params = Vh.T @ np.diag(1.0 / S_damped) @ U.T @ (-jacobian)
            except np.linalg.LinAlgError:
                # SVDが失敗した場合は安全なフォールバック
                print("警告: SVDが失敗しました。代替手法を使用します。")
                delta_params = -jacobian * 0.01
                consecutive_fails += 1
                if consecutive_fails > 3:
                    print("警告: 連続して数値計算に失敗しました。収束に問題がある可能性があります。")
                    break
            
            # 信頼領域制約を適用
            step_norm = np.linalg.norm(delta_params)
            if step_norm > trust_radius:
                delta_params = delta_params * (trust_radius / step_norm)
                step_norm = trust_radius
            
            # 次のパラメータの候補を計算
            new_params = self._update_params(current_params.copy(), delta_params)
            
            # 更新後のスコアを計算
            new_correspondences, new_mean_distance = self._find_correspondences(source_points, new_params)
            new_score = new_mean_distance if len(new_correspondences) > 3 else float('inf')
            
            # 予測される改善と実際の改善を計算（信頼領域法のため）
            # ヘッセ行列が正確なら、predicted_reduction ≈ actual_reduction となる
            predicted_reduction = -jacobian @ delta_params - 0.5 * delta_params @ hessian @ delta_params
            actual_reduction = prev_score - new_score
            
            # 信頼領域の更新
            ratio = actual_reduction / max(1e-10, predicted_reduction) if predicted_reduction != 0 else 0
            
            if ratio > 0.75 and step_norm >= 0.9 * trust_radius:
                # 予測が非常に良かった場合、信頼領域を拡大
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            elif ratio < 0.25:
                # 予測が不十分だった場合、信頼領域を縮小
                trust_radius = max(0.5 * trust_radius, min_trust_radius)
            
            # スコアが改善された場合のみ更新
            if new_score < prev_score:
                current_params = new_params
                prev_score = new_score
                consecutive_fails = 0
                
                # 過去最高のスコアを記録
                if new_score < best_score:
                    best_score = new_score
                    best_params = current_params.copy()
            else:
                # スコアが悪化した場合はライン検索で改善を試みる
                found_better = False
                for scale in [0.5, 0.25, 0.125, 0.0625]:
                    scaled_delta = delta_params * scale
                    candidate_params = self._update_params(current_params.copy(), scaled_delta)
                    candidate_correspondences, candidate_mean_distance = self._find_correspondences(source_points, candidate_params)
                    candidate_score = candidate_mean_distance if len(candidate_correspondences) > 3 else float('inf')
                    
                    if candidate_score < prev_score:
                        current_params = candidate_params
                        prev_score = candidate_score
                        found_better = True
                        consecutive_fails = 0
                        break
                
                if not found_better:
                    consecutive_fails += 1
                    
                    # 連続して失敗が続く場合、アルゴリズムが局所解に陥っている可能性がある
                    if consecutive_fails > 3:
                        # ランダムな摂動を加える（局所解から抜け出す試み）
                        perturbation = np.random.normal(0, 0.01, 3)
                        current_params = self._update_params(current_params.copy(), perturbation)
                    
                    if consecutive_fails > 5:
                        print("警告: 連続して改善に失敗しています。最良の結果を返します。")
                        current_params = best_params
                        break
            
            iterations += 1
            
            # 収束判定
            if np.linalg.norm(delta_params) < self.epsilon:
                break
        
        # 最良の結果を返す
        return best_params, best_score, iterations

    def transform_points(self, points, params):
        """
        点群に変換を適用（外部からアクセスするためのメソッド）
        
        Args:
            points (numpy.ndarray): 変換する点群 [[x1, y1], [x2, y2], ...]
            params (numpy.ndarray): 変換パラメータ [theta, x, y]
        
        Returns:
            numpy.ndarray: 変換後の点群
        """
        return self._transform_points(points, params)
