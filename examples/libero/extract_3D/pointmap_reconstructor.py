import numpy as np
import json
import os
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
import robosuite.macros as macros


class PointMapReconstructor:
    """LIBERO 4视角 RGB-D 点云重建器 - 带空间过滤"""

    def __init__(self, max_points: int = 30000,
                 spatial_bounds: dict = None):
        """
        Args:
            max_points: 每帧最大点数
            spatial_bounds: 空间过滤范围，格式：
                {
                    'x': (min, max),  # 例如 (-0.8, 0.8)
                    'y': (min, max),  # 例如 (-0.8, 0.8)
                    'z': (min, max)   # 例如 (0.6, 1.5)
                }
                如果为None，则不过滤
        """
        self.max_points = max_points
        self.spatial_bounds = spatial_bounds
        self.frames = []
        self.cam_names = None

        if spatial_bounds:
            print(f"[PointMap] 启用空间过滤:")
            print(f"  X范围: {spatial_bounds.get('x', 'None')}")
            print(f"  Y范围: {spatial_bounds.get('y', 'None')}")
            print(f"  Z范围: {spatial_bounds.get('z', 'None')}")

    def reset(self):
        """清空缓存，准备处理新的episode"""
        self.frames = []
        # cam_names保留，避免重复检测
        print(f"[PointMap] 缓存已清空，准备新episode")

    def _apply_spatial_filter(self, points: np.ndarray, colors: np.ndarray):
        """应用空间过滤"""
        if self.spatial_bounds is None:
            return points, colors

        # 创建mask
        mask = np.ones(len(points), dtype=bool)

        if 'x' in self.spatial_bounds:
            x_min, x_max = self.spatial_bounds['x']
            mask &= (points[:, 0] >= x_min) & (points[:, 0] <= x_max)

        if 'y' in self.spatial_bounds:
            y_min, y_max = self.spatial_bounds['y']
            mask &= (points[:, 1] >= y_min) & (points[:, 1] <= y_max)

        if 'z' in self.spatial_bounds:
            z_min, z_max = self.spatial_bounds['z']
            mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)

        filtered_points = points[mask]
        filtered_colors = colors[mask]

        # if len(points) > 0:
        #     filter_rate = 100 * (1 - len(filtered_points) / len(points))
        #     # if filter_rate > 50:  # 只在过滤超过50%时打印
        #     #     print(f"[PointMap] 空间过滤: {len(points)} → {len(filtered_points)} "
        #     #           f"(过滤掉 {filter_rate:.1f}%)")

        return filtered_points, filtered_colors

    def capture_frame(self, obs: dict, env, timestamp: float, step_idx: int):
        """捕获一帧4视角合并点云"""
        if self.cam_names is None:
            self.cam_names = [k.replace('_depth', '') for k in obs.keys() if k.endswith('_depth')]
            print(f"[PointMap] 检测到 {len(self.cam_names)} 个深度相机: {self.cam_names}")

        all_points = []
        all_colors = []

        for cam_name in self.cam_names:
            depth = obs[f'{cam_name}_depth'].squeeze()
            rgb = obs[f'{cam_name}_image']

            points, colors = self._depth_to_pointcloud(depth, rgb, cam_name, env)

            # 应用空间过滤
            points, colors = self._apply_spatial_filter(points, colors)

            all_points.append(points)
            all_colors.append(colors)

        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)
        # ====== 新增：坐标系转换 ======
        # MuJoCo (x, y, z) → Three.js (x, y, z)
        # MuJoCo: x=右, y=前, z=上
        # Three.js: x=右, y=上, z=前
        # =============================

        # 下采样到max_points
        if merged_points.shape[0] > self.max_points:
            indices = np.random.choice(merged_points.shape[0], self.max_points, replace=False)
            merged_points = merged_points[indices]
            merged_colors = merged_colors[indices]

        self.frames.append({
            'timestamp': float(timestamp),
            'step_idx': int(step_idx),
            'points': merged_points.tolist(),
            'colors': (merged_colors * 255).astype(np.uint8).tolist()
        })

        # print(f"[PointMap] 帧 {step_idx}: {merged_points.shape[0]} 点")
        return merged_points.shape[0]

    def _depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray, cam_name: str, env):
        """使用robosuite工具进行深度反投影"""
        sim = env.sim
        H, W = depth.shape

        # 检查图像约定（OpenGL是flipped，需要上下翻转）
        is_opengl_flipped = (macros.IMAGE_CONVENTION == "opengl")

        # 归一化深度 -> 真实距离
        model = sim.model
        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        depth_real = near / (1.0 - depth * (1.0 - near / far))

        # 如果是OpenGL约定，上下翻转深度图和RGB
        if is_opengl_flipped:
            depth_real = np.flipud(depth_real)
            rgb = np.flipud(rgb)

        # 使用robosuite获取内参和外参
        K = get_camera_intrinsic_matrix(sim, cam_name, H, W)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # robosuite的get_camera_extrinsic_matrix返回的是cam_to_world
        cam_to_world = get_camera_extrinsic_matrix(sim, cam_name)

        # 像素网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()
        z = depth_real.flatten()

        # 深度过滤
        valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        colors = rgb.reshape(-1, 3)[valid]

        # 像素坐标 -> 相机坐标（OpenCV约定：+Z前方）
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones(len(z))], axis=1)

        # 相机坐标 -> 世界坐标
        points_world_homo = (cam_to_world @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]

        return points_world, colors

    def save_frames_as_json(self, output_dir: str, episode_id: int):
        """保存点云帧序列为JSON"""
        os.makedirs(output_dir, exist_ok=True)

        metadata = {
            'episode_id': episode_id,
            'total_frames': len(self.frames),
            'max_points_per_frame': self.max_points,
            'cameras': self.cam_names
        }

        meta_path = f"{output_dir}/pointmeta_ep_{episode_id}.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        frames_path = f"{output_dir}/pointframes_ep_{episode_id}.json"
        print(f"[PointMap] 正在保存 {len(self.frames)} 帧到 {frames_path}...")
        with open(frames_path, 'w') as f:
            json.dump(self.frames, f)

        size_mb = os.path.getsize(frames_path) / 1024 / 1024
        print(f"[PointMap] 已保存 -> {frames_path} ({size_mb:.1f} MB)")
        return meta_path, frames_path

    def get_summary(self):
        """返回重建摘要"""
        if not self.frames:
            return {"error": "没有捕获任何帧"}

        point_counts = [len(f['points']) for f in self.frames]
        duration = self.frames[-1]['timestamp'] - self.frames[0]['timestamp']

        return {
            'total_frames': len(self.frames),
            'avg_points_per_frame': int(np.mean(point_counts)),
            'max_points_per_frame': max(point_counts),
            'min_points_per_frame': min(point_counts),
            'duration_seconds': float(duration),
            'cameras': self.cam_names
        }