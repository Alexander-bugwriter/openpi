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
        self.class_id_to_name = {}

        if spatial_bounds:
            print(f"[PointMap] 启用空间过滤:")
            print(f"  X范围: {spatial_bounds.get('x', 'None')}")
            print(f"  Y范围: {spatial_bounds.get('y', 'None')}")
            print(f"  Z范围: {spatial_bounds.get('z', 'None')}")

    def reset(self):
        """清空缓存，准备处理新的episode"""
        self.frames = []
        self.class_id_to_name={}
        # cam_names保留，避免重复检测
        print(f"[PointMap] 缓存已清空，准备新episode")

    # def _apply_spatial_filter(self, points: np.ndarray, colors: np.ndarray, labels: np.ndarray):
    #     """应用空间过滤"""
    #     if self.spatial_bounds is None:
    #         return points, colors, labels
    #
    #     # 创建mask
    #     mask = np.ones(len(points), dtype=bool)
    #
    #     if 'x' in self.spatial_bounds:
    #         x_min, x_max = self.spatial_bounds['x']
    #         mask &= (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    #
    #     if 'y' in self.spatial_bounds:
    #         y_min, y_max = self.spatial_bounds['y']
    #         mask &= (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    #
    #     if 'z' in self.spatial_bounds:
    #         z_min, z_max = self.spatial_bounds['z']
    #         mask &= (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    #
    #     filtered_points = points[mask]
    #     filtered_colors = colors[mask]
    #     filtered_labels = labels[mask]
    #
    #     # if len(points) > 0:
    #     #     filter_rate = 100 * (1 - len(filtered_points) / len(points))
    #     #     # if filter_rate > 50:  # 只在过滤超过50%时打印
    #     #     #     print(f"[PointMap] 空间过滤: {len(points)} → {len(filtered_points)} "
    #     #     #           f"(过滤掉 {filter_rate:.1f}%)")
    #
    #     return filtered_points, filtered_colors, filtered_labels

    # def capture_frame(self, obs: dict, env, timestamp: float, step_idx: int):
    #     """捕获一帧4视角合并点云"""
    #     if self.cam_names is None:
    #         self.cam_names = [k.replace('_depth', '') for k in obs.keys() if k.endswith('_depth')]
    #         print(f"[PointMap] 检测到 {len(self.cam_names)} 个深度相机: {self.cam_names}")
    #
    #     all_points = []
    #     all_colors = []
    #     all_labels = []  # 🔥 新增
    #
    #     for cam_name in self.cam_names:
    #         depth = obs[f'{cam_name}_depth'].squeeze()
    #         rgb = obs[f'{cam_name}_image']
    #         seg = obs[f'{cam_name}_segmentation_instance']  # 🔥 直接获取分割图
    #
    #         # points, colors = self._depth_to_pointcloud(depth, rgb, cam_name, env)
    #         points, colors, labels = self._depth_to_pointcloud(depth, rgb, seg, cam_name, env)
    #
    #         # 应用空间过滤
    #         # points, colors = self._apply_spatial_filter(points, colors)
    #         points, colors, labels = self._apply_spatial_filter(points, colors, labels)
    #
    #         all_points.append(points)
    #         all_colors.append(colors)
    #         all_labels.append(labels)
    #
    #     merged_points = np.vstack(all_points)
    #     merged_colors = np.vstack(all_colors)
    #     merged_labels = np.concatenate(all_labels)
    #     # ====== 新增：坐标系转换 ======
    #     # MuJoCo (x, y, z) → Three.js (x, y, z)
    #     # MuJoCo: x=右, y=前, z=上
    #     # Three.js: x=右, y=上, z=前
    #     # =============================
    #
    #     # 下采样到max_points
    #     if merged_points.shape[0] > self.max_points:
    #         indices = np.random.choice(merged_points.shape[0], self.max_points, replace=False)
    #         merged_points = merged_points[indices]
    #         merged_colors = merged_colors[indices]
    #         merged_labels = merged_labels[indices]
    #
    #     self.frames.append({
    #         'timestamp': float(timestamp),
    #         'step_idx': int(step_idx),
    #         'points': merged_points.tolist(),
    #         'colors': (merged_colors * 255).astype(np.uint8).tolist(),
    #         'labels': merged_labels.tolist(),
    #     })
    #
    #     # print(f"[PointMap] 帧 {step_idx}: {merged_points.shape[0]} 点")
    #     return merged_points.shape[0]
    def _build_class_id_mapping(self, env, first_frame_obs):
        """根据get_segmentation_instances的逻辑构建映射"""

        if hasattr(env, 'segmentation_id_mapping'):
            print("\n=== 构建ID映射（基于官方逻辑）===")

            # 关键：分割图ID = segmentation_id_mapping的key + 1
            for seg_id, instance_name in env.segmentation_id_mapping.items():
                pixel_id = seg_id + 1  # ← 这是分割图中的实际ID

                # 提取类名
                if instance_name in ["OnTheGroundPanda0", "NullMount0"]:
                    class_name = "robot"
                elif "_" in instance_name and instance_name.split("_")[-1].isdigit():
                    class_name = "_".join(instance_name.split("_")[:-1])
                else:
                    class_name = instance_name

                self.class_id_to_name[pixel_id] = class_name
                print(f"  ID {pixel_id} -> {class_name} (from {instance_name})")

            # 添加机器人ID（如果有）
            if hasattr(env, 'segmentation_robot_id') and env.segmentation_robot_id is not None:
                robot_pixel_id = env.segmentation_robot_id + 1
                self.class_id_to_name[robot_pixel_id] = "robot"
                print(f"  ID {robot_pixel_id} -> robot (robot_id)")

            if env.segmentation_id_mapping:
                max_seg_id = max(env.segmentation_id_mapping.keys())
                gripper_pixel_id = max_seg_id + 2  # +1映射到分割图，再+1是gripper
                self.class_id_to_name[gripper_pixel_id] = "gripper"
                print(f"  ID {gripper_pixel_id} -> gripper (固定，最大ID+2)")

            print(f"[PointMap] 获取映射: {len(self.class_id_to_name)} 个")

        # 固定ID 0为环境
        if 0 not in self.class_id_to_name:
            self.class_id_to_name[0] = "environment"
            print(f"  ID 0 -> environment (固定)")

        # 检查未映射的ID
        visible_ids = set()
        for cam_name in self.cam_names:
            seg = first_frame_obs[f'{cam_name}_segmentation_instance']
            visible_ids.update(np.unique(seg).tolist())

        unmapped_ids = visible_ids - set(self.class_id_to_name.keys())
        if unmapped_ids:
            print(f"[PointMap] 未映射的ID: {sorted(unmapped_ids)}")
            for uid in unmapped_ids:
                self.class_id_to_name[int(uid)] = f"unknown_{uid}"


    def capture_frame(self, obs: dict, env, timestamp: float, step_idx: int):
        """捕获一帧4视角合并点云"""
        if self.cam_names is None:
            self.cam_names = [k.replace('_depth', '') for k in obs.keys() if k.endswith('_depth')]
            print(f"[PointMap] 检测到 {len(self.cam_names)} 个深度相机: {self.cam_names}")
        if not self.class_id_to_name:
            try:
                self._build_class_id_mapping(env, obs)
                # print("成功保存class id")
                # print(f"[PointMap] 总映射数: {len(self.class_id_to_name)}")
                #
                # # ← 添加这个调试
                # print(f"[PointMap] 所有keys: {list(self.class_id_to_name.keys())}")
                # print(f"[PointMap] keys类型: {[type(k) for k in self.class_id_to_name.keys()]}")
                #
                # print("\n========== 完整ID映射表 ==========")
                # for bid in sorted(self.class_id_to_name.keys()):
                #     # 检查类型
                #     print(f"  ID {int(bid):3d} -> {self.class_id_to_name[bid]}")
                # print("==================================\n")
            except Exception as e:
                print(f"保存class id失败: {e}")
                import traceback
                traceback.print_exc()

        all_points = []
        all_colors_rgb = []  # 🔥 RGB颜色
        all_colors_seg = []  # 🔥 分割图颜色
        all_labels = []

        for cam_name in self.cam_names:
            depth = obs[f'{cam_name}_depth'].squeeze()
            rgb = obs[f'{cam_name}_image']
            seg = obs[f'{cam_name}_segmentation_instance']

            # 🔥 获取分割图的可视化颜色
            seg_vis = self._visualize_segmentation(seg, env)

            points, colors_rgb, colors_seg, labels = self._depth_to_pointcloud(
                depth, rgb, seg, seg_vis, cam_name, env
            )

            # 空间过滤
            points, colors_rgb, colors_seg, labels = self._apply_spatial_filter(
                points, colors_rgb, colors_seg, labels
            )

            all_points.append(points)
            all_colors_rgb.append(colors_rgb)
            all_colors_seg.append(colors_seg)
            all_labels.append(labels)

        merged_points = np.vstack(all_points)
        merged_colors_rgb = np.vstack(all_colors_rgb)
        merged_colors_seg = np.vstack(all_colors_seg)
        merged_labels = np.concatenate(all_labels)

        # 下采样
        if merged_points.shape[0] > self.max_points:
            indices = np.random.choice(merged_points.shape[0], self.max_points, replace=False)
            merged_points = merged_points[indices]
            merged_colors_rgb = merged_colors_rgb[indices]
            merged_colors_seg = merged_colors_seg[indices]
            merged_labels = merged_labels[indices]

        self.frames.append({
            'timestamp': float(timestamp),
            'step_idx': int(step_idx),
            'points': merged_points.tolist(),
            'colors_rgb': (merged_colors_rgb * 255).astype(np.uint8).tolist(),  # 🔥 RGB颜色
            'colors_seg': (merged_colors_seg * 255).astype(np.uint8).tolist(),  # 🔥 分割图颜色
            'labels': merged_labels.tolist(),
        })

        return merged_points.shape[0]

    # def _depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray, seg: np.ndarray, cam_name: str, env):
    #     """使用robosuite工具进行深度反投影"""
    #     sim = env.sim
    #     H, W = depth.shape
    #     # 检查图像约定（OpenGL是flipped，需要上下翻转）
    #     is_opengl_flipped = (macros.IMAGE_CONVENTION == "opengl")
    #
    #     # 归一化深度 -> 真实距离
    #     model = sim.model
    #     extent = model.stat.extent
    #     near = model.vis.map.znear * extent
    #     far = model.vis.map.zfar * extent
    #     depth_real = near / (1.0 - depth * (1.0 - near / far))
    #
    #     # 如果是OpenGL约定，上下翻转深度图和RGB
    #     if is_opengl_flipped:
    #         depth_real = np.flipud(depth_real)
    #         rgb = np.flipud(rgb)
    #         seg = np.flipud(seg)  # 🔥 分割图也要翻转
    #
    #     # 使用robosuite获取内参和外参
    #     K = get_camera_intrinsic_matrix(sim, cam_name, H, W)
    #     fx, fy = K[0, 0], K[1, 1]
    #     cx, cy = K[0, 2], K[1, 2]
    #
    #     # robosuite的get_camera_extrinsic_matrix返回的是cam_to_world
    #     cam_to_world = get_camera_extrinsic_matrix(sim, cam_name)
    #
    #     # 像素网格
    #     u, v = np.meshgrid(np.arange(W), np.arange(H))
    #     u = u.flatten()
    #     v = v.flatten()
    #     z = depth_real.flatten()
    #     seg_flat = seg.flatten()
    #
    #     # 深度过滤
    #     valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)
    #     u = u[valid]
    #     v = v[valid]
    #     z = z[valid]
    #     colors = rgb.reshape(-1, 3)[valid]
    #     labels = seg_flat[valid]
    #
    #     # 像素坐标 -> 相机坐标（OpenCV约定：+Z前方）
    #     x_cam = (u - cx) * z / fx
    #     y_cam = (v - cy) * z / fy
    #     z_cam = z
    #     points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones(len(z))], axis=1)
    #
    #     # 相机坐标 -> 世界坐标
    #     points_world_homo = (cam_to_world @ points_cam_homo.T).T
    #     points_world = points_world_homo[:, :3]
    #
    #     return points_world, colors, labels
    def _depth_to_pointcloud(self, depth: np.ndarray, rgb: np.ndarray,
                             seg: np.ndarray, seg_vis: np.ndarray, cam_name: str, env):
        """深度反投影，返回RGB颜色和分割颜色"""
        sim = env.sim
        H, W = depth.shape

        if len(seg.shape) == 3:
            seg = seg.squeeze()

        is_opengl_flipped = (macros.IMAGE_CONVENTION == "opengl")

        # 归一化深度
        model = sim.model
        extent = model.stat.extent
        near = model.vis.map.znear * extent
        far = model.vis.map.zfar * extent
        depth_real = near / (1.0 - depth * (1.0 - near / far))

        # OpenGL翻转
        if is_opengl_flipped:
            depth_real = np.flipud(depth_real)
            rgb = np.flipud(rgb)
            seg = np.flipud(seg)
            seg_vis = np.flipud(seg_vis)  # 🔥 分割图可视化也要翻转

        # 相机参数
        K = get_camera_intrinsic_matrix(sim, cam_name, H, W)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cam_to_world = get_camera_extrinsic_matrix(sim, cam_name)

        # 像素网格
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        u = u.flatten()
        v = v.flatten()
        z = depth_real.flatten()
        seg_flat = seg.flatten()

        # 深度过滤
        valid = (z > 0.1) & (z < 5.0) & np.isfinite(z)
        u = u[valid]
        v = v[valid]
        z = z[valid]
        colors_rgb = rgb.reshape(-1, 3)[valid] / 255.0  # 🔥 归一化到0-1
        colors_seg = seg_vis.reshape(-1, 3)[valid]  # 🔥 分割颜色，已经是0-1
        labels = seg_flat[valid]

        # 反投影
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones(len(z))], axis=1)

        points_world_homo = (cam_to_world @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]

        return points_world, colors_rgb, colors_seg, labels  # 🔥 返回4个值

    def _apply_spatial_filter(self, points: np.ndarray, colors_rgb: np.ndarray,
                              colors_seg: np.ndarray, labels: np.ndarray):
        """空间过滤"""
        if self.spatial_bounds is None:
            return points, colors_rgb, colors_seg, labels

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

        return points[mask], colors_rgb[mask], colors_seg[mask], labels[mask]

    def _visualize_segmentation(self, seg: np.ndarray, env):
        """将分割图转换为彩色可视化图像（返回float 0-1）"""
        if len(seg.shape) == 3:
            seg = seg.squeeze()

        H, W = seg.shape
        vis_img = np.zeros((H, W, 3), dtype=np.float32)  # 🔥 float32，范围0-1

        unique_ids = np.unique(seg)
        for uid in unique_ids:
            mask = (seg == uid)
            hue = (uid * 137.508) % 360
            r, g, b = self._hsl_to_rgb(hue / 360, 0.8, 0.6)
            vis_img[mask] = [r, g, b]

        return vis_img

    def _hsl_to_rgb(self, h, s, l):
        """HSL转RGB（返回0-1范围）"""
        if s == 0:
            r = g = b = l
        else:
            def hue2rgb(p, q, t):
                if t < 0: t += 1
                if t > 1: t -= 1
                if t < 1 / 6: return p + (q - p) * 6 * t
                if t < 1 / 2: return q
                if t < 2 / 3: return p + (q - p) * (2 / 3 - t) * 6
                return p

            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue2rgb(p, q, h + 1 / 3)
            g = hue2rgb(p, q, h)
            b = hue2rgb(p, q, h - 1 / 3)

        return r, g, b

    def save_frames_as_json(self, output_dir: str, episode_id: int, env=None):
        """保存点云帧序列为JSON"""
        os.makedirs(output_dir, exist_ok=True)
        # 🔥 如果传入了env，就保存site映射
        site_mapping = {}
        if env is not None:
            for site_id in range(env.sim.model.nsite):
                name = env.sim.model.site_id2name(site_id)
                if name:
                    site_mapping[site_id] = name

        metadata = {
            'episode_id': episode_id,
            'total_frames': len(self.frames),
            'max_points_per_frame': self.max_points,
            'cameras': self.cam_names,
            # 'site_mapping': site_mapping  # 🔥 直接保存在metadata里
            'class_id_to_name': self.class_id_to_name,  # 而不是 'site_mapping'
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