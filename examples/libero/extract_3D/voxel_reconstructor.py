import numpy as np
import json
import os
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
import robosuite.macros as macros


class VoxelReconstructor:
    """LIBERO 4视角 RGB-D 体素重建器 - 基于点云体素化"""

    def __init__(self,
                 voxel_grid_size: tuple = (64, 64, 64),
                 spatial_bounds: dict = None,
                 min_points_per_voxel: int = 10):
        """
        Args:
            voxel_grid_size: 体素网格分辨率 (nx, ny, nz)
            spatial_bounds: 空间过滤范围，格式：
                {
                    'x': (min, max),
                    'y': (min, max),
                    'z': (min, max)
                }
            min_points_per_voxel: 体素被标记为占用的最小点数阈值
        """
        self.voxel_grid_size = voxel_grid_size
        self.spatial_bounds = spatial_bounds
        self.min_points_per_voxel = min_points_per_voxel
        self.frames = []
        self.cam_names = None
        self.class_id_to_name = {}
        self.step_counter = 0

        # 计算体素大小
        if spatial_bounds:
            self.voxel_size = [
                (spatial_bounds['x'][1] - spatial_bounds['x'][0]) / voxel_grid_size[0],
                (spatial_bounds['y'][1] - spatial_bounds['y'][0]) / voxel_grid_size[1],
                (spatial_bounds['z'][1] - spatial_bounds['z'][0]) / voxel_grid_size[2]
            ]
            # print(f"[Voxel] 网格分辨率: {voxel_grid_size}")
            # print(f"[Voxel] 空间范围: X{spatial_bounds['x']}, Y{spatial_bounds['y']}, Z{spatial_bounds['z']}")
            # print(f"[Voxel] 体素尺寸: {[f'{s:.4f}' for s in self.voxel_size]} m")
            # print(f"[Voxel] 最小点数阈值: {min_points_per_voxel} 点/体素")
        else:
            raise ValueError("VoxelReconstructor需要指定spatial_bounds来计算体素大小")

    def reset(self):
        """清空缓存，准备处理新的episode"""
        self.frames = []
        self.class_id_to_name = {}
        self.step_counter = 0
        # print(f"[Voxel] 缓存已清空，准备新episode")

    # def _build_class_id_mapping(self, env, first_frame_obs):
    #     """构建ID到类别名称的映射"""
    #     if hasattr(env, 'segmentation_id_mapping'):
    #         # print("\n=== 构建ID映射（基于官方逻辑）===")
    #
    #         for seg_id, instance_name in env.segmentation_id_mapping.items():
    #             pixel_id = seg_id + 1
    #
    #             if instance_name in ["OnTheGroundPanda0", "NullMount0"]:
    #                 class_name = "robot"
    #             elif "_" in instance_name and instance_name.split("_")[-1].isdigit():
    #                 class_name = "_".join(instance_name.split("_")[:-1])
    #             else:
    #                 class_name = instance_name
    #
    #             self.class_id_to_name[pixel_id] = class_name
    #             # print(f"  ID {pixel_id} -> {class_name} (from {instance_name})")
    #
    #         if hasattr(env, 'segmentation_robot_id') and env.segmentation_robot_id is not None:
    #             robot_pixel_id = env.segmentation_robot_id + 1
    #             self.class_id_to_name[robot_pixel_id] = "robot"
    #             # print(f"  ID {robot_pixel_id} -> robot (robot_id)")
    #
    #         if env.segmentation_id_mapping:
    #             max_seg_id = max(env.segmentation_id_mapping.keys())
    #             gripper_pixel_id = max_seg_id + 2
    #             self.class_id_to_name[gripper_pixel_id] = "gripper"
    #             # print(f"  ID {gripper_pixel_id} -> gripper (固定，最大ID+2)")
    #
    #         # print(f"[Voxel] 获取映射: {len(self.class_id_to_name)} 个")
    #
    #     if 0 not in self.class_id_to_name:
    #         self.class_id_to_name[0] = "environment"
    #         # print(f"  ID 0 -> environment (固定)")
    #
    #     # 检查未映射的ID
    #     visible_ids = set()
    #     for cam_name in self.cam_names:
    #         seg = first_frame_obs[f'{cam_name}_segmentation_instance']
    #         visible_ids.update(np.unique(seg).tolist())
    #
    #     unmapped_ids = visible_ids - set(self.class_id_to_name.keys())
    #     if unmapped_ids:
    #         print(f"[Voxel] 未映射的ID: {sorted(unmapped_ids)}")
    #         for uid in unmapped_ids:
    #             self.class_id_to_name[int(uid)] = f"unknown_{uid}"
    #
    #     all_instances = list(env.instance_to_id.keys())
    #     print(f"完整实例顺序: {all_instances}")
    #     print(f"segmentation_robot_id: {env.segmentation_robot_id}")
    #     print(f"segmentation_id_mapping: {env.segmentation_id_mapping}")
    def _build_class_id_mapping(self, env, first_frame_obs):
        """构建ID到类别名称的映射 - 修复版"""

        # 1. 先获取分割图中实际出现的所有ID
        visible_ids = set()
        for cam_name in self.cam_names:
            seg = first_frame_obs[f'{cam_name}_segmentation_instance']
            visible_ids.update(np.unique(seg).tolist())

        # 2. 找到最大的ID（这就是gripper）
        non_bg_ids = visible_ids - {0}  # 临时移除背景来找最大值
        if non_bg_ids:
            max_visible_id = max(non_bg_ids)
        else:
            max_visible_id = 0

        # 3. 获取完整的实例列表
        all_instances = list(env.instance_to_id.keys())

        # 4. 找出所有机器人相关实例的索引
        robot_related_indices = []
        for i, instance_name in enumerate(all_instances):
            # if any(keyword in instance_name for keyword in ['Panda', 'Mount']):
            #     robot_related_indices.append(i)
            if 'Panda' in instance_name:
                robot_related_indices.append(i)

        # 5. 确定robot的ID范围：从第一个robot实例到最大ID-1
        if robot_related_indices:
            robot_start_id = min(robot_related_indices) + 1  # +1因为像素ID = 索引+1
            robot_end_id = max_visible_id - 1  # 最大ID是gripper，所以robot到最大ID-1
        else:
            robot_start_id = None
            robot_end_id = None

        # 6. 构建映射
        # self.class_id_to_name = {}

        # 背景
        self.class_id_to_name[0] = "environment"

        # 遍历所有实例，构建物品的映射
        for i, instance_name in enumerate(all_instances):
            pixel_id = i + 1  # 像素ID = 索引 + 1

            # 如果是机器人相关实例，跳过（稍后统一处理）
            # if any(keyword in instance_name for keyword in ['Panda', 'Mount', 'Gripper']):
            #     continue
            if 'Panda' in instance_name:
                continue

            # 物品：去掉末尾的数字后缀
            if "_" in instance_name and instance_name.split("_")[-1].isdigit():
                class_name = "_".join(instance_name.split("_")[:-1])
            else:
                class_name = instance_name

            self.class_id_to_name[pixel_id] = class_name

        # 7. 统一标记robot ID范围（从第一个robot实例到最大ID-1）
        if robot_start_id is not None and robot_end_id is not None:
            for robot_id in range(robot_start_id, robot_end_id + 1):
                if robot_id in visible_ids:  # 只标记实际出现的ID
                    self.class_id_to_name[robot_id] = "robot"

        # 8. 标记gripper（最大的ID）
        self.class_id_to_name[max_visible_id] = "gripper"

        # 9. 检查是否有未映射的ID（用于调试）
        unmapped_ids = visible_ids - set(self.class_id_to_name.keys())
        if unmapped_ids:
            print(f"[Voxel] ⚠️ 仍有未映射的ID: {sorted(unmapped_ids)}")
            # 给未映射的ID分配默认名称
            for uid in unmapped_ids:
                self.class_id_to_name[int(uid)] = f"unknown_{uid}"

        # 打印映射摘要（可选）
        # print(f"[Voxel] ID映射构建完成:")
        # print(f"  - 物品ID范围: 1 ~ {robot_start_id - 1 if robot_start_id else 'N/A'}")
        # print(f"  - Robot ID范围: {robot_start_id} ~ {robot_end_id}")
        # print(f"  - Gripper ID: {max_visible_id}")
        # print(f"  - 总映射数: {len(self.class_id_to_name)}")

    def capture_frame(self, obs: dict, env):
        """捕获一帧并体素化"""
        if self.cam_names is None:
            self.cam_names = [k.replace('_depth', '') for k in obs.keys() if k.endswith('_depth')]
            # print(f"[Voxel] 检测到 {len(self.cam_names)} 个深度相机: {self.cam_names}")

        if not self.class_id_to_name:
            try:
                self._build_class_id_mapping(env, obs)
            except Exception as e:
                print(f"构建ID映射失败: {e}")
                import traceback
                traceback.print_exc()

        # 收集所有相机的点云
        all_points = []
        all_labels = []

        for cam_name in self.cam_names:
            depth = obs[f'{cam_name}_depth'].squeeze()
            seg = obs[f'{cam_name}_segmentation_instance']

            points, labels = self._depth_to_pointcloud(depth, seg, cam_name, env)
            points, labels = self._apply_spatial_filter(points, labels)

            all_points.append(points)
            all_labels.append(labels)

        merged_points = np.vstack(all_points)
        merged_labels = np.concatenate(all_labels)

        # 体素化
        voxel_grid = self._voxelize_points(merged_points, merged_labels)

        self.frames.append({
            'step_idx': int(self.step_counter),
            'voxel_grid': voxel_grid  # shape: (nx, ny, nz), 每个元素是该体素的ID
        })

        occupied_voxels = np.sum(voxel_grid >= 0)
        self.step_counter += 1
        # print(f"[Voxel] 帧 {step_idx}: {merged_points.shape[0]} 点 -> {occupied_voxels} 占用体素")

        return occupied_voxels

    def _depth_to_pointcloud(self, depth: np.ndarray, seg: np.ndarray, cam_name: str, env):
        """深度反投影（只返回点坐标和标签，不需要RGB）"""
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
            seg = np.flipud(seg)

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
        labels = seg_flat[valid]

        # 反投影
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z
        points_cam_homo = np.stack([x_cam, y_cam, z_cam, np.ones(len(z))], axis=1)

        points_world_homo = (cam_to_world @ points_cam_homo.T).T
        points_world = points_world_homo[:, :3]

        return points_world, labels

    def _apply_spatial_filter(self, points: np.ndarray, labels: np.ndarray):
        """空间过滤"""
        if self.spatial_bounds is None:
            return points, labels

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

        return points[mask], labels[mask]

    def _voxelize_points(self, points: np.ndarray, labels: np.ndarray):
        """将点云体素化到网格中，只保留点数足够多的体素"""
        nx, ny, nz = self.voxel_grid_size
        voxel_grid = np.full((nx, ny, nz), -1, dtype=np.int32)  # -1表示空体素

        # 计算每个点所属的体素索引
        x_min, x_max = self.spatial_bounds['x']
        y_min, y_max = self.spatial_bounds['y']
        z_min, z_max = self.spatial_bounds['z']

        # 归一化到[0, 1]
        points_norm = np.zeros_like(points)
        points_norm[:, 0] = (points[:, 0] - x_min) / (x_max - x_min)
        points_norm[:, 1] = (points[:, 1] - y_min) / (y_max - y_min)
        points_norm[:, 2] = (points[:, 2] - z_min) / (z_max - z_min)

        # 转换为体素索引
        voxel_indices = (points_norm * np.array([nx, ny, nz])).astype(np.int32)
        voxel_indices = np.clip(voxel_indices, 0, np.array([nx - 1, ny - 1, nz - 1]))

        # 统计每个体素中各ID的点数
        voxel_id_counts = {}  # key: (ix, iy, iz), value: {id: count}

        for i in range(len(points)):
            ix, iy, iz = voxel_indices[i]
            label = int(labels[i])

            key = (ix, iy, iz)
            if key not in voxel_id_counts:
                voxel_id_counts[key] = {}

            if label not in voxel_id_counts[key]:
                voxel_id_counts[key][label] = 0
            voxel_id_counts[key][label] += 1

        # 为每个体素分配ID（只选择点数最多且满足阈值的ID）
        filtered_voxels = 0
        assigned_voxels = 0

        for (ix, iy, iz), id_counts in voxel_id_counts.items():
            # 找到占用最多的ID及其点数
            dominant_id, max_count = max(id_counts.items(), key=lambda x: x[1])

            # 只有当该ID的点数 >= 阈值时才分配
            if max_count >= self.min_points_per_voxel:
                voxel_grid[ix, iy, iz] = dominant_id
                assigned_voxels += 1
            else:
                filtered_voxels += 1

        # if filtered_voxels > 0:
            # print(f"[Voxel] 过滤了 {filtered_voxels} 个点数不足的体素 "
            #       f"(保留 {assigned_voxels} 个, 阈值: {self.min_points_per_voxel})")

        return voxel_grid

    def save_frames_as_json(self, output_dir: str, episode_id: int, env=None):
        """保存体素帧序列为紧凑格式JSON: [time_step, instance_id, ix, iy, iz]"""
        os.makedirs(output_dir, exist_ok=True)

        # 收集所有体素数据为紧凑格式 [time_step, instance_id, ix, iy, iz]
        compact_voxels = []
        step_indices = []  # 记录每个time_step对应的step_idx

        for frame_idx, frame in enumerate(self.frames):
            voxel_grid = frame['voxel_grid']
            step_indices.append(frame['step_idx'])

            # 找到所有非空体素
            occupied_indices = np.argwhere(voxel_grid >= 0)

            for ix, iy, iz in occupied_indices:
                instance_id = int(voxel_grid[ix, iy, iz])
                compact_voxels.append([frame_idx, instance_id, int(ix), int(iy), int(iz)])

        # 合并metadata和紧凑数据到一个文件
        combined_data = {
            'metadata': {
                'episode_id': episode_id,
                'total_frames': len(self.frames),
                'voxel_grid_size': list(self.voxel_grid_size),
                'spatial_bounds': self.spatial_bounds,
                'voxel_size': self.voxel_size,
                'min_points_per_voxel': self.min_points_per_voxel,
                'cameras': self.cam_names,
                'class_id_to_name': self.class_id_to_name,
                'step_indices': step_indices,  # 每帧对应的step_idx
                'data_format': 'compact',
                'data_description': 'Each row: [time_step, instance_id, ix, iy, iz]'
            },
            'voxels': compact_voxels  # 紧凑格式: [[t, id, x, y, z], ...]
        }

        output_path = f"{output_dir}/voxel_ep_{episode_id}.json"
        # print(f"[Voxel] 正在保存 {len(self.frames)} 帧 ({len(compact_voxels)} 体素) 到 {output_path}...")

        with open(output_path, 'w') as f:
            json.dump(combined_data, f)

        # size_mb = os.path.getsize(output_path) / 1024 / 1024
        # print(f"[Voxel] 已保存 -> {output_path} ({size_mb:.1f} MB)")
        # print(f"[Voxel] 压缩率: {len(compact_voxels)} 体素记录")

        return output_path

    def get_summary(self):
        """返回重建摘要"""
        if not self.frames:
            return {"error": "没有捕获任何帧"}

        occupied_counts = []
        for frame in self.frames:
            voxel_grid = frame['voxel_grid']
            occupied_counts.append(np.sum(voxel_grid >= 0))

        return {
            'total_frames': len(self.frames),
            'voxel_grid_size': self.voxel_grid_size,
            'avg_occupied_voxels': int(np.mean(occupied_counts)),
            'max_occupied_voxels': max(occupied_counts),
            'min_occupied_voxels': min(occupied_counts),
            'cameras': self.cam_names
        }
