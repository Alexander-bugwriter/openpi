import numpy as np
import mujoco as mj
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

@dataclass
class VoxelFrame:
    """单帧体素数据"""
    timestamp: float
    step_idx: int
    occupancy: np.ndarray  # [H, W, D] bool array
    labels: np.ndarray     # [H, W, D] int array
    object_info: Dict[str, int]  # object_name -> label_id mapping

class SimpleVoxelReconstructor:
    """简化的MuJoCo体素重建器"""

    def __init__(self, voxel_size: float = 0.02,
                 workspace_bounds: Tuple[Tuple[float, float], ...] = None):
        """
        Args:
            voxel_size: 体素尺寸(米)，默认2cm
            workspace_bounds: 工作空间边界 ((x_min,x_max), (y_min,y_max), (z_min,z_max))
        """
        self.voxel_size = voxel_size

        # 默认LIBERO桌面工作空间
        if workspace_bounds is None:
            self.workspace_bounds = ((-0.4, 0.4), (-0.4, 0.4), (0.75, 1.2))
        else:
            self.workspace_bounds = workspace_bounds

        # 计算网格尺寸
        self.grid_dims = []
        self.origin = []
        for (min_val, max_val) in self.workspace_bounds:
            dim = int(np.ceil((max_val - min_val) / voxel_size))
            self.grid_dims.append(dim)
            self.origin.append(min_val)

        self.origin = np.array(self.origin)
        self.voxel_frames = []  # 存储时间序列

        print(f"体素网格初始化: {self.grid_dims} = {np.prod(self.grid_dims)} 个体素")
        print(f"工作空间: {self.workspace_bounds}")

    def _world_to_voxel_idx(self, world_pos: np.ndarray) -> np.ndarray:
        """世界坐标转体素索引"""
        voxel_pos = (world_pos - self.origin) / self.voxel_size
        return np.floor(voxel_pos).astype(int)

    def _get_object_label(self, body_name: str) -> int:
        """简单的语义标签映射"""
        name_lower = body_name.lower()

        # 基础标签映射
        if any(x in name_lower for x in ['robot', 'gripper', 'hand', 'arm']):
            return 1  # 机器人
        elif 'table' in name_lower:
            return 2  # 桌子
        elif any(x in name_lower for x in ['mug', 'cup']):
            return 3  # 杯子类
        elif any(x in name_lower for x in ['plate', 'bowl']):
            return 4  # 餐具类
        elif any(x in name_lower for x in ['box', 'container']):
            return 5  # 容器类
        else:
            return 6  # 其他物体

    def capture_frame(self, model: mj.MjModel, data: mj.MjData,
                      timestamp: float, step_idx: int) -> VoxelFrame:
        """捕获当前帧的体素化场景"""

        # 初始化体素网格
        occupancy = np.zeros(self.grid_dims, dtype=bool)
        labels = np.zeros(self.grid_dims, dtype=int)
        object_info = {}

        # 遍历所有几何体
        for geom_id in range(model.ngeom):
            geom_pos = data.geom_xpos[geom_id]
            geom_size = model.geom_size[geom_id]
            body_id = model.geom_bodyid[geom_id]

            # 获取物体名称
            body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id)
            if body_name is None:
                body_name = f"body_{body_id}"

            label = self._get_object_label(body_name)
            object_info[body_name] = label

            # 简化处理：每个几何体用一个小立方体表示
            self._mark_occupied_voxels(occupancy, labels, geom_pos, geom_size, label)

        frame = VoxelFrame(
            timestamp=timestamp,
            step_idx=step_idx,
            occupancy=occupancy,
            labels=labels,
            object_info=object_info
        )

        self.voxel_frames.append(frame)
        return frame

    def _mark_occupied_voxels(self, occupancy: np.ndarray, labels: np.ndarray,
                              pos: np.ndarray, size: np.ndarray, label: int):
        """标记占用的体素"""
        # 使用物体的包围盒
        half_size = np.maximum(size[:3], [self.voxel_size/2] * 3)  # 至少一个体素大小

        min_pos = pos - half_size
        max_pos = pos + half_size

        # 转换为体素索引
        min_idx = self._world_to_voxel_idx(min_pos)
        max_idx = self._world_to_voxel_idx(max_pos)

        # 限制在网格范围内
        min_idx = np.maximum(min_idx, 0)
        max_idx = np.minimum(max_idx, np.array(self.grid_dims) - 1)

        # 标记占用
        if min_idx[0] <= max_idx[0] and min_idx[1] <= max_idx[1] and min_idx[2] <= max_idx[2]:
            occupancy[min_idx[0]:max_idx[0]+1,
            min_idx[1]:max_idx[1]+1,
            min_idx[2]:max_idx[2]+1] = True
            labels[min_idx[0]:max_idx[0]+1,
            min_idx[1]:max_idx[1]+1,
            min_idx[2]:max_idx[2]+1] = label

    def save_frames_as_json(self, output_dir: str, episode_id: int):
        """保存体素帧序列为JSON格式，便于网页可视化"""
        os.makedirs(output_dir, exist_ok=True)

        # 保存元数据
        metadata = {
            'episode_id': episode_id,
            'voxel_size': self.voxel_size,
            'grid_dims': self.grid_dims,
            'origin': self.origin.tolist(),
            'workspace_bounds': self.workspace_bounds,
            'total_frames': len(self.voxel_frames),
            'label_mapping': {
                0: 'empty',
                1: 'robot',
                2: 'table',
                3: 'mug/cup',
                4: 'plate/bowl',
                5: 'box/container',
                6: 'other'
            }
        }

        with open(f"{output_dir}/metadata_ep_{episode_id}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # 保存压缩的体素数据
        frames_data = []
        for frame in self.voxel_frames:
            # 只保存占用的体素位置和标签
            occupied_indices = np.where(frame.occupancy)
            if len(occupied_indices[0]) > 0:
                voxel_data = []
                for i, j, k in zip(*occupied_indices):
                    voxel_data.append([i, j, k, int(frame.labels[i, j, k])])

                frame_data = {
                    'timestamp': frame.timestamp,
                    'step_idx': frame.step_idx,
                    'occupied_voxels': voxel_data,  # [[x,y,z,label], ...]
                    'object_count': len(frame.object_info)
                }
                frames_data.append(frame_data)

        with open(f"{output_dir}/voxel_frames_ep_{episode_id}.json", 'w') as f:
            json.dump(frames_data, f)

        print(f"已保存 {len(frames_data)} 帧体素数据到: {output_dir}")

    def get_summary(self) -> Dict:
        """获取重建摘要信息"""
        if not self.voxel_frames:
            return {"error": "没有捕获任何帧"}

        total_voxels = []
        unique_objects = set()

        for frame in self.voxel_frames:
            total_voxels.append(np.sum(frame.occupancy))
            unique_objects.update(frame.object_info.keys())

        return {
            'total_frames': len(self.voxel_frames),
            'avg_occupied_voxels': np.mean(total_voxels),
            'max_occupied_voxels': np.max(total_voxels),
            'unique_objects': list(unique_objects),
            'duration': self.voxel_frames[-1].timestamp - self.voxel_frames[0].timestamp,
            'grid_size': self.grid_dims,
            'total_grid_voxels': np.prod(self.grid_dims)
        }