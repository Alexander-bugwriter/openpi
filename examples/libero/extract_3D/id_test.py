import argparse
import os
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv
from voxel_reconstructor import VoxelReconstructor


def get_libero_env(task, resolution=256):
    """初始化 LIBERO 环境"""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    camera_names = ["agentview", "robot0_eye_in_hand", "sideview", "frontview"]

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "camera_depths": True,
        "camera_segmentations": "instance",
    }

    env = SegmentationRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def test_id_mapping_for_task(task, task_id, reconstructor, output_dir):
    """测试单个任务的 ID 映射并保存一帧体素数据"""
    print(f"\n{'=' * 80}")
    print(f"任务 {task_id}: {task.language}")
    print(f"{'=' * 80}")

    # 初始化环境
    env, task_description = get_libero_env(task)

    # 重置环境
    obs = env.reset()

    # 重置 reconstructor
    reconstructor.reset()

    # 获取相机名称
    reconstructor.cam_names = [k.replace('_depth', '') for k in obs.keys() if k.endswith('_depth')]
    print(f"检测到相机: {reconstructor.cam_names}")

    # 调用 _build_class_id_mapping
    print(f"\n调用 _build_class_id_mapping...")
    reconstructor._build_class_id_mapping(env, obs)

    # 输出构建的映射
    print(f"\n构建的 class_id_to_name 映射:")
    for pixel_id in sorted(reconstructor.class_id_to_name.keys()):
        class_name = reconstructor.class_id_to_name[pixel_id]
        print(f"  ID {pixel_id:2d} -> {class_name}")

    # 检查分割图中的实际 ID
    print(f"\n分割图中的实际 ID:")
    all_unique_ids = set()
    for cam_name in reconstructor.cam_names:
        seg = obs[f'{cam_name}_segmentation_instance']
        unique_ids = np.unique(seg)
        all_unique_ids.update(unique_ids.tolist())
        print(f"  {cam_name}: {sorted(unique_ids.tolist())}")

    print(f"  所有相机合并: {sorted(all_unique_ids)}")

    # 检查未映射的 ID
    unmapped_ids = all_unique_ids - set(reconstructor.class_id_to_name.keys())
    if unmapped_ids:
        print(f"\n❌ 未映射的 ID: {sorted(unmapped_ids)}")
    else:
        print(f"\n✅ 所有 ID 都已映射")

    # 捕获一帧并体素化
    print(f"\n捕获并体素化当前帧...")
    occupied_voxels = reconstructor.capture_frame(obs, env)
    print(f"  占用体素数: {occupied_voxels}")

    # 保存体素数据为JSON
    print(f"\n保存体素数据...")
    voxel_path = reconstructor.save_frames_as_json(output_dir,task_id,env)

    # 关闭环境
    env.close()

    return {
        'task_name': task_description,
        'mapped_ids': sorted(reconstructor.class_id_to_name.keys()),
        'actual_ids': sorted(all_unique_ids),
        'unmapped_ids': sorted(unmapped_ids),
        'occupied_voxels': occupied_voxels,
        'voxel_file': voxel_path
    }


def main(args):
    print(f"开始测试 LIBERO-10 任务套件的 VoxelReconstructor ID 映射")
    print(f"数据目录: {args.libero_raw_data_dir}")
    print(f"输出目录: {args.output_dir}\n")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取 LIBERO-10 任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_10"]()
    num_tasks = task_suite.n_tasks

    print(f"LIBERO-10 共有 {num_tasks} 个任务\n")

    # 初始化 VoxelReconstructor
    spatial_bounds = {
        'x': (-0.8, 0.8),
        'y': (-0.8, 0.8),
        'z': (0.35, 1.5)
    }
    reconstructor = VoxelReconstructor(
        voxel_grid_size=(64, 64, 64),
        spatial_bounds=spatial_bounds,
        min_points_per_voxel=1
    )

    # 存储所有任务的结果
    all_results = []

    # 遍历所有任务
    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        result = test_id_mapping_for_task(task, task_id, reconstructor, args.output_dir)
        all_results.append(result)

    # 汇总统计
    print(f"\n{'#' * 80}")
    print(f"汇总统计")
    print(f"{'#' * 80}\n")

    tasks_with_unmapped = [r for r in all_results if r['unmapped_ids']]
    print(f"有未映射 ID 的任务数: {len(tasks_with_unmapped)} / {num_tasks}")

    if tasks_with_unmapped:
        print(f"\n未映射 ID 详情:")
        for r in tasks_with_unmapped:
            print(f"  - {r['task_name']}")
            print(f"    未映射 ID: {r['unmapped_ids']}")
            print(f"    实际 ID 范围: {min(r['actual_ids'])} ~ {max(r['actual_ids'])}")
    else:
        print(f"\n✅ 所有任务的 ID 都成功映射！")

    # 输出体素文件信息
    print(f"\n生成的体素文件:")
    for r in all_results:
        print(f"  - {r['voxel_file']}")
        print(f"    占用体素数: {r['occupied_voxels']}")

    print(f"\n测试完成！")
    print(f"所有体素数据已保存到: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero_raw_data_dir",
        type=str,
        help="原始 HDF5 数据集目录路径",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./id_test_voxels",
        help="输出目录路径",
    )
    args = parser.parse_args()

    main(args)