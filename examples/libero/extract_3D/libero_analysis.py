import json
import pathlib
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np


def get_libero_task_info(dataset_task_idx):
    """根据数据集task_index获取对应的libero套件和local_task_id"""
    TASK_TO_SUITE_MAPPING = {
        0: ('libero_10', 4),  # libero_10套件
        10: ('libero_goal', 8),  # libero_goal套件
        20: ('libero_object', 9),  # libero_object套件
        30: ('libero_spatial', 6),  # libero_spatial套件
    }
    if dataset_task_idx in TASK_TO_SUITE_MAPPING:
        suite_name, local_task_id = TASK_TO_SUITE_MAPPING[dataset_task_idx]
        return suite_name, local_task_id
    else:
        raise ValueError(f'Unknown task_index: {dataset_task_idx}')


def _get_libero_env(task, resolution, seed):
    """初始化LIBERO环境"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    camera_names = ["agentview", "robot0_eye_in_hand", "sideview", "frontview"]

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "camera_depths": True,
        "camera_segmentations": "instance",
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def analyze_environment_mappings(env, task_description):
    """分析环境的完整映射关系"""
    model = env.sim.model

    # 收集所有映射
    mappings = {
        'task_description': task_description,
        'site_mapping': {},
        'body_mapping': {},
        'geom_mapping': {},
        'actuator_mapping': {},
        'sensor_mapping': {},
        'joint_mapping': {}
    }

    print(f"\n{'=' * 80}")
    print(f"=== 环境分析: {task_description}")
    print(f"{'=' * 80}")

    # Site映射
    print(f"\n--- Site映射 (共{model.nsite}个) ---")
    for site_id in range(model.nsite):
        name = model.site_id2name(site_id)
        if name:
            mappings['site_mapping'][site_id] = name
            print(f"  Site ID {site_id}: {name}")

    # Body映射
    print(f"\n--- Body映射 (共{model.nbody}个) ---")
    for body_id in range(model.nbody):
        name = model.body_names[body_id]
        mappings['body_mapping'][body_id] = name
        if "robot" in name.lower() or "table" in name.lower() or any(
                obj in name.lower() for obj in ['plate', 'bowl', 'mug', 'cabinet', 'drawer', 'stove']):
            print(f"  Body ID {body_id}: {name}")

    # Geom映射 (用于分割标签)
    print(f"\n--- Geom映射 (共{model.ngeom}个) ---")
    for geom_id in range(model.ngeom):
        name = model.geom_id2name(geom_id)
        mappings['geom_mapping'][geom_id] = name
        if "robot" in name.lower() or "table" in name.lower() or any(
                obj in name.lower() for obj in ['plate', 'bowl', 'mug', 'cabinet', 'drawer', 'stove']):
            print(f"  Geom ID {geom_id}: {name}")

    # Actuator映射
    print(f"\n--- Actuator映射 (共{model.nu}个) ---")
    for actuator_id in range(model.nu):
        name = model.actuator_id2name(actuator_id)
        if name:
            mappings['actuator_mapping'][actuator_id] = name
            print(f"  Actuator ID {actuator_id}: {name}")

    # 按类别统计关键物体
    print(f"\n--- 关键物体分类统计 ---")
    key_objects = {
        'robot': [],
        'table': [],
        'plate': [],
        'bowl': [],
        'mug': [],
        'cabinet': [],
        'drawer': [],
        'stove': [],
        'basket': [],
        'microwave': []
    }

    for body_id, name in mappings['body_mapping'].items():
        for category in key_objects.keys():
            if category in name.lower():
                key_objects[category].append((body_id, name))

    for category, objects in key_objects.items():
        if objects:
            print(f"  {category}: {len(objects)}个")
            for obj_id, obj_name in objects[:3]:  # 只显示前3个
                print(f"    - {obj_id}: {obj_name}")
            if len(objects) > 3:
                print(f"    ... 还有{len(objects) - 3}个")

    return mappings


def compare_suite_mappings():
    """比较不同任务套件的映射关系"""
    benchmark_dict = benchmark.get_benchmark_dict()
    resolution = 256
    seed = 0

    # 选择每个套件的一个代表性任务
    task_indices = [0, 10, 20, 30]  # 每个套件选一个任务

    all_suite_mappings = {}

    for task_idx in task_indices:
        try:
            suite_name, local_task_id = get_libero_task_info(task_idx)
            print(f"\n{'#' * 80}")
            print(f"正在分析套件: {suite_name}, 任务索引: {local_task_id}")
            print(f"{'#' * 80}")

            # 初始化任务套件
            task_suite = benchmark_dict[suite_name]()
            task = task_suite.get_task(local_task_id)

            # 初始化环境
            env, task_description = _get_libero_env(task, resolution, seed)
            env.reset()

            # 分析映射关系
            mappings = analyze_environment_mappings(env, task_description)
            all_suite_mappings[suite_name] = mappings

            # 关闭环境
            env.close()

        except Exception as e:
            print(f"处理任务索引 {task_idx} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_suite_mappings


def save_mappings_to_json(all_mappings, output_file="libero_suite_mappings.json"):
    """将映射关系保存为JSON文件"""
    # 转换为可JSON序列化的格式
    serializable_mappings = {}

    for suite_name, mappings in all_mappings.items():
        serializable_mappings[suite_name] = {
            'task_description': mappings['task_description'],
            'site_mapping': {str(k): v for k, v in mappings['site_mapping'].items()},
            'body_mapping': {str(k): v for k, v in mappings['body_mapping'].items()},
            'geom_mapping': {str(k): v for k, v in mappings['geom_mapping'].items()},
            'actuator_mapping': {str(k): v for k, v in mappings['actuator_mapping'].items()},
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_mappings, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 80}")
    print(f"映射关系已保存到: {output_file}")
    print(f"{'=' * 80}")

    return output_file


def analyze_cross_suite_consistency(all_mappings):
    """分析跨套件的ID一致性"""
    print(f"\n{'=' * 80}")
    print("=== 跨套件ID一致性分析 ===")
    print(f"{'=' * 80}")

    # 收集所有套件中的物体名称
    all_objects = {}
    for suite_name, mappings in all_mappings.items():
        all_objects[suite_name] = set(mappings['body_mapping'].values())

    # 查找共同物体
    common_objects = set.intersection(*[set(objects) for objects in all_objects.values()])

    print(f"\n所有套件共有的物体 ({len(common_objects)}个):")
    for obj in sorted(common_objects)[:20]:  # 只显示前20个
        print(f"  - {obj}")
    if len(common_objects) > 20:
        print(f"  ... 还有{len(common_objects) - 20}个")

    # 检查相同物体名称在不同套件中的ID
    print(f"\n--- 相同物体名称的ID比较 ---")
    test_objects = ['robot0', 'table', 'main_table']  # 测试一些常见物体

    for obj_name in test_objects:
        print(f"\n物体 '{obj_name}':")
        found = False
        for suite_name, mappings in all_mappings.items():
            for body_id, name in mappings['body_mapping'].items():
                if obj_name in name:
                    print(f"  {suite_name}: Body ID = {body_id}")
                    found = True
                    break
        if not found:
            print(f"  在所有套件中未找到")


if __name__ == "__main__":
    print("开始分析LIBERO任务套件的映射关系...")

    # 比较所有套件的映射
    all_mappings = compare_suite_mappings()

    if all_mappings:
        # 保存为JSON
        output_file = save_mappings_to_json(all_mappings)

        # 分析跨套件一致性
        analyze_cross_suite_consistency(all_mappings)

        print(f"\n分析完成！")
        print(f"共分析了 {len(all_mappings)} 个任务套件")
        print(f"详细信息已保存到: {output_file}")
    else:
        print("未能获取任何套件的映射信息")