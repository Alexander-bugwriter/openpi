import numpy as np
import json
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import pathlib


def get_segmentation_instance_ids(env, camera_name="agentview", steps=3):
    """
    获取分割渲染的instance ID及其对应的几何体名称
    使用libero_replay.py中的方法获取观察值
    """

    print("=" * 80)
    print("分割渲染Instance ID分析")
    print("=" * 80)

    model = env.sim.model
    instance_mappings = []

    for step in range(steps):
        print(f"\n--- 步骤 {step} ---")


        # 后续步骤执行零动作获取观察值
        action = [0.0] * 7  # 零动作
        obs, reward, done, info = env.step(action)

        # 获取分割图
        seg_key = f"{camera_name}_segmentation_instance"
        if seg_key not in obs:
            print(f"警告: 观察值中没有找到 {seg_key}")
            # 尝试其他可能的分割图键名
            available_keys = [k for k in obs.keys() if 'seg' in k.lower()]
            print(f"可用的分割图键: {available_keys}")
            if available_keys:
                seg_key = available_keys[0]
                print(f"使用 {seg_key} 作为分割图")
            else:
                return {}, {"error": "找不到分割图"}

        seg = obs[seg_key]

        # 处理分割图形状
        if len(seg.shape) == 3:
            seg = seg.squeeze()

        print(f"分割图形状: {seg.shape}, 数据类型: {seg.dtype}")
        print(f"分割图值范围: {seg.min()} - {seg.max()}")

        # 获取唯一的instance ID
        unique_ids = np.unique(seg)
        print(f"找到 {len(unique_ids)} 个唯一instance ID: {sorted(unique_ids.tolist())}")

        # 创建ID到几何体名称的映射
        id_to_geom = {}
        for uid in unique_ids:
            # 过滤掉背景和无效ID
            if uid == 0 or uid >= 10000:
                id_to_geom[uid] = "背景/无效"
                continue

            # 尝试映射到几何体
            if 0 <= uid < model.ngeom:
                geom_name = model.geom_id2name(uid)
                if geom_name:
                    id_to_geom[uid] = geom_name
                else:
                    id_to_geom[uid] = f"几何体ID_{uid}(无名)"
            else:
                id_to_geom[uid] = f"未知ID_{uid}"

        instance_mappings.append(id_to_geom)

        # 打印当前步骤的映射
        print(f"步骤 {step} 的instance ID映射:")
        for uid, name in sorted(id_to_geom.items()):
            if "背景" not in name and "未知" not in name:
                print(f"  ID {uid:3d} -> {name}")

    # 分析一致性
    consistency_report = analyze_consistency(instance_mappings)

    return instance_mappings[0] if instance_mappings else {}, consistency_report


def analyze_consistency(instance_mappings):
    """分析不同步骤间instance ID的一致性"""
    print(f"\n--- Instance ID一致性分析 ---")

    if len(instance_mappings) < 2:
        return {"warning": "步骤数不足，无法分析一致性"}

    # 检查所有步骤中共同的ID
    all_ids = [set(mapping.keys()) for mapping in instance_mappings]
    common_ids = set.intersection(*all_ids)

    print(f"所有步骤共同的ID数量: {len(common_ids)}")
    print(f"共同的ID: {sorted(common_ids)}")

    # 检查ID到名称映射的一致性
    consistency_issues = []
    for uid in common_ids:
        names = [mapping[uid] for mapping in instance_mappings]
        if len(set(names)) > 1:
            consistency_issues.append({
                'id': uid,
                'names': names
            })
            print(f"  ⚠️ ID {uid} 名称不一致: {names}")
        else:
            print(f"  ✅ ID {uid} 名称一致: {names[0]}")

    return {
        'total_steps': len(instance_mappings),
        'common_ids_count': len(common_ids),
        'consistency_issues': consistency_issues,
        'consistency_rate': f"{(len(common_ids) - len(consistency_issues)) / len(common_ids) * 100:.1f}%" if common_ids else "N/A"
    }


def get_detailed_geom_info(env, instance_ids):
    """获取几何体的详细信息"""
    model = env.sim.model
    geom_info = {}

    for uid in instance_ids:
        if 0 <= uid < model.ngeom:
            geom_info[uid] = {
                'name': model.geom_id2name(uid),
                'type': model.geom_type[uid],
                'body_name': model.body_names[model.geom_bodyid[uid]] if model.geom_bodyid[
                                                                             uid] < model.nbody else "未知",
                'rgba': model.geom_rgba[uid].tolist() if hasattr(model, 'geom_rgba') else [1, 1, 1, 1],
                'size': model.geom_size[uid].tolist() if hasattr(model, 'geom_size') else [0, 0, 0]
            }

    return geom_info


def setup_libero_environment(suite_name='libero_10', task_index=0):
    """设置LIBERO环境 - 使用libero_replay.py中的方法"""
    benchmark_dict = benchmark.get_benchmark_dict()

    try:
        task_suite = benchmark_dict[suite_name]()
        task = task_suite.get_task(task_index)

        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        # 使用libero_replay.py中的相机配置
        camera_names = ["agentview", "robot0_eye_in_hand", "sideview", "frontview"]

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": 256,
            "camera_widths": 256,
            "camera_names": camera_names,
            "camera_depths": True,
            "camera_segmentations": "instance",
        }

        env = OffScreenRenderEnv(**env_args)
        env.seed(42)
        env.reset()

        print(f"成功初始化环境: {suite_name}, 任务: {task.language}")
        return env, task.language

    except Exception as e:
        print(f"初始化环境失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """主函数：运行分割渲染分析"""

    # 设置环境
    env, task_description = setup_libero_environment('libero_10', 0)

    if env is None:
        print("环境初始化失败，退出")
        return

    try:
        # 分析所有相机的分割渲染
        cameras = ["agentview", "robot0_eye_in_hand", "sideview", "frontview"]
        all_results = {}

        for camera in cameras:
            print(f"\n{'=' * 60}")
            print(f"分析相机: {camera}")
            print(f"{'=' * 60}")

            # 重置环境以确保每个相机从相同状态开始
            env.reset()

            instance_mapping, consistency_report = get_segmentation_instance_ids(
                env, camera_name=camera, steps=2
            )

            # 获取几何体详细信息
            valid_ids = [uid for uid in instance_mapping.keys()
                         if uid > 0 and "背景" not in instance_mapping[uid] and "未知" not in instance_mapping[uid]]
            geom_details = get_detailed_geom_info(env, valid_ids)

            all_results[camera] = {
                'task_description': task_description,
                'instance_mapping': instance_mapping,
                'consistency_report': consistency_report,
                'geom_details': geom_details
            }

        # 保存结果到JSON
        output_file = "segmentation_instance_analysis.json"

        # 转换为可JSON序列化的格式
        serializable_results = {}
        for camera, result in all_results.items():
            serializable_results[camera] = {
                'task_description': result['task_description'],
                'instance_mapping': {str(k): v for k, v in result['instance_mapping'].items()},
                'consistency_report': result['consistency_report'],
                'geom_details': {str(k): v for k, v in result['geom_details'].items()}
            }

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\n{'=' * 80}")
        print(f"分析完成！结果已保存到: {output_file}")
        print(f"{'=' * 80}")

        # 打印摘要
        print(f"\n=== 分析摘要 ===")
        for camera, result in all_results.items():
            valid_objects = len([k for k, v in result['instance_mapping'].items()
                                 if "背景" not in v and "未知" not in v])
            print(f"{camera}: {valid_objects} 个可识别物体")
            print(f"  一致性: {result['consistency_report'].get('consistency_rate', 'N/A')}")

    finally:
        # 确保环境被关闭
        env.close()
        print("环境已关闭")


def quick_test():
    """快速测试函数：只分析一个相机"""
    env, task_description = setup_libero_environment('libero_10', 0)

    if env is None:
        return

    try:
        # 只分析agentview相机
        instance_mapping, consistency_report = get_segmentation_instance_ids(
            env, camera_name="agentview", steps=2
        )

        print(f"\n=== 最终结果 ===")
        print(f"任务: {task_description}")
        print(f"找到 {len(instance_mapping)} 个instance ID")

        # 打印非背景的物体
        print("\n检测到的物体:")
        for uid, name in sorted(instance_mapping.items()):
            if "背景" not in name and "未知" not in name:
                print(f"  ID {uid}: {name}")

        # 保存简化结果
        quick_result = {
            'task_description': task_description,
            'instance_mapping': {str(k): v for k, v in instance_mapping.items()},
            'consistency': consistency_report
        }

        with open('quick_segmentation_analysis.json', 'w') as f:
            json.dump(quick_result, f, indent=2)

        print(f"\n快速分析结果已保存到: quick_segmentation_analysis.json")

    finally:
        env.close()


if __name__ == "__main__":
    # 运行完整分析
    main()

    # 或者运行快速测试
    # quick_test()
