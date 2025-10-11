import json
import pathlib
from libero.libero import benchmark


def get_libero_suite_tasks():
    """获取libero所有套件的任务列表"""

    # 所有可用的套件
    suite_names = [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90"
    ]

    benchmark_dict = benchmark.get_benchmark_dict()
    all_suite_tasks = {}

    for suite_name in suite_names:
        print(f"\n=== {suite_name} ===")
        try:
            task_suite = benchmark_dict[suite_name]()
            num_tasks = task_suite.n_tasks
            print(f"任务数量: {num_tasks}")

            suite_tasks = []
            for task_id in range(num_tasks):
                task = task_suite.get_task(task_id)
                task_description = task.language
                suite_tasks.append({
                    'local_id': task_id,
                    'description': task_description
                })
                print(f"  {task_id}: {task_description}")

            all_suite_tasks[suite_name] = suite_tasks

        except Exception as e:
            print(f"加载 {suite_name} 失败: {e}")
            all_suite_tasks[suite_name] = []

    return all_suite_tasks


def load_dataset_tasks(dataset_path):
    """加载数据集的任务列表"""

    tasks_file = pathlib.Path(dataset_path) / "tasks.jsonl"
    dataset_tasks = []

    print(f"\n=== 数据集任务 (来自 {tasks_file}) ===")

    if not tasks_file.exists():
        print(f"错误: 文件不存在 {tasks_file}")
        return []

    with open(tasks_file, 'r') as f:
        for line in f:
            task_data = json.loads(line)
            dataset_tasks.append({
                'task_index': task_data['task_index'],
                'description': task_data['task']
            })
            print(f"  {task_data['task_index']}: {task_data['task']}")

    print(f"数据集任务总数: {len(dataset_tasks)}")
    return dataset_tasks


def find_task_mapping(libero_tasks, dataset_tasks):
    """尝试找出任务映射关系"""

    print(f"\n=== 任务映射分析 ===")

    # 创建libero任务的全局索引
    libero_global_tasks = []
    global_idx = 0

    for suite_name, tasks in libero_tasks.items():
        for task in tasks:
            libero_global_tasks.append({
                'global_idx': global_idx,
                'suite': suite_name,
                'local_id': task['local_id'],
                'description': task['description']
            })
            global_idx += 1

    print(f"Libero总任务数: {len(libero_global_tasks)}")
    print(f"数据集任务数: {len(dataset_tasks)}")

    # 尝试精确匹配
    exact_matches = []
    partial_matches = []
    no_matches = []

    for dataset_task in dataset_tasks:
        dataset_desc = dataset_task['description'].strip().lower()
        found_exact = False
        found_partial = False

        for libero_task in libero_global_tasks:
            libero_desc = libero_task['description'].strip().lower()

            # 精确匹配
            if dataset_desc == libero_desc:
                exact_matches.append({
                    'dataset_idx': dataset_task['task_index'],
                    'dataset_desc': dataset_task['description'],
                    'libero_global_idx': libero_task['global_idx'],
                    'libero_suite': libero_task['suite'],
                    'libero_local_id': libero_task['local_id'],
                    'libero_desc': libero_task['description']
                })
                found_exact = True
                break

        if not found_exact:
            # 部分匹配（关键词匹配）
            for libero_task in libero_global_tasks:
                libero_desc = libero_task['description'].strip().lower()

                # 提取关键词进行匹配
                dataset_keywords = set(dataset_desc.split())
                libero_keywords = set(libero_desc.split())

                # 计算词汇重叠度
                overlap = len(dataset_keywords & libero_keywords)
                total_words = len(dataset_keywords | libero_keywords)
                similarity = overlap / total_words if total_words > 0 else 0

                if similarity > 0.7:  # 70%以上相似度
                    partial_matches.append({
                        'dataset_idx': dataset_task['task_index'],
                        'dataset_desc': dataset_task['description'],
                        'libero_global_idx': libero_task['global_idx'],
                        'libero_suite': libero_task['suite'],
                        'libero_local_id': libero_task['local_id'],
                        'libero_desc': libero_task['description'],
                        'similarity': similarity
                    })
                    found_partial = True
                    break

        if not found_exact and not found_partial:
            no_matches.append(dataset_task)

    # 打印结果
    print(f"\n精确匹配: {len(exact_matches)}")
    for match in exact_matches:
        print(f"  数据集[{match['dataset_idx']}] -> Libero[{match['libero_suite']}:{match['libero_local_id']}]")
        print(f"    {match['dataset_desc']}")

    print(f"\n部分匹配: {len(partial_matches)}")
    for match in partial_matches:
        print(
            f"  数据集[{match['dataset_idx']}] -> Libero[{match['libero_suite']}:{match['libero_local_id']}] (相似度: {match['similarity']:.2f})")
        print(f"    数据集: {match['dataset_desc']}")
        print(f"    Libero: {match['libero_desc']}")

    print(f"\n无匹配: {len(no_matches)}")
    for task in no_matches:
        print(f"  数据集[{task['task_index']}]: {task['description']}")

    return exact_matches, partial_matches, no_matches


def generate_mapping_code(exact_matches, partial_matches):
    """生成映射代码"""

    print(f"\n=== 生成的映射代码 ===")

    all_matches = exact_matches + partial_matches

    if not all_matches:
        print("没有找到任何匹配，无法生成映射代码")
        return

    print("# 数据集task_index到libero套件的映射")
    print("TASK_TO_SUITE_MAPPING = {")

    for match in sorted(all_matches, key=lambda x: x['dataset_idx']):
        print(
            f"    {match['dataset_idx']}: ('{match['libero_suite']}', {match['libero_local_id']}),  # {match['dataset_desc']}")

    print("}")

    print("\ndef get_libero_task_info(dataset_task_idx):")
    print("    if dataset_task_idx in TASK_TO_SUITE_MAPPING:")
    print("        suite_name, local_task_id = TASK_TO_SUITE_MAPPING[dataset_task_idx]")
    print("        return suite_name, local_task_id")
    print("    else:")
    print("        raise ValueError(f'Unknown task_index: {dataset_task_idx}')")


def main():
    """主函数"""

    print("开始分析libero任务映射...")

    # 获取libero所有任务
    print("1. 获取libero套件任务...")
    libero_tasks = get_libero_suite_tasks()

    # 获取数据集任务
    print("\n2. 加载数据集任务...")
    dataset_path = "/home/lyh/PycharmProjects/openpi/examples/libero/lerobot_libero_dataset/meta"
    dataset_tasks = load_dataset_tasks(dataset_path)

    # 分析映射关系
    print("\n3. 分析映射关系...")
    exact_matches, partial_matches, no_matches = find_task_mapping(libero_tasks, dataset_tasks)

    # 生成映射代码
    print("\n4. 生成映射代码...")
    generate_mapping_code(exact_matches, partial_matches)

    # 保存结果
    output_dir = pathlib.Path("data/libero_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_result = {
        'libero_tasks': libero_tasks,
        'dataset_tasks': dataset_tasks,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'no_matches': no_matches
    }

    output_file = output_dir / "task_mapping_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    print(f"\n分析结果已保存到: {output_file}")


if __name__ == "__main__":
    main()