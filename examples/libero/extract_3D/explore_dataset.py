import os
import pandas as pd
import numpy as np
from pathlib import Path
import json

# 可选依赖，如果没有安装就跳过图像处理
try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("注意: PIL/Pillow未安装，跳过图像保存功能")

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("注意: matplotlib未安装，跳过图表功能")


def explore_libero_dataset(dataset_path):
    """
    探索Libero数据集的结构和内容

    Args:
        dataset_path: 数据集根目录路径
    """
    print("=== Libero数据集探索 ===\n")

    # 1. 检查目录结构
    print("1. 目录结构:")
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 路径 {dataset_path} 不存在!")
        return

    # 寻找数据文件
    data_dir = dataset_path / "data"
    if data_dir.exists():
        chunks = list(data_dir.glob("chunk-*"))
        print(f"   发现 {len(chunks)} 个数据chunks:")
        for chunk in sorted(chunks):
            episodes = list(chunk.glob("episode_*.parquet"))
            print(f"     {chunk.name}: {len(episodes)} 个episodes")

    # 检查元数据
    meta_files = list(dataset_path.glob("*.json"))
    if meta_files:
        print(f"   发现元数据文件: {[f.name for f in meta_files]}")

    print()

    # 2. 读取第一个episode进行详细分析
    first_episode = None
    for chunk in sorted(chunks):
        episodes = sorted(chunk.glob("episode_*.parquet"))
        if episodes:
            first_episode = episodes[0]
            break

    if first_episode is None:
        print("错误: 未找到任何episode文件!")
        return

    print(f"2. 分析第一个episode: {first_episode}")

    try:
        df = pd.read_parquet(first_episode)
        print(f"   总帧数: {len(df)}")
        print(f"   数据列: {list(df.columns)}")
        print(f"   数据形状: {df.shape}")
        print()

        # 3. 详细分析各个字段
        print("3. 字段详细信息:")
        for col in df.columns:
            col_data = df[col]
            print(f"   {col}:")
            print(f"     - 数据类型: {col_data.dtype}")

            if col_data.dtype == 'object':
                # 检查是否是图像数据或其他数组
                first_val = col_data.iloc[0]
                if isinstance(first_val, np.ndarray):
                    print(f"     - 数组形状: {first_val.shape}")
                    print(f"     - 数组数据类型: {first_val.dtype}")
                    print(f"     - 值范围: [{first_val.min():.3f}, {first_val.max():.3f}]")
                else:
                    print(f"     - 示例值: {first_val}")
            else:
                print(f"     - 值范围: [{col_data.min():.3f}, {col_data.max():.3f}]")
                if len(col_data) > 0:
                    print(f"     - 示例值: {col_data.iloc[0]}")
            print()

        # 4. 分析时间序列特性
        print("4. 时间序列分析:")
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            if len(timestamps) > 1:
                time_diffs = np.diff(timestamps)
                print(f"   时间间隔 - 平均: {np.mean(time_diffs):.3f}s, 标准差: {np.std(time_diffs):.4f}s")
                print(f"   总持续时间: {timestamps[-1] - timestamps[0]:.2f}s")

        if 'frame_index' in df.columns:
            print(f"   帧索引范围: {df['frame_index'].min()} - {df['frame_index'].max()}")

        if 'episode_index' in df.columns:
            unique_episodes = df['episode_index'].unique()
            print(f"   episode索引: {unique_episodes}")

        if 'task_index' in df.columns:
            unique_tasks = df['task_index'].unique()
            print(f"   任务索引: {unique_tasks}")

        print()

        # 5. 状态和动作分析
        print("5. 状态和动作分析:")
        if 'state' in df.columns:
            states = np.stack(df['state'].values)
            print(f"   状态维度: {states.shape}")
            print(f"   各维度统计:")
            for i in range(states.shape[1]):
                state_dim = states[:, i]
                print(
                    f"     维度{i}: 均值={np.mean(state_dim):.4f}, 标准差={np.std(state_dim):.4f}, 范围=[{np.min(state_dim):.4f}, {np.max(state_dim):.4f}]")

        if 'actions' in df.columns:
            actions = np.stack(df['actions'].values)
            print(f"   动作维度: {actions.shape}")
            print(f"   各维度统计:")
            for i in range(actions.shape[1]):
                action_dim = actions[:, i]
                print(
                    f"     维度{i}: 均值={np.mean(action_dim):.4f}, 标准差={np.std(action_dim):.4f}, 范围=[{np.min(action_dim):.4f}, {np.max(action_dim):.4f}]")

        print()

        # 6. 图像数据分析
        print("6. 图像数据分析:")
        image_cols = ['image', 'wrist_image']
        for img_col in image_cols:
            if img_col in df.columns:
                img_data = df[img_col].iloc[0]
                print(f"   {img_col}:")
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    print(f"     - 格式: PNG字节流")
                    print(f"     - 字节长度: {len(img_data['bytes'])}")
                    print(f"     - 路径信息: {img_data.get('path', 'N/A')}")

                    # 尝试解析PNG图像获取尺寸
                    if HAS_PIL:
                        try:
                            import io
                            img = Image.open(io.BytesIO(img_data['bytes']))
                            print(f"     - 图像尺寸: {img.size}")
                            print(f"     - 图像模式: {img.mode}")
                        except Exception as e:
                            print(f"     - 图像解析错误: {e}")
                elif isinstance(img_data, np.ndarray):
                    print(f"     - 形状: {img_data.shape}")
                    print(f"     - 数据类型: {img_data.dtype}")
                    print(f"     - 像素值范围: [{img_data.min()}, {img_data.max()}]")
                else:
                    print(f"     - 数据类型: {type(img_data)}")
                    print(f"     - 示例: {str(img_data)[:100]}...")

        print()

        # 7. 保存示例数据用于可视化
        print("7. 保存示例数据:")
        output_dir = Path("libero_analysis")
        output_dir.mkdir(exist_ok=True)

        # 保存前几帧的图像（如果PIL可用）
        if HAS_PIL:
            import io
            for i in range(min(3, len(df))):
                # 处理主摄像头图像
                if 'image' in df.columns:
                    img_data = df['image'].iloc[i]
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        try:
                            img = Image.open(io.BytesIO(img_data['bytes']))
                            img.save(output_dir / f"sample_image_frame_{i}.png")
                        except Exception as e:
                            print(f"     保存image帧{i}失败: {e}")
                    elif isinstance(img_data, np.ndarray):
                        img = Image.fromarray(img_data.astype(np.uint8))
                        img.save(output_dir / f"sample_image_frame_{i}.png")

                # 处理手腕摄像头图像
                if 'wrist_image' in df.columns:
                    wrist_img_data = df['wrist_image'].iloc[i]
                    if isinstance(wrist_img_data, dict) and 'bytes' in wrist_img_data:
                        try:
                            wrist_img = Image.open(io.BytesIO(wrist_img_data['bytes']))
                            wrist_img.save(output_dir / f"sample_wrist_image_frame_{i}.png")
                        except Exception as e:
                            print(f"     保存wrist_image帧{i}失败: {e}")
                    elif isinstance(wrist_img_data, np.ndarray):
                        wrist_img = Image.fromarray(wrist_img_data.astype(np.uint8))
                        wrist_img.save(output_dir / f"sample_wrist_image_frame_{i}.png")
            print(f"   示例图像已保存")
        else:
            print(f"   跳过图像保存 (需要安装Pillow)")
            # 至少打印图像数组的信息
            if 'image' in df.columns:
                img_data = df['image'].iloc[0]
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    print(f"   image: PNG字节流, 长度={len(img_data['bytes'])}")
                elif isinstance(img_data, np.ndarray):
                    print(f"   image数组信息: shape={img_data.shape}, dtype={img_data.dtype}")
            if 'wrist_image' in df.columns:
                wrist_img_data = df['wrist_image'].iloc[0]
                if isinstance(wrist_img_data, dict) and 'bytes' in wrist_img_data:
                    print(f"   wrist_image: PNG字节流, 长度={len(wrist_img_data['bytes'])}")
                elif isinstance(wrist_img_data, np.ndarray):
                    print(f"   wrist_image数组信息: shape={wrist_img_data.shape}, dtype={wrist_img_data.dtype}")

        # 保存数据摘要
        summary = {
            'total_frames': int(len(df)),  # 转换为Python int
            'columns': list(df.columns),
            'episode_info': {
                'episode_index': int(df['episode_index'].iloc[0]) if 'episode_index' in df.columns else None,
                'task_index': int(df['task_index'].iloc[0]) if 'task_index' in df.columns else None,
            },
            'duration': float(
                df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) if 'timestamp' in df.columns else None,
            'image_format': 'PNG_bytes' if 'image' in df.columns and isinstance(df['image'].iloc[0],
                                                                                dict) else 'unknown',
            'state_dimensions': int(len(df['state'].iloc[0])) if 'state' in df.columns else None,
            'action_dimensions': int(len(df['actions'].iloc[0])) if 'actions' in df.columns else None,
        }

        with open(output_dir / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=int)  # 添加default=int处理int64类型

        print(f"   分析结果已保存到 {output_dir}")
        print(f"   - 示例图像: sample_image_frame_*.png, sample_wrist_image_frame_*.png")
        print(f"   - 数据摘要: data_summary.json")

        return df

    except Exception as e:
        print(f"错误: 无法读取parquet文件: {e}")
        return None


def analyze_multiple_episodes(dataset_path, max_episodes=5):
    """
    分析多个episodes的统计信息
    """
    print("\n=== 多Episode统计分析 ===\n")

    dataset_path = Path(dataset_path)
    data_dir = dataset_path / "data"

    all_episodes = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        episodes = sorted(chunk_dir.glob("episode_*.parquet"))
        all_episodes.extend(episodes[:max_episodes])

    print(f"分析前 {min(len(all_episodes), max_episodes)} 个episodes:")

    episode_stats = []
    for ep_file in all_episodes[:max_episodes]:
        try:
            df = pd.read_parquet(ep_file)
            stats = {
                'file': ep_file.name,
                'frames': int(len(df)),  # 转换为Python int
                'duration': float(
                    df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) if 'timestamp' in df.columns else 0,
                'task_id': int(df['task_index'].iloc[0]) if 'task_index' in df.columns else None,
                'episode_id': int(df['episode_index'].iloc[0]) if 'episode_index' in df.columns else None,
            }
            episode_stats.append(stats)
            print(f"  {stats['file']}: {stats['frames']}帧, {stats['duration']:.2f}s, 任务{stats['task_id']}")
        except Exception as e:
            print(f"  错误读取 {ep_file.name}: {e}")

    if episode_stats:
        total_frames = sum(s['frames'] for s in episode_stats)
        total_duration = sum(s['duration'] for s in episode_stats)
        unique_tasks = len(set(s['task_id'] for s in episode_stats if s['task_id'] is not None))

        print(f"\n总结:")
        print(f"  总帧数: {total_frames}")
        print(f"  总时长: {total_duration:.2f}s")
        print(f"  不同任务数: {unique_tasks}")
        print(f"  平均每episode: {total_frames / len(episode_stats):.1f}帧, {total_duration / len(episode_stats):.2f}s")


if __name__ == "__main__":
    # 使用示例

    dataset_path = "/home/lyh/openpi/lerobot_libero_dataset"  # 默认路径

    df = explore_libero_dataset(dataset_path)
    if df is not None:
        analyze_multiple_episodes(dataset_path)

        print("\n=== 下一步3D重建建议 ===")
        print("1. 基于state数据重建机器人姿态 (8维状态可能包含关节角度/位置)")
        print("2. 使用双目视觉 (image + wrist_image) 进行深度估计")
        print("3. 结合actions序列分析运动轨迹")
        print("4. 可视化时间序列中的3D场景变化")