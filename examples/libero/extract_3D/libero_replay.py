import collections
import dataclasses
import logging
import math
import pathlib
import pandas as pd
import numpy as np
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv,SegmentationRenderEnv
import tqdm
import tyro
import io
from PIL import Image
# from Voxel_Reconstructor import SimpleVoxelReconstructor
from pointmap_reconstructor import PointMapReconstructor
from voxel_reconstructor import VoxelReconstructor
import robosuite.macros as macros
import json
print(f"IMAGE_CONVENTION: {macros.IMAGE_CONVENTION}")

LIBERO_ENV_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASK_TO_SUITE_MAPPING = {
    # libero_10套件 (task_index 0-9)
    7: ('libero_10', 4),  # put the white mug on the left plate and put the yellow and white mug on the right plate
    8: ('libero_10', 6),  # put the white mug on the plate and put the chocolate pudding to the right of the plate
    2: ('libero_10', 9),  # put the yellow and white mug in the microwave and close it
    0: ('libero_10', 2),  # turn on the stove and put the moka pot on it
    4: ('libero_10', 7),  # put both the alphabet soup and the cream cheese box in the basket
    5: ('libero_10', 0),  # put both the alphabet soup and the tomato sauce in the basket
    3: ('libero_10', 8),  # put both moka pots on the stove
    6: ('libero_10', 1),  # put both the cream cheese box and the butter in the basket
    1: ('libero_10', 3),  # put the black bowl in the bottom drawer of the cabinet and close it
    9: ('libero_10', 5),  # pick up the book and place it in the back compartment of the caddy
    #
    # # libero_goal套件 (task_index 10-19)
    # 10: ('libero_goal', 8),  # put the bowl on the plate
    # 11: ('libero_goal', 9),  # put the wine bottle on the rack
    # 12: ('libero_goal', 3),  # open the top drawer and put the bowl inside
    # 13: ('libero_goal', 6),  # put the cream cheese in the bowl
    # 14: ('libero_goal', 2),  # put the wine bottle on top of the cabinet
    # 15: ('libero_goal', 5),  # push the plate to the front of the stove
    # 16: ('libero_goal', 7),  # turn on the stove
    # 17: ('libero_goal', 1),  # put the bowl on the stove
    # 18: ('libero_goal', 4),  # put the bowl on top of the cabinet
    # 19: ('libero_goal', 0),  # open the middle drawer of the cabinet
    #
    # # libero_object套件 (task_index 20-29)
    # 20: ('libero_object', 9),  # pick up the orange juice and place it in the basket
    # 21: ('libero_object', 4),  # pick up the ketchup and place it in the basket
    # 22: ('libero_object', 1),  # pick up the cream cheese and place it in the basket
    # 23: ('libero_object', 3),  # pick up the bbq sauce and place it in the basket
    # 24: ('libero_object', 0),  # pick up the alphabet soup and place it in the basket
    # 25: ('libero_object', 7),  # pick up the milk and place it in the basket
    # 26: ('libero_object', 2),  # pick up the salad dressing and place it in the basket
    # 27: ('libero_object', 6),  # pick up the butter and place it in the basket
    # 28: ('libero_object', 5),  # pick up the tomato sauce and place it in the basket
    # 29: ('libero_object', 8),  # pick up the chocolate pudding and place it in the basket
    #
    # # libero_spatial套件 (task_index 30-39)
    # 30: ('libero_spatial', 6),  # pick up the black bowl next to the cookie box and place it on the plate
    # 31: ('libero_spatial', 4),
    # # pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
    # 32: ('libero_spatial', 5),  # pick up the black bowl on the ramekin and place it on the plate
    # 33: ('libero_spatial', 7),  # pick up the black bowl on the stove and place it on the plate
    # 34: ('libero_spatial', 0),  # pick up the black bowl between the plate and the ramekin and place it on the plate
    # 35: ('libero_spatial', 3),  # pick up the black bowl on the cookie box and place it on the plate
    # 36: ('libero_spatial', 8),  # pick up the black bowl next to the plate and place it on the plate
    # 37: ('libero_spatial', 1),  # pick up the black bowl next to the ramekin and place it on the plate
    # 38: ('libero_spatial', 2),  # pick up the black bowl from table center and place it on the plate
    # 39: ('libero_spatial', 9),  # pick up the black bowl on the wooden cabinet and place it on the plate
}
# TASK_TO_SUITE_MAPPING = {
#     # libero_10套件 (task_index 0-9)
#     0: ('libero_10', 4),  # put the white mug on the left plate and put the yellow and white mug on the right plate
#     1: ('libero_10', 6),  # put the white mug on the plate and put the chocolate pudding to the right of the plate
#     2: ('libero_10', 9),  # put the yellow and white mug in the microwave and close it
#     3: ('libero_10', 2),  # turn on the stove and put the moka pot on it
#     4: ('libero_10', 7),  # put both the alphabet soup and the cream cheese box in the basket
#     5: ('libero_10', 0),  # put both the alphabet soup and the tomato sauce in the basket
#     6: ('libero_10', 8),  # put both moka pots on the stove
#     7: ('libero_10', 1),  # put both the cream cheese box and the butter in the basket
#     8: ('libero_10', 3),  # put the black bowl in the bottom drawer of the cabinet and close it
#     9: ('libero_10', 5),  # pick up the book and place it in the back compartment of the caddy
#
#     # libero_goal套件 (task_index 10-19)
#     10: ('libero_goal', 8),  # put the bowl on the plate
#     11: ('libero_goal', 9),  # put the wine bottle on the rack
#     12: ('libero_goal', 3),  # open the top drawer and put the bowl inside
#     13: ('libero_goal', 6),  # put the cream cheese in the bowl
#     14: ('libero_goal', 2),  # put the wine bottle on top of the cabinet
#     15: ('libero_goal', 5),  # push the plate to the front of the stove
#     16: ('libero_goal', 7),  # turn on the stove
#     17: ('libero_goal', 1),  # put the bowl on the stove
#     18: ('libero_goal', 4),  # put the bowl on top of the cabinet
#     19: ('libero_goal', 0),  # open the middle drawer of the cabinet
#
#     # libero_object套件 (task_index 20-29)
#     20: ('libero_object', 9),  # pick up the orange juice and place it in the basket
#     21: ('libero_object', 4),  # pick up the ketchup and place it in the basket
#     22: ('libero_object', 1),  # pick up the cream cheese and place it in the basket
#     23: ('libero_object', 3),  # pick up the bbq sauce and place it in the basket
#     24: ('libero_object', 0),  # pick up the alphabet soup and place it in the basket
#     25: ('libero_object', 7),  # pick up the milk and place it in the basket
#     26: ('libero_object', 2),  # pick up the salad dressing and place it in the basket
#     27: ('libero_object', 6),  # pick up the butter and place it in the basket
#     28: ('libero_object', 5),  # pick up the tomato sauce and place it in the basket
#     29: ('libero_object', 8),  # pick up the chocolate pudding and place it in the basket
#
#     # libero_spatial套件 (task_index 30-39)
#     30: ('libero_spatial', 6),  # pick up the black bowl next to the cookie box and place it on the plate
#     31: ('libero_spatial', 4),
#     # pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
#     32: ('libero_spatial', 5),  # pick up the black bowl on the ramekin and place it on the plate
#     33: ('libero_spatial', 7),  # pick up the black bowl on the stove and place it on the plate
#     34: ('libero_spatial', 0),  # pick up the black bowl between the plate and the ramekin and place it on the plate
#     35: ('libero_spatial', 3),  # pick up the black bowl on the cookie box and place it on the plate
#     36: ('libero_spatial', 8),  # pick up the black bowl next to the plate and place it on the plate
#     37: ('libero_spatial', 1),  # pick up the black bowl next to the ramekin and place it on the plate
#     38: ('libero_spatial', 2),  # pick up the black bowl from table center and place it on the plate
#     39: ('libero_spatial', 9),  # pick up the black bowl on the wooden cabinet and place it on the plate
# }
def get_libero_task_info(dataset_task_idx):
    """根据数据集task_index获取对应的libero套件和local_task_id"""
    if dataset_task_idx in TASK_TO_SUITE_MAPPING:
        suite_name, local_task_id = TASK_TO_SUITE_MAPPING[dataset_task_idx]
        return suite_name, local_task_id
    else:
        raise ValueError(f'Unknown task_index: {dataset_task_idx}')


# 🔥 在文件开头添加可视化函数
def visualize_segmentation(seg: np.ndarray, env):
    """将分割图转换为彩色可视化图像"""
    if len(seg.shape) == 3:
        seg = seg.squeeze()

    H, W = seg.shape
    vis_img = np.zeros((H, W, 3), dtype=np.uint8)

    # 为每个ID分配颜色
    unique_ids = np.unique(seg)
    for uid in unique_ids:
        mask = (seg == uid)
        # 使用黄金角分布生成颜色
        hue = (uid * 137.508) % 360
        r, g, b = hsl_to_rgb(hue / 360, 0.8, 0.6)
        vis_img[mask] = [int(r * 255), int(g * 255), int(b * 255)]

    # 添加文字标注显示有哪些物体
    if len(unique_ids) <= 10:  # 只在物体不太多时标注
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(vis_img)
        draw = ImageDraw.Draw(pil_img)

        y_offset = 5
        for uid in sorted(unique_ids):
            if 0 <= uid < env.sim.model.ngeom:
                name = env.sim.model.geom_id2name(uid)
                # 简短名称
                short_name = name.split('_')[0] if '_' in name else name
                text = f"ID{uid}:{short_name}"
                draw.text((5, y_offset), text, fill=(255, 255, 255))
                y_offset += 15

        vis_img = np.array(pil_img)

    return vis_img
def hsl_to_rgb(h, s, l):
    """HSL转RGB"""
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

@dataclasses.dataclass
class ReplayArgs:
    # dataset_path: str = "/home/lyh/PycharmProjects/openpi/examples/libero/lerobot_3D_libero_dataset"  # 硬编码数据集路径
    # dataset_path: str = "/home/lyh/PycharmProjects/openpi/examples/libero/lerobot_libero_dataset"
    dataset_path: str = "/home/lyh/PycharmProjects/openpi/examples/libero/lerobot_3D_libero_dataset_test/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo"
    video_out_path: str = "data/libero_replay_videos"  # 输出视频路径
    num_steps_wait: int = 10  # 等待物体稳定的步数
    seed: int = 0  # 随机种子
    max_episodes: int = 10  # 最大回放episodes数量
    debug: bool = True  # 调试模式
    save_comparison_video: bool = True  # 是否保存对比视频（原始vs重放）


def decode_image_from_bytes(image_data):
    """从PNG字节流解码图像"""
    if isinstance(image_data, dict) and 'bytes' in image_data:
        img = Image.open(io.BytesIO(image_data['bytes']))
        return np.array(img)
    elif isinstance(image_data, np.ndarray):
        return image_data
    else:
        raise ValueError(f"Unsupported image format: {type(image_data)}")


def check_parquet_precision(episode_file):
    """检查parquet文件中的数据精度"""
    df = pd.read_parquet(episode_file)
    actions = df['actions'].values

    print("=== Parquet数据精度检查 ===")
    print(f"actions列的dtype: {df['actions'].dtype}")
    print(f"单个action的类型: {type(actions[0])}")
    print(f"单个action的dtype: {actions[0].dtype if hasattr(actions[0], 'dtype') else 'N/A'}")

    # 检查前几个动作的精确值
    for i in range(min(3, len(actions))):
        action = actions[i]
        print(f"Action {i}: {action}")
        print(f"  精确值: {[f'{x:.10f}' for x in action]}")

    return actions


def trace_action_conversion(action_from_df):
    """追踪动作转换过程中的精度变化"""
    print("=== 动作转换精度追踪 ===")

    # 1. 原始从DataFrame读取的数据
    print(f"1. 从DataFrame读取: {action_from_df}")
    print(f"   类型: {type(action_from_df)}, dtype: {getattr(action_from_df, 'dtype', 'N/A')}")
    print(f"   精确值: {[f'{x:.10f}' for x in action_from_df]}")

    # 2. numpy stack后
    stacked = np.stack([action_from_df])  # 模拟你的处理过程
    single_action = stacked[0]
    print(f"2. numpy stack后: {single_action}")
    print(f"   类型: {type(single_action)}, dtype: {single_action.dtype}")
    print(f"   精确值: {[f'{x:.10f}' for x in single_action]}")

    # 3. action_to_libero_action转换
    libero_action = action_to_libero_action(single_action)
    print(f"3. libero转换后: {libero_action}")
    print(f"   类型: {type(libero_action)}")
    print(f"   精确值: {[f'{x:.10f}' for x in libero_action]}")

    # 4. 检查是否有精度损失
    if hasattr(action_from_df, 'dtype') and hasattr(single_action, 'dtype'):
        if action_from_df.dtype != single_action.dtype:
            print(f"⚠️ 检测到dtype变化: {action_from_df.dtype} -> {single_action.dtype}")

    return libero_action


def high_precision_action_conversion(action_array):
    """高精度动作转换"""
    # 确保使用double精度
    if action_array.dtype != np.float64:
        print(f"转换精度: {action_array.dtype} -> float64")
        action_array = action_array.astype(np.float64)

    # 直接传递numpy数组而不是转换为list
    if len(action_array) == 7:
        return action_array  # 不调用tolist()
    else:
        padded_action = np.zeros(7, dtype=np.float64)
        min_len = min(len(action_array), 7)
        padded_action[:min_len] = action_array[:min_len]
        return padded_action

def state_to_robot_state(state_8d):
    """将8维状态转换为机器人状态
    假设状态格式为: [x, y, z, qx, qy, qz, qw, gripper]
    """
    if len(state_8d) == 8:
        # 位置 (3D) + 四元数 (4D) + 夹爪 (1D)
        pos = state_8d[:3]
        quat = state_8d[3:7]
        gripper = state_8d[7:8]
        return np.concatenate([pos, quat, gripper])
    else:
        return state_8d


def action_to_libero_action(action_7d):
    """将7维动作转换为LIBERO格式
    假设动作格式为: [dx, dy, dz, drx, dry, drz, gripper_action]
    """

    return action_7d.tolist()


def integrate_voxel_reconstruction(reconstructor, env, timestamp, step_idx):
    """集成到replay循环中的简单调用"""
    try:
        # 获取MuJoCo模型和数据
        model = env.sim.model
        data = env.sim.data
        # print(f"环境类型: {type(env)}")
        # print(f"模型类型: {type(env.sim.model)}")
        # print(f"模型属性: {dir(env.sim.model)}")
        # 捕获当前帧
        frame = reconstructor.capture_frame(model, data, timestamp, step_idx)

        return True
    except Exception as e:
        print(f"体素重建出错: {e}")
        return False


def print_scene_objects(env, step_idx):
    """打印场景中关键物体的世界坐标"""
    if step_idx != 0:
        return

    print("\n" + "=" * 60)
    print("=== Step 0: 场景物体世界坐标 ===")
    print("=" * 60)

    model = env.sim.model
    data = env.sim.data

    # 关键词列表 - 根据LIBERO常见物体
    keywords = ['robot', 'base', 'plate', 'cup', 'mug', 'bowl',
                'table', 'cabinet', 'drawer', 'stove']

    print("\n物体名称 → 世界坐标 (x, y, z)")
    print("-" * 60)

    for body_id in range(model.nbody):
        body_name = model.body_names[body_id]

        # 只打印包含关键词的物体
        if any(keyword in body_name.lower() for keyword in keywords):
            pos = data.xpos[body_id]
            print(f"{body_name:30s} → ({pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f})")

    print("\n" + "=" * 60)
    print("注意: 这些是MuJoCo世界坐标系的坐标")
    print("     +X: 右, +Y: 前, +Z: 上")
    print("=" * 60 + "\n")

def replay_libero_episodes(args: ReplayArgs) -> None:
    """回放Libero数据集中的episodes"""

    # 设置随机种子
    np.random.seed(args.seed)

    # 创建输出目录
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    logging.info(f"数据集路径: {args.dataset_path}")

    # 查找数据集中的episodes
    dataset_path = pathlib.Path(args.dataset_path)
    data_dir = dataset_path / "data"

    episode_files = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        episodes = sorted(chunk_dir.glob("episode_*.parquet"))
        episode_files.extend(episodes)

    logging.info(f"找到 {len(episode_files)} 个episodes")

    # 限制回放的episodes数量
    episode_files = episode_files[:args.max_episodes]

    # 初始化任务套件映射
    benchmark_dict = benchmark.get_benchmark_dict()
    # reconstructor = SimpleVoxelReconstructor(voxel_size=0.05)
    spatial_bounds = {
        'x': (-0.8, 0.8),  # 左右范围
        'y': (-0.8, 0.8),  # 前后范围
        'z': (0.35, 1.5)  # 高度范围（桌面以上）
    }

    # reconstructor = PointMapReconstructor(
    #     max_points=30000,
    #     spatial_bounds=spatial_bounds  # 或者设为None不过滤
    # )
    reconstructor = VoxelReconstructor(
        voxel_grid_size=(64, 64, 64),  # 可以调整分辨率
        spatial_bounds=spatial_bounds,  # 必须指定，用于计算体素大小
        min_points_per_voxel=1  # 最小点数阈值（可调整）
    )

    for episode_idx, episode_file in enumerate(episode_files):
        logging.info(f"\n=== 回放 Episode {episode_idx + 1}: {episode_file.name} ===")


        # 加载episode数据
        df = pd.read_parquet(episode_file)
        # actions = check_parquet_precision(episode_file)
        # trace_action_conversion(actions[0])

        # 提取数据
        states = np.stack(df['state'].values)
        actions = np.stack(df['actions'].values)
        timestamps = df['timestamp'].values
        task_idx = int(df['task_index'].iloc[0])
        episode_id = int(df['episode_index'].iloc[0])

        logging.info(f"任务ID: {task_idx}, Episode ID: {episode_id}")
        logging.info(f"总帧数: {len(df)}, 持续时间: {timestamps[-1] - timestamps[0]:.2f}s")
        logging.info(f"状态形状: {states.shape}, 动作形状: {actions.shape}")

        # 解码原始图像序列
        original_images = []
        for i in range(len(df)):
            try:
                img_data = df['image'].iloc[i]
                img = decode_image_from_bytes(img_data)
                original_images.append(img)
            except Exception as e:
                logging.warning(f"解码图像 {i} 失败: {e}")
                # 使用前一帧或创建空白帧
                if original_images:
                    original_images.append(original_images[-1])
                else:
                    original_images.append(np.zeros((256, 256, 3), dtype=np.uint8))

        logging.info(f"成功解码 {len(original_images)} 张原始图像")

        try:
            suite_name, local_task_id = get_libero_task_info(task_idx)
            logging.info(f"使用任务套件: {suite_name}, 套件内任务索引: {local_task_id}")
        except ValueError as e:
            logging.error(f"任务映射错误: {e}")
            continue

            # 初始化LIBERO任务套件
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[suite_name]()

        # 获取任务
        task = task_suite.get_task(local_task_id)
        initial_states = task_suite.get_task_init_states(local_task_id)

        # 初始化环境
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        logging.info(f"任务描述: {task_description}")

        # 重置环境
        env.reset()
        reconstructor.reset()

        # 设置初始状态
        init_state_idx = min(episode_id % len(initial_states), len(initial_states) - 1)
        obs = env.set_init_state(initial_states[init_state_idx])

        # 回放动作序列
        replay_images = []
        # seg_images = []  # 🔥 新增：收集分割图
        success = False

        logging.info("开始回放动作序列...")


        # 等待环境稳定
        for t in range(args.num_steps_wait):
            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
            if done:
                success = True
                break

        # 执行记录的动作
        for step_idx, action in enumerate(tqdm.tqdm(actions, desc="执行动作")):

            # if step_idx == 0:
            #     print_scene_objects(env, step_idx)
            #     print_scene_bounds(env.sim.model, env.sim.data, reconstructor)
            #     print_plate_objects(env.sim.model, env.sim.data)
            # 获取当前观察图像
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])

            # 转换为uint8并保存
            if img.dtype != np.uint8:
                img = np.clip(img * 255, 0, 255).astype(np.uint8)

            replay_images.append(img)
            # # 🔥 收集并可视化分割图
            # seg = obs["agentview_segmentation_instance"].squeeze()
            # seg_vis = visualize_segmentation(seg, env)
            # seg_images.append(seg_vis)
            # if step_idx < 5:  # 前5步打印调试信息
            #     print(f"步骤 {step_idx}:")
            #     trace_action_conversion(action)

            # 转换动作格式
            libero_action = action_to_libero_action(action)
            # libero_action = high_precision_action_conversion(action)

            # 执行动作
            # obs, reward, done, info = env.step(libero_action)
            try:
                obs, reward, done, info = env.step(libero_action)
            except:
                # 如果不接受numpy数组，再转为list但保持精度
                libero_action_list = [float(x) for x in libero_action]
                obs, reward, done, info = env.step(libero_action_list)
            # debug_and_capture_voxel(reconstructor, env, 0.0, 0, debug_mode=False)
            timestamp = timestamps[step_idx] if step_idx < len(timestamps) else step_idx
            # integrate_voxel_reconstruction(reconstructor, env, timestamp, step_idx)
            reconstructor.capture_frame(obs, env)

            if done:
                success = True
                logging.info(f"任务在第 {step_idx + 1} 步完成！")
                break

        # 保存最后一帧
        if not success and len(replay_images) > 0:
            final_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            if final_img.dtype != np.uint8:
                final_img = np.clip(final_img * 255, 0, 255).astype(np.uint8)
            replay_images.append(final_img)
            # # 🔥 也保存最后的分割图
            # seg = obs["agentview_segmentation_instance"].squeeze()
            # seg_vis = visualize_segmentation(seg, env)
            # seg_images.append(seg_vis)

        # 保存视频
        if replay_images:
            suffix = "success" if success else "replay"
            safe_task_desc = task_description.replace(" ", "_").replace(",", "").replace(".", "")
            video_filename = f"ep_{episode_idx:03d}_task_{task_idx}_{safe_task_desc}_{suffix}.mp4"
            video_path = pathlib.Path(args.video_out_path) / video_filename

            logging.info(f"保存重放视频: {video_path}")
            logging.info(f"重放视频帧数: {len(replay_images)}")

            imageio.mimwrite(
                video_path,
                replay_images,
                fps=10,
                codec='libx264'
            )
            # 🔥 保存分割图视频
            # if seg_images:
            #     seg_video_path = pathlib.Path(args.video_out_path) / f"ep_{episode_idx:03d}_segmentation.mp4"
            #     logging.info(f"保存分割图视频: {seg_video_path}")
            #     imageio.mimwrite(seg_video_path, seg_images, fps=10, codec='libx264')

            # 保存原始视频用于对比
            if args.save_comparison_video and original_images:
                original_video_path = pathlib.Path(args.video_out_path) / f"ep_{episode_idx:03d}_original.mp4"
                logging.info(f"保存原始视频: {original_video_path}")

                # 确保原始图像尺寸正确
                processed_original = []
                for img in original_images:
                    if img.shape[:2] != (256, 256):
                        from PIL import Image as PILImage
                        img_pil = PILImage.fromarray(img)
                        img_pil = img_pil.resize((256, 256))
                        img = np.array(img_pil)
                    processed_original.append(img)

                imageio.mimwrite(
                    original_video_path,
                    processed_original,
                    fps=10,
                    codec='libx264'
                )

                # 创建并排对比视频
                if len(replay_images) > 0 and len(processed_original) > 0:
                    comparison_video_path = pathlib.Path(
                        args.video_out_path) / f"ep_{episode_idx:03d}_comparison.mp4"
                    create_comparison_video(processed_original, replay_images, comparison_video_path)

            logging.info(f"任务状态: {'成功' if success else '未完成'}")
            reconstructor.save_frames_as_json(args.video_out_path, episode_idx,env)

            summary = reconstructor.get_summary()
            print(f"重建摘要: {summary}")
        else:
            logging.warning("没有重放图像帧可保存")



    logging.info(f"\n=== 回放完成！所有视频已保存到: {args.video_out_path} ===")


def create_comparison_video(original_images, replay_images, output_path):
    """创建原始和重放的并排对比视频"""
    min_frames = min(len(original_images), len(replay_images))
    comparison_frames = []

    for i in range(min_frames):
        orig = original_images[i]
        replay = replay_images[i]

        # 确保两个图像尺寸相同
        if orig.shape != replay.shape:
            from PIL import Image as PILImage
            if orig.shape[:2] != (256, 256):
                orig_pil = PILImage.fromarray(orig)
                orig_pil = orig_pil.resize((256, 256))
                orig = np.array(orig_pil)

            if replay.shape[:2] != (256, 256):
                replay_pil = PILImage.fromarray(replay)
                replay_pil = replay_pil.resize((256, 256))
                replay = np.array(replay_pil)

        # 并排拼接
        combined = np.hstack([orig, replay])
        comparison_frames.append(combined)

    if comparison_frames:
        logging.info(f"保存对比视频: {output_path}")
        imageio.mimwrite(
            output_path,
            comparison_frames,
            fps=10,
            codec='libx264'
        )


# def _get_libero_env(task, resolution, seed):
#     """初始化LIBERO环境"""
#     task_description = task.language
#     task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
#     env_args = {
#         "bddl_file_name": task_bddl_file,
#         "camera_heights": resolution,
#         "camera_widths": resolution,
#         "camera_depths": True,  # 启用所有相机的深度
#     }
#     env = OffScreenRenderEnv(**env_args)
#     env.seed(seed)
#     return env, task_description

def _get_libero_env(task, resolution, seed):
    """初始化LIBERO环境，启用4视角深度"""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    # 显式指定4个相机
    camera_names = ["agentview", "robot0_eye_in_hand", "sideview", "frontview"]

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_names": camera_names,
        "camera_depths": True,  # 全部启用深度
        "camera_segmentations": "instance",  # 🔥 新增：启用instance分割
    }
    # env = OffScreenRenderEnv(**env_args)
    env = SegmentationRenderEnv(**env_args)
    env.seed(0)
    # === 调试：打印环境结构 ===
    # print("\n=== 调试环境属性 ===")
    # print(f"env类型: {type(env)}")
    # print(f"env属性: {[attr for attr in dir(env) if 'object' in attr.lower()]}")
    # if hasattr(env, 'env'):
    #     print(f"底层env属性: {[attr for attr in dir(env.env) if 'object' in attr.lower()]}")
    # if hasattr(env.env, 'model') and hasattr(env.env.model, 'instances_to_ids'):
    #     print("\n=== instances_to_ids ===")
    #     for i, (name, ids) in enumerate(env.env.model.instances_to_ids.items()):
    #         print(f"  Index {i}: {name} -> {ids}")
    #     print("========================\n")
    # ========================
    return env, task_description


def analyze_dataset_structure(args: ReplayArgs) -> None:
    """分析数据集结构"""
    dataset_path = pathlib.Path(args.dataset_path)
    data_dir = dataset_path / "data"

    episode_files = []
    for chunk_dir in sorted(data_dir.glob("chunk-*")):
        episodes = sorted(chunk_dir.glob("episode_*.parquet"))
        episode_files.extend(episodes)

    print(f"\n=== 数据集结构分析 ===")
    print(f"找到 {len(episode_files)} 个episodes")

    # 分析前几个episodes
    task_distribution = {}
    total_frames = 0

    for i, episode_file in enumerate(episode_files[:20]):  # 分析前20个
        df = pd.read_parquet(episode_file)
        task_idx = int(df['task_index'].iloc[0])
        episode_id = int(df['episode_index'].iloc[0])
        duration = float(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0])
        frame_count = len(df)
        total_frames += frame_count

        if task_idx not in task_distribution:
            task_distribution[task_idx] = 0
        task_distribution[task_idx] += 1

        print(f"Episode {i}: 文件={episode_file.name}, 任务={task_idx}, "
              f"Episode_ID={episode_id}, 帧数={frame_count}, 时长={duration:.2f}s")

    print(f"\n任务分布: {task_distribution}")
    print(f"平均帧数: {total_frames / min(20, len(episode_files)):.1f}")

    # 分析第一个episode的数据结构
    if episode_files:
        df = pd.read_parquet(episode_files[0])
        print(f"\n数据结构:")
        print(f"列: {list(df.columns)}")
        print(f"状态形状: {df['state'].iloc[0].shape}")
        print(f"动作形状: {df['actions'].iloc[0].shape}")

        # 检查图像格式
        img_data = df['image'].iloc[0]
        print(f"图像数据类型: {type(img_data)}")
        if isinstance(img_data, dict):
            print(f"图像字典键: {list(img_data.keys())}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        # 分析模式
        args = ReplayArgs()
        analyze_dataset_structure(args)
    else:
        # 回放模式
        tyro.cli(replay_libero_episodes)