import argparse
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from config import LIBERO_FEATURES
from libero_utils import load_local_episodes


def main(
        src_path: Path,
        output_path: Path,
):
    # 找到第一个HDF5文件
    input_h5 = None
    pattern1 = re.compile(r"_SCENE\d+_(.*?)_demo\.hdf5")
    pattern2 = re.compile(r"(.*?)_demo\.hdf5")

    for h5_file in src_path.glob("*.hdf5"):
        match = pattern1.search(h5_file.name)
        if match is None:
            match = pattern2.search(h5_file.name)
        if match:
            input_h5 = h5_file
            task_instruction = match.group(1).replace("_", " ")
            break

    if input_h5 is None:
        raise ValueError(f"No HDF5 file found in {src_path}")

    output_dir = (output_path / input_h5.stem).resolve()

    if output_dir.exists():
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=f"{src_path.name}/{input_h5.name}",
        root=output_dir,
        fps=10,
        robot_type="panda",
        features=LIBERO_FEATURES,
    )

    print(f"Processing {input_h5}")

    raw_dataset = load_local_episodes(input_h5)

    # 只处理第一个episode
    episode_data = next(raw_dataset)

    for frame_data in episode_data:
        dataset.add_frame(
            frame_data,
            task=task_instruction,
        )

    dataset.save_episode()

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    main(**vars(args))