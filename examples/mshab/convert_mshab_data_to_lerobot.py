"""
用于将 mshab HDF5 数据集转换为 LeRobot 格式的脚本。

用法:
uv run convert_mshab_data_to_lerobot.py --data_dir /path/to/your/mshab-data-directory --repo_name mshab_lerobot_of_open_the_fridge
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple

import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ##################################################################
# 1. 自动检测 mshab 数据集元数据
# ##################################################################

def get_mshab_metadata(data_dir: Path) -> Dict:
    """
    通过查看第一个HDF5文件来自动检测数据集的元数据（形状信息）。
    """
    print(f"正在从 {data_dir} 检测元数据...")

    # 找到第一个 .h5 和 .json 文件
    h5_files = sorted(list(data_dir.glob("*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"在 {data_dir} 中未找到 .h5 文件。")

    h5_file_path = h5_files[0]
    json_file_path = h5_file_path.with_suffix(".json")
    if not json_file_path.exists():
        raise FileNotFoundError(f"找不到对应的 .json 文件: {json_file_path}")

    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    with h5py.File(h5_file_path, "r") as f:
        # 获取第一个有效的 episode ID
        if not json_data["episodes"]:
            raise ValueError(f"JSON 文件 {json_file_path} 不包含任何 episode。")
        first_ep_id = json_data["episodes"][0]["episode_id"]
        ep_data = f[f"traj_{first_ep_id}"]

        # 1. 获取动作形状 (actions)
        # 假设动作在第一个时间步的数据
        action_sample = ep_data["actions"][0]
        action_shape = action_sample.shape

        # 2. 获取图像形状 (image, wrist_image, depth)
        head_rgb_sample = ep_data["obs"]["sensor_data"]["fetch_head"]["rgb"][0]
        hand_rgb_sample = ep_data["obs"]["sensor_data"]["fetch_hand"]["rgb"][0]
        head_depth_sample = ep_data["obs"]["sensor_data"]["fetch_head"]["depth"][0]
        hand_depth_sample = ep_data["obs"]["sensor_data"]["fetch_hand"]["depth"][0]

        # 3. 获取 state 形状 (state)
        obs_agent = ep_data["obs"]["agent"]
        obs_extra = ep_data["obs"]["extra"]

        # 这些是 BCDataset 中被拼接的键
        # (排除了 'qvel', 'obj_pose_wrt_base', 'is_grasped')
        agent_keys = sorted([k for k in obs_agent.keys() 
                             if k != "qvel"
                             ])
        extra_keys = sorted([
            k
            for k in obs_extra.keys()
            # if k not in ["obj_pose_wrt_base", "is_grasped"]
        ])

        state_parts = []
        for k in agent_keys:
            state_parts.append(np.array(obs_agent[k][0]).flatten())
        for k in extra_keys:
            state_parts.append(np.array(obs_extra[k][0]).flatten())

        state_sample = np.concatenate(state_parts)
        state_shape = state_sample.shape
        
        metadata = {
            "action_shape": action_shape,
            "state_shape": state_shape,
            "head_rgb_shape": head_rgb_sample.shape,
            "hand_rgb_shape": hand_rgb_sample.shape,
            "head_depth_shape": head_depth_sample.shape,
            "hand_depth_shape": hand_depth_sample.shape,
            "agent_keys": agent_keys,
            "extra_keys": extra_keys,
        }

        print("元数据检测完成:")
        print(json.dumps(metadata, indent=2, default=lambda x: str(x)))
        return metadata


# ##################################################################
# 2. LeRobot 数据集转换主函数
# ##################################################################

def main(
    data_dir: str,
    repo_name: str = "mshab_lerobot_of_open_the_fridge", # 包含任务名称
    *,
    push_to_hub: bool = False,
):
    """
    将 mshab HDF5 数据集转换为 LeRobot 格式。

    Args:
        data_dir: 包含 .h5 和 .json 文件的 mshab 数据目录。
        repo_name: 输出数据集的名称。
                    我们期望格式为 '..._of_TASK_NAME'，
                    例如: 'mshab_lerobot_of_open_the_fridge'。
        push_to_hub: 是否将数据集推送到 Hugging Face Hub。
    """
    data_dir_path = Path(data_dir)

    # 1. 清理已存在的输出目录
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"警告：输出目录 {output_path} 已存在。正在删除...")
        shutil.rmtree(output_path)
    print(f"数据集将保存到: {output_path}")

    # 2. 自动检测数据形状
    try:
        metadata = get_mshab_metadata(data_dir_path)
    except Exception as e:
        print(f"元数据检测失败: {e}")
        print("请确保 data_dir 指向包含 .h5 和 .json 文件的目录。")
        return

    # 3. 从 repo_name 解析任务指令
    parts = repo_name.split("_of_")
    if len(parts) > 1:
        task_keyword = parts[-1]  # 例如: "open_the_fridge"
        # 转换为自然语言
        task_instruction = task_keyword.replace("_", " ") # 例如: "open the fridge"
        print(f"INFO: 已从 'repo_name' 解析任务: '{task_instruction}'")
    else:
        task_instruction = "perform the task" # 通用占位符
        print(f"警告: 'repo_name' ({repo_name}) 不包含 '_of_' 分隔符。")
        print(f"INFO: 正在使用通用占位符指令: '{task_instruction}'")
    # ================================================

    # 4. 创建 LeRobot 数据集
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="fetch",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": metadata["head_rgb_shape"],
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": metadata["hand_rgb_shape"],
                "names": ["height", "width", "channel"],
            },
            # "head_depth": {
            #     "dtype": "float32",
            #     "shape": metadata["head_depth_shape"],
            #     "names": ["height", "width", "channel"],
            # },
            # "hand_depth": {
            #     "dtype": "float32",
            #     "shape": metadata["hand_depth_shape"],
            #     "names": ["height", "width", "channel"],
            # },
            "state": {
                "dtype": "float32",
                "shape": metadata["state_shape"], # 22 or 30
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": metadata["action_shape"], # 13
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # 5. 循环遍历原始 mshab 数据并写入 LeRobot 数据集
    h5_files = sorted(list(data_dir_path.glob("*.h5")))
    total_eps = 0

    for h5_file_path in h5_files:
        json_file_path = h5_file_path.with_suffix(".json")
        if not json_file_path.exists():
            print(f"跳过 {h5_file_path.name}: 找不到 .json 文件。")
            continue

        print(f"--- 正在处理文件: {h5_file_path.name} ---")
        
        with open(json_file_path, "r") as f:
            json_data = json.load(f)

        with h5py.File(h5_file_path, "r") as h5_file:
            for ep_json in json_data["episodes"]:
                ep_id = ep_json["episode_id"]
                ep_data = h5_file[f"traj_{ep_id}"]
                num_steps = ep_json["elapsed_steps"]
                success = ep_data["success"][:].tolist()
                num_steps = min(success.index(True) + 1, num_steps)
                
                # 提取数据
                all_actions = ep_data["actions"][:num_steps]
                all_obs = ep_data["obs"]

                for step_idx in range(num_steps):
                    # a) 拼接 'state'
                    state_parts = []
                    for k in metadata["agent_keys"]:
                        state_parts.append(np.array(all_obs["agent"][k][step_idx]).flatten())
                    for k in metadata["extra_keys"]:
                        state_parts.append(np.array(all_obs["extra"][k][step_idx]).flatten())
                    state = np.concatenate(state_parts)
                    
                    # b) 获取图像 (HWC格式, numpy)
                    image = all_obs["sensor_data"]["fetch_head"]["rgb"][step_idx].astype(np.uint8)
                    wrist_image = all_obs["sensor_data"]["fetch_hand"]["rgb"][step_idx].astype(np.uint8)
                    # head_depth = all_obs["sensor_data"]["fetch_head"]["depth"][step_idx].astype(np.float32)
                    # hand_depth = all_obs["sensor_data"]["fetch_hand"]["depth"][step_idx].astype(np.float32)
             
                    # c) 获取动作
                    action = all_actions[step_idx]

                    # d) 构建并添加帧
                    frame_data = {
                        "image": image,
                        "wrist_image": wrist_image,
                        # "head_depth": head_depth,
                        # "hand_depth": hand_depth,
                        "state": state,
                        "actions": action,
                        "task": task_instruction, 
                    }
                    dataset.add_frame(frame_data)

                # e) 保存 episode
                dataset.save_episode()
                total_eps += 1

    print(f"\n--- 转换完成 ---")
    print(f"总共处理了 {total_eps} 个 episodes。")
    print(f"数据集已保存到: {output_path}")

    # 6. (可选) 推送到 Hub
    if push_to_hub:
        print("正在推送到 Hugging Face Hub...")
        dataset.push_to_hub(
            tags=["mshab", "fetch", "h5", task_keyword], # 添加任务关键词作为标签
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("推送完成。")


if __name__ == "__main__":
    # 使用 tyro.cli 来解析参数，
    # 可以从命令行设置 --repo_name
    tyro.cli(main)