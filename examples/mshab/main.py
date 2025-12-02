"""
用于在 mshab (ManiSkill) 环境中评估 Pi0 模型的脚本。

此脚本仿照 libero 的 main.py 评估脚本，通过 Websocket 连接到
一个正在运行的 Pi0 策略服务器 (serve_policy.py) 并计算成功率。

用法:
1. 启动 Pi0 模型服务器:
   (uv run) python serve_policy.py --policy.config <config_name> --policy.dir <checkpoint_dir>

2. 运行此评估脚本:
   (uv run) python mshab_eval_main.py \
       --mshab_env_config_path /path/to/your/mshab_env.yml \
       --prompt "open the fridge"
"""

import collections
import dataclasses
import logging
import math
import pathlib
from typing import Dict

import imageio
import numpy as np
import torch
import tqdm
import tyro

# OpenPI / Websocket 客户端
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# mshab (ManiSkill) 环境
from mshab.envs.make import EnvConfig, make_env

# 配置加载
from dacite import from_dict
from omegaconf import OmegaConf


@dataclasses.dataclass
class Args:
    #################################################################
    # Model server parameters
    #################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224  # 必须与 Pi0 训练时使用的 resize 匹配
    replan_steps: int = 5   # 每次重新规划的步数

    #################################################################
    # mshab 环境参数
    #################################################################
    mshab_env_config_path: str = "/raid/ljh/openpi/examples/mshab/config.yml"
    """
    指向 mshab EnvConfig 的 YAML 文件路径。
    此环境配置将用于评估。
    """

    prompt: str = "close the kitchen counter"
    """
    提供给 Pi0 模型的自然语言指令。
    必须与 'mshab_env_config_path' 中的任务相匹配。
    """
    
    num_eval_episodes: int = 100  # 每个任务的 rollout 次数

    #################################################################
    # Utils
    #################################################################
    video_out_path: str = "data/mshab/videos"  # 保存视频的路径
    seed: int = 7  # 随机种子 (用于可复现性)


def _get_mshab_state(obs: Dict) -> np.ndarray:
    """
    从 mshab 观测字典中提取并拼接状态向量。
    """
    obs_agent = obs["agent"]
    obs_extra = obs["extra"]

    # 1. 定义哪些键被包含在 'state' 中
    agent_keys = sorted([k for k in obs_agent.keys() if k != "qvel"])
    extra_keys = sorted([
        k
        for k in obs_extra.keys()
        # if k not in ["obj_pose_wrt_base", "is_grasped"]
    ])

    state_parts = []
    for k in agent_keys:
        val = obs_agent[k][0].cpu().numpy()
        state_parts.append(val.flatten())
    
    for k in extra_keys:
        val = obs_extra[k][0].cpu().numpy()
        state_parts.append(val.flatten())

    # 3. 拼接
    return np.concatenate(state_parts)


def eval_mshab(args: Args) -> None:
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 准备视频保存路径
    video_path = pathlib.Path(args.video_out_path)
    video_path.mkdir(parents=True, exist_ok=True)

    # 1. 初始化 mshab 环境
    logging.info(f"加载 mshab 环境配置: {args.mshab_env_config_path}")
    try:
        env_cfg_dict = OmegaConf.to_container(OmegaConf.load(args.mshab_env_config_path))
        env_cfg = from_dict(data_class=EnvConfig, data=env_cfg_dict)
    except Exception as e:
        logging.error(f"加载环境配置失败: {e}")
        logging.error("请确保路径正确，且文件是有效的 EnvConfig YAML。")
        return

    # 评估时我们使用非并行的单个环境
    env_cfg.num_envs = 1 

    logging.info("正在创建 mshab 环境...")
    env = make_env(env_cfg, str(video_path))
    logging.info("环境创建成功。")

    max_steps = env.max_episode_steps

    # 2. 连接到 Pi0 策略服务器
    logging.info(f"正在连接到策略服务器 {args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    logging.info("连接成功。")

    # 3. 开始评估
    total_episodes, total_successes, total_successes_at_end = 0, 0, 0
    
    for episode_idx in tqdm.tqdm(range(args.num_eval_episodes), desc="评估 Trials"):
        logging.info(f"\n任务: {args.prompt}")

        # 重置环境
        # 为 mshab/ManiSkill 设置每个 trial 的种子
        obs, _ = env.reset(seed=args.seed + episode_idx)
        action_plan = collections.deque()

        t = 0

        logging.info(f"开始 episode {episode_idx+1}...")
        
        # 内部循环（单个 episode）
        success_at_once = False
        while t < max_steps:
            try:
                # 1. 获取并处理观测
                # a) 图像
                img_np = obs["fetch_head_rgb"][0].cpu().numpy()
                wrist_img_np = obs["fetch_hand_rgb"][0].cpu().numpy()
                
                # 缩放图像以匹配 Pi0 输入
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img_np, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img_np, args.resize_size, args.resize_size)
                )
                
                # b) 状态
                state_vector = _get_mshab_state(obs)

                # print(img.shape)
                # print(wrist_img.shape)
                # print(state_vector.shape)
                
                # 2. 获取动作 (重新规划)
                if not action_plan:
                    # 准备发送给 Pi0 模型的 'element'
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": state_vector,
                        "prompt": str(args.prompt),
                    }
                    
                    # 查询模型
                    action_chunk = client.infer(element)["actions"]
                    
                    assert (
                        len(action_chunk) >= args.replan_steps
                    ), f"模型预测 {len(action_chunk)} 步, 但需要 {args.replan_steps} 步。"
                    action_plan.extend(action_chunk[: args.replan_steps])
                    
                action = action_plan.popleft()
                
                # 3. 执行动作
                action_for_env = torch.from_numpy(action.copy()).unsqueeze(0) # (1, 13)
                
                obs, reward, terminated, truncated, info = env.step(action_for_env)
                
                if success_at_once == False:
                    success_at_once = info.get("success", [False])[0]

                # 4. 检查是否结束
                done = terminated[0] or truncated[0]
                if done:

                    success_at_end = info.get("success", [False])[0]
                    
                    if success_at_once:
                        total_successes += 1

                    if success_at_end:
                        total_successes_at_end += 1
                    
                    logging.info(f"Episode 结束。成功: {success_at_once}")
                    break
                
                t += 1

            except Exception as e:
                logging.error(f"评估 trial 中捕获到异常: {e}")
                break # 结束这个 trial

        # --- End of while loop ---

        total_episodes += 1

        # 记录当前结果
        logging.info(f"# episodes 完成: {total_episodes}")
        logging.info(f"# 成功次数: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
        logging.info(f"# success_at_end成功次数: {total_successes_at_end} ({total_successes_at_end / total_episodes * 100:.1f}%)")

    # --- End of for loop ---

    env.close()
    logging.info("--- 评估完成 ---")
    logging.info(f"总成功率: {float(total_successes) / float(total_episodes) * 100:.1f}%")
    logging.info(f"success_at_end成功次数总成功率: {float(total_successes_at_end) / float(total_episodes) * 100:.1f}%")
    logging.info(f"总 episodes: {total_episodes}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_mshab)