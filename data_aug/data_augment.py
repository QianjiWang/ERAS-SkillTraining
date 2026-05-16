import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# 图像增强参数范围配置
ENHANCE_CONFIG = {
    'brightness': (-50, 50),       # 亮度调整范围
    'contrast': (0.5, 1.5),         # 对比度调整范围
    'saturation': (0.5, 1.5),       # 饱和度调整范围  
    'hue': (-10, 10)               # 色调调整范围
}

def apply_augmentation(image, params):
    """应用图像增强"""
    # 调整亮度
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=params['brightness'])
    # 调整对比度
    image = cv2.convertScaleAbs(image, alpha=params['contrast'], beta=0)
    # 调整饱和度和色调
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= params['saturation']
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 0] += params['hue']
    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)  # 色调范围 0-179
    image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image

def generate_aug_params():
    """生成随机增强参数组合"""
    return {
        'brightness': np.random.uniform(*ENHANCE_CONFIG['brightness']),
        'contrast': np.random.uniform(*ENHANCE_CONFIG['contrast']),
        'saturation': np.random.uniform(*ENHANCE_CONFIG['saturation']),
        'hue': np.random.uniform(*ENHANCE_CONFIG['hue'])
    }

def main(dataset_path):
    # 路径配置
    paths = {
        'episode_ends': os.path.join(dataset_path, "episode_ends.csv"),
        'agent_obs': os.path.join(dataset_path, "agent_obs.csv"),
        'action': os.path.join(dataset_path, "action.csv"),
        'reward': os.path.join(dataset_path, "reward.csv"),
        'task_name': os.path.join(dataset_path, "task_name.csv"),
        'img_obs_1': os.path.join(dataset_path, "img_obs_1"),
        'img_obs_2': os.path.join(dataset_path, "img_obs_2")
    }

    # 加载原始数据（只加载一次，不包括后续增强数据）
    orig_episode_ends = pd.read_csv(paths['episode_ends'], header=None)[0].tolist()
    agent_obs = pd.read_csv(paths['agent_obs'], header=None)
    actions = pd.read_csv(paths['action'], header=None)
    rewards = pd.read_csv(paths['reward'], header=None)
    task_names = pd.read_csv(paths['task_name'], header=None)

    # 创建图像存储目录（图片会保存到相同的文件夹中，新数据不会覆盖原始数据）
    for folder in [paths['img_obs_1'], paths['img_obs_2']]:
        os.makedirs(folder, exist_ok=True)

    # 用于新数据索引的起始值
    current_max_idx = orig_episode_ends[-1] if orig_episode_ends else 0
    # episode_ends 最终会包含原始 episode 与增强生成的 episode
    new_episode_ends = orig_episode_ends.copy()

    # 只对原始数据扩充 9 轮
    for aug_round in tqdm(range(9), desc="总扩充轮次"):
        # 遍历每个原始 episode（保持 episode_ends 不更新，作为索引参考）
        for ep_idx in tqdm(range(len(orig_episode_ends)), desc="处理episode", leave=False):
            start = orig_episode_ends[ep_idx-1] if ep_idx > 0 else 0
            end = orig_episode_ends[ep_idx]
            start = int(start)
            end = int(end)
            episode_length = int(end - start)

            params = generate_aug_params()

            new_agent_obs = []
            new_actions = []
            new_rewards = []
            new_task_names = []

            for step in tqdm(range(episode_length), desc="处理steps", leave=False):
                original_idx = int(start + step)
                new_idx = int(current_max_idx + step)

                # 加载并增强图像
                for cam in ['img_obs_1', 'img_obs_2']:
                    img_path = os.path.join(paths[cam], f"{cam}_{original_idx:012d}.png")
                    img = cv2.imread(img_path)
                    if img is None:
                        raise FileNotFoundError(f"图像缺失: {img_path}")
                    aug_img = apply_augmentation(img, params)
                    new_img_path = os.path.join(paths[cam], f"{cam}_{new_idx:012d}.png")
                    cv2.imwrite(new_img_path, aug_img)

                # 复制非图像数据（直接拷贝原始数据）
                new_agent_obs.append(agent_obs.iloc[original_idx].values)
                new_actions.append(actions.iloc[original_idx].values)
                new_rewards.append(rewards.iloc[original_idx].values)
                new_task_names.append(task_names.iloc[original_idx].values)

            # 追加新数据到 CSV 文件
            pd.DataFrame(new_agent_obs).to_csv(paths['agent_obs'], mode='a', header=False, index=False)
            pd.DataFrame(new_actions).to_csv(paths['action'], mode='a', header=False, index=False)
            pd.DataFrame(new_rewards).to_csv(paths['reward'], mode='a', header=False, index=False)
            pd.DataFrame(new_task_names).to_csv(paths['task_name'], mode='a', header=False, index=False)

            current_max_idx += episode_length
            new_episode_ends.append(current_max_idx)

    # 保存最终扩充后的 episode_ends
    pd.DataFrame(new_episode_ends).to_csv(paths['episode_ends'], index=False, header=False)

if __name__ == "__main__":
    dataset_path = "./_PolicyProject3_PGAS/_trained_models_env1/_teach1_dataset_amplify1"
    main(dataset_path)
    print("数据集增强完成！总数据量扩充至原始数据的10倍")
