# %%
#@markdown ### **Imports**
# file import
import assemble_env_1
import DiffusionPolicy_Networks
import math
import utils
# module import
from random import uniform,choice
import numpy as np
import time
import os
import pandas as pd
import cv2
import zipfile
import torch
from torch import nn
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import keyboard
from IPython.display import display, clear_output, Javascript
"记录标记!"
dataset_path =           (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset_amplify1")
MinMax_dir_path =        (".\\_PolicyProject3_PGAS\\_trained_models_env1\\MinMax")

ckpt_path =              (".\\_PolicyProject3_PGAS\\env1_train对比\\9_RDT_ResNet50\\model.ckpt")
output_dir =             (".\\_PolicyProject3_PGAS\\env1_train对比\\9_RDT_ResNet50")
training_loss_csv_path = (".\\_PolicyProject3_PGAS\\env1_train对比\\9_RDT_ResNet50\\loss.csv")
max_iterations = 50000  # 目标最大迭代次数

info_path = (".\\_PolicyProject3_PGAS\\env1_train对比\\_test\\9_RDT_ResNet50\\info.csv")
from utils_file import try_to_csv, try_read_csv
"记录标记!"

img_height = [256,256]
img_width = [256,256]
img_channel = 3

agent_obs_dim = 12
action_dim = 7

pred_horizon = 8
obs_horizon = 1
action_horizon = 4



# %%
# device transfer
device = torch.device('cuda')
print("示教数据文件夹的路径:\t", dataset_path)

# %%
#@markdown ### **Dataset**
#@markdown
#@markdown Defines `PolicyProject1Dataset` and helper functions
#@markdown
#@markdown The dataset class
#@markdown - Load data (image, action) from `dataset_path`
#@markdown - Returns
#@markdown  - All possible segments with length `pred_horizon`
#@markdown  - Pads the beginning and the end of each episode with repetition
#@markdown  - key `image`: shape (obs_hoirzon, 3, 512, 512)
#@markdown  - key `action`: shape (pred_horizon, 6)

img_height
img_width
img_channel

agent_obs_dim
action_dim

def create_sample_indices(
        episode_ends:np.ndarray, sequence_length:int,
        pad_before: int=0, pad_after: int=0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats, eps=1e-8):
    range_ = stats['max'] - stats['min'] # 归一化到 [0,1]
    range_[range_ < eps] = 1 # 当 range_ 小于 eps 时，视为常数 0
    ndata = (data - stats['min']) / range_ 
    ndata = ndata * 2 - 1 # 归一化到 [-1, 1]
    return ndata

def unnormalize_data(ndata, stats, eps=1e-8):
    ndata = (ndata + 1) / 2 # 反归一化回原始范围
    range_ = stats['max'] - stats['min']
    data = ndata * range_ + stats['min']
    return data

def normalize_data_byValue(data, max, min, eps=1e-8):
    range_ = max - min # 归一化到 [0,1]
    range_[range_ < eps] = 1 # 当 range_ 小于 eps 时，视为常数 0
    ndata = (data - min) / range_ 
    ndata = ndata * 2 - 1 # 归一化到 [-1, 1]
    return ndata

def unnormalize_data_byValue(ndata, max, min, eps=1e-8):
    ndata = (ndata + 1) / 2 # 反归一化回原始范围
    range_ = max - min
    data = ndata * range_ + min
    return data



# dataset
class PolicyProject1Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int):
        
        self.dataset_path = dataset_path
        
        """
        # Read image files
        images_paths = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs")))
        train_image_data = []
        for img_name in images_paths:
            img_path = os.path.join(self.dataset_path, "img_obs", img_name)
            img = Image.open(img_path).convert('RGB')
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
            img = img.transpose((2, 0, 1))  # Change the channel dimension to be the first one
            train_image_data.append(img)
        train_image_data = np.stack(train_image_data)
        train_image_data = train_image_data.astype(np.float32)
        print("train_image_data的形状及性质为:", train_image_data.shape, train_image_data.dtype)
        """
        # Store image paths  
        self.image_paths_1 = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs_1")))
        self.image_paths_2 = sorted(os.listdir(os.path.join(self.dataset_path, "img_obs_2")))
        
        
        # Read CSV files
        agent_obs_csv = pd.read_csv(os.path.join(self.dataset_path, "agent_obs.csv"), header=None)
        action_csv = pd.read_csv(os.path.join(self.dataset_path, "action.csv"), header=None)
        episode_ends_csv = pd.read_csv(os.path.join(self.dataset_path, "episode_ends.csv"), header=None)

        # Remove the first row with indices
        agent_obs_csv = agent_obs_csv.iloc[0:]
        action_csv = action_csv.iloc[0:]
        episode_ends_csv = episode_ends_csv.iloc[0:]

        # Convert DataFrame to NumPy arrays and reshape them
        agent_obs = agent_obs_csv.to_numpy().astype(np.float32).reshape(-1, agent_obs_dim)
        action = action_csv.to_numpy().astype(np.float32).reshape(-1, action_dim)
        episode_ends = episode_ends_csv.to_numpy().astype(int).reshape(-1) #episode_ends需要是个1维数组
        print("episode_ends的形状及性质为:", episode_ends.shape, episode_ends.dtype)
        
        train_data = {
            'agent_obs': agent_obs,
            # (N, 6)
            'action': action
            # (N, 6)
        }
        print("agent_obs的形状及性质为:", agent_obs.shape, agent_obs.dtype)
        print("action的形状及性质为:", action.shape, action.dtype)
        
        """重写读取：结束"""

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon-1,
            pad_after=action_horizon-1)

        # compute statistics and normalized data to [-1,1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = get_data_stats(data)
            normalized_train_data[key] = normalize_data(data, stats[key])
            print(key, "的最小:\n", stats[key]['min'], "\n的最大:\n", stats[key]['max'])
            try_to_csv(MinMax_dir_path+"\\"+key+".csv", pd.DataFrame([stats[key]['min'],stats[key]['max']]), info=key+"的MinMax", isPrintInfo=True)
            
        # images are already normalized
        """
        normalized_train_data['image'] = train_image_data
        """

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        buffer_start_idx, buffer_end_idx, \
            sample_start_idx, sample_end_idx = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # discard unused observations
        """
        nsample['image'] = nsample['image'][:self.obs_horizon,:]
        """
        
        """新增模块: 图像实时采样读取,而非预先加载所有的把内存爆了"""
        # Load images on-the-fly  
        nsample['image_1'] = np.zeros((self.obs_horizon, 3, img_height[0], img_width[0]), dtype=np.float32)  
        for i in range(self.obs_horizon):
            # print([buffer_start_idx, buffer_end_idx], [sample_start_idx, sample_end_idx])
            idx = max(min(i, sample_end_idx-1), sample_start_idx) - sample_start_idx
            img_path = os.path.join(self.dataset_path, "img_obs_1", self.image_paths_1[buffer_start_idx + idx])  
            img = Image.open(img_path).convert('RGB')  
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]  
            img = img.transpose((2, 0, 1))  # Change channel dimension to be first  
            nsample['image_1'][i] = img
        
        nsample['image_2'] = np.zeros((self.obs_horizon, 3, img_height[1], img_width[1]), dtype=np.float32)  
        for i in range(self.obs_horizon):
            # print([buffer_start_idx, buffer_end_idx], [sample_start_idx, sample_end_idx])
            idx = max(min(i, sample_end_idx-1), sample_start_idx) - sample_start_idx
            img_path = os.path.join(self.dataset_path, "img_obs_2", self.image_paths_2[buffer_start_idx + idx])  
            img = Image.open(img_path).convert('RGB')  
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]  
            img = img.transpose((2, 0, 1))  # Change channel dimension to be first  
            nsample['image_2'][i] = img
        
        nsample['agent_obs'] = nsample['agent_obs'][:self.obs_horizon,:]
        
        # print(nsample['image'].shape[0], nsample['action'].shape[0])
        # flag11 = np.array_equal(nsample['image'][0],nsample['image'][1])
        # flag12 = np.array_equal(nsample['agent_obs'][0],nsample['agent_obs'][1])
        # flag21 = np.array_equal(nsample['image'][-1],nsample['image'][-2])
        # flag22 = np.array_equal(nsample['agent_obs'][-1],nsample['agent_obs'][-2])
        # print("nsample测试1:\n", flag11, flag12)
        # print("nsample测试2:\n", flag21, flag22)
        # print("nsample测试3:\t", flag11 == flag12)
        # print("nsample测试4:\t", flag21 == flag22)
           
        return nsample
    


#@markdown ### **Dataset Demo**

# parameters
pred_horizon
obs_horizon
action_horizon
#|o|o|                         observations: 2
#| |a|a|a|a|a|a|               actions executed: 6
#|p|p|p|p|p|p|p|p|p|p|p|p|     actions predicted: 12

"设置是否花时间加载数据集，还是说已知数据集的最大值和最小值了"
is_load_dataset = False

if is_load_dataset:
    # create dataset from file
    dataset = PolicyProject1Dataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    # import torch.utils.data
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        # num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        # persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['image'].shape:", batch['image'].shape)
    print("batch['action'].shape", batch['action'].shape)
else:
    action_min = try_read_csv(MinMax_dir_path+"\\action.csv", info="action_MinMax", header=None).iloc[0:].to_numpy().reshape(2,-1)[0]
    action_max = try_read_csv(MinMax_dir_path+"\\action.csv", info="action_MinMax", header=None).iloc[0:].to_numpy().reshape(2,-1)[1]
    print("action_min:", action_min)
    print("action_max:", action_max)
    



import DiffusionPolicy_Networks as nets
import DiffusionPolicy_Networks_RDT as nets_RDT
import DiffusionPolicy_Networks_VE1 as nets_VE1


#@markdown ### **Network Demo**

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
# vision_encoder_1 = nets.get_resnet_with_attention('resnet18')
# vision_encoder_2 = nets.get_resnet_with_attention('resnet18')
vision_encoder_1 = nets_VE1.ResNet50VisionEncoder()
vision_encoder_2 = nets_VE1.ResNet50VisionEncoder()
print("视觉编码器1的形状:\n", vision_encoder_1)
print("视觉编码器2的形状:\n", vision_encoder_2)

# ResNet18 has output dim of 512
vision_feature_dim = 512
obs_dim = vision_feature_dim + 0

# create network object
# noise_pred_net = nets.ConditionalUnet1D(
#     input_dim=action_dim,
#     global_cond_dim=obs_dim*2*obs_horizon
# )
noise_pred_net = nets_RDT.RDTForDP(
    input_dim=action_dim,
    global_cond_dim=obs_dim*2*obs_horizon
)
print("噪声预测网络形状:\n", noise_pred_net)

# the final arch has 2 parts
nets = nn.ModuleDict({
    'vision_encoder_1': vision_encoder_1,
    'vision_encoder_2': vision_encoder_2,
    'noise_pred_net': noise_pred_net
})

print("number of all parameters: {:e}".format(
    sum(p.numel() for p in nets['vision_encoder_1'].parameters())+
    sum(p.numel() for p in nets['vision_encoder_2'].parameters())+
    sum(p.numel() for p in nets['noise_pred_net'].parameters()))
)

# demo
with torch.no_grad():
    # example inputs
    image_1 = torch.zeros((1, obs_horizon,3,img_height[0],img_width[0]))
    image_2 = torch.zeros((1, obs_horizon,3,img_height[1],img_width[1]))
    # agent_obs = torch.zeros((1, obs_horizon, 6))
    # vision encoder
    image_features_1 = nets['vision_encoder_1'](
        image_1.flatten(end_dim=1))
    # (2,512)
    image_features_1 = image_features_1.reshape(*image_1.shape[:2],-1)
    # (1,2,512)
    image_features_2 = nets['vision_encoder_2'](image_2.flatten(end_dim=1))
    image_features_2 = image_features_2.reshape(*image_2.shape[:2],-1)
    # obs = torch.cat([image_features,agent_obs],dim=-1)
    obs = torch.cat([image_features_1, image_features_2],dim=-1)
    print("obs.shape:\t", obs.shape)
    print("obs.flatten(start_dim=1).shape:\t",obs.flatten(start_dim=1).shape)
    noised_action = torch.randn((1, pred_horizon, action_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = nets['noise_pred_net'](
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
# device = torch.device('cuda')
_ = nets.to(device)



#@markdown ### **load pretrained weights**
load_pretrained = True
if load_pretrained:
    if os.path.isfile(ckpt_path):
        # 加载检查点文件
        state_dict = torch.load(ckpt_path, map_location='cuda')
        # 从字典中提取模型状态
        model_state_dict = state_dict['model_state_dict']
        # 加载模型状态
        ema_nets = nets
        ema_nets.load_state_dict(model_state_dict)
        print('Pretrained weights loaded.')
    else:
        print("No pretrained weights found. Training from scratch.")
else:
    print("Skipped pretrained weight loading.")
    
    

#@markdown ### **save model**

def save_model(num_epochs, optimizer, lr_scheduler, ema):
    # Save the trained model, optimizer, and scheduler states
    os.makedirs(output_dir, exist_ok=True)

    # Define the checkpoint filename
    checkpoint_filename = ckpt_path

    # Create a dictionary containing all necessary states
    state_dict = {
        'epoch': num_epochs,
        'model_state_dict': nets.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': lr_scheduler.state_dict(),
        'ema_state_dict': ema.state_dict(),
    }

    # Save the checkpoint
    torch.save(state_dict, checkpoint_filename)

    print(f'Trained model saved to {checkpoint_filename}')
    
# save_model()





# %%
#@markdown ### **Inference**
#@markdown

env = assemble_env_1.AssembleEnv(img_obs_height=img_height, img_obs_width=img_width, img_obs_channel=img_channel,
                                agent_obs_dim=agent_obs_dim, action_dim=action_dim,
                                obs_horizon=obs_horizon, action_horizon=action_horizon, pred_horizon=pred_horizon)
env.rob.RTDE_SOFT_F_THRESHOLD = 25
env.rob.RTDE_SOFT_RETURN = 0.004

"记录标记!"
for episode in range(100):
    "记录标记!"
    step = 1
    # Calculate angle for this episode (0-360 degrees in 36° increments)
    angle_deg = (episode % 10) * 36
    angle_rad = np.deg2rad(angle_deg)
    # Calculate position offset (1cm radius)
    pos_offset = 0.002  # cm in meters
    x_offset = pos_offset * np.cos(angle_rad)
    y_offset = pos_offset * np.sin(angle_rad)
    # Set rotation (first 10 episodes: +°, next 10: -°)
    if episode < 10:
        z_rotation = np.deg2rad(3)  # +° in radians
    else:
        z_rotation = np.deg2rad(-3)  # -° in radians
    
    print("重置.......")
    
    time.sleep(0.5)
    isPrintStepInfo = False
    isUseRandomEnv =True
    
    env.reset(isAddError=False, is_drag_mode=False)
    env.rob.moveIK_changePosOrt_onEnd(d_pos=[0,0,-0.024], d_ort=[0,0,0], threshold_time=0.5, wait=True)
    if isUseRandomEnv:
        "记录标记!"
        # Create initial position/orientation array with offsets
        po = np.array([
            env.initial_pos_ort[0] + x_offset,  # X position with offset
            env.initial_pos_ort[1] + y_offset,  # Y position with offset
            env.initial_pos_ort[2],             # Z position (no change)
            env.initial_pos_ort[3],                    # RX (no change)
            env.initial_pos_ort[4],             # RY (no change)
            env.initial_pos_ort[5] + z_rotation # RZ with our rotation offset
        ])
        env.rob.moveIK(po[0:3], po[3:6], wait=True)
    #     env.reset(isAddError=True, is_drag_mode=False, error_scale_pos=3, error_scale_ort=1.5)
    # else:
    #     env.reset(isAddError=False, is_drag_mode=False)
    
    "记录标记!"
    # print("按 s 开始本回合")
    # while True:
    #     env._get_obs(isPrintInfo=False)
    #     if keyboard.is_pressed("s"): break
    
    pred_steps = 1000
    for ps in range(pred_steps):
        print("按 x 进行紧急停止运动")
        if keyboard.is_pressed("x"): break
        print("---------------------------------------------------------------------")
        # env.obs = env._get_obs()
        
        """--------------------------------------------------"""
        """核心: get action"""
        B = 1
        # stack the last obs_horizon number of observations
        img_obs_1_list = np.stack([x for x in env.img_obs_1_list])
        img_obs_2_list = np.stack([x for x in env.img_obs_2_list])

        # images are already normalized to [0,1]
        # print(img_obs_list.shape)
        nimage_obses_1 = img_obs_1_list.transpose((0, 3, 1, 2))  # Change the channel dimension to be the first one (2,img_height,img_width,3)->(2,3,img_height,img_width)
        nimage_obses_2 = img_obs_2_list.transpose((0, 3, 1, 2)) 
        
        # device transfer
        nimage_obses_1 = torch.from_numpy(nimage_obses_1).to(device, dtype=torch.float32)
        nimage_obses_2 = torch.from_numpy(nimage_obses_2).to(device, dtype=torch.float32)
        # (2,3,img_height,img_width)

        # infer action
        with torch.no_grad():
            # get image features
            image_features_1 = ema_nets['vision_encoder_1'](nimage_obses_1.flatten(end_dim=0))
            image_features_2 = ema_nets['vision_encoder_2'](nimage_obses_2.flatten(end_dim=0))
            # (2,512)

            # concat with low-dim observations
            obs_features = torch.cat([image_features_1, image_features_2], dim=-1)
            # print(obs_features)
            # print("两次视觉观测编码之差: ", torch.norm(obs_features[0] - obs_features[1], p=float('inf')))
            # reshape observation to (B,obs_horizon*obs_dim)
            obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

            # initialize action from Guassian noise
            noisy_action = torch.randn(
                (B, pred_horizon, action_dim), device=device)
            naction = noisy_action

            # init scheduler
            noise_scheduler.set_timesteps(num_diffusion_iters)

            for k in noise_scheduler.timesteps:
                # predict noise
                noise_pred = ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k.to(device),
                    global_cond=obs_cond
                )

                # inverse diffusion step (remove noise)
                naction = noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

        # unnormalize action
        naction = naction.detach().to('cpu').numpy()
        # (B, pred_horizon, action_dim)
        naction = naction[0]
        if is_load_dataset:
            action_pred = unnormalize_data(naction, stats=stats['action'])
        else:
            action_pred = unnormalize_data_byValue(naction, max=action_max, min=action_min)

        # only take action_horizon number of actions
        start = obs_horizon - 1
        end = start + action_horizon
        action = action_pred[start:end,:]
        clear_output(wait=True) # 清空当前输出
        print("DiffusionPolicy预测的动作为:")
        utils.print_array_with_precision(action, 3)
        # (action_horizon, action_dim)
        """--------------------------------------------------"""
        
        # print("按 x 进行紧急停止运动")
        for i in range(len(action)):
            # if keyboard.is_pressed("x"): break
            # "锁死转动"
            # action[i][3:6] = np.zeros(3)
            env.step(action=action[i], 
                    isPrintInfo=isPrintStepInfo)
            "记录标记!"
            info = [step, env.assembleDepth, env.agent_obs[6], env.agent_obs[7], env.agent_obs[8], 
                    env.agent_obs[9], env.agent_obs[10], env.agent_obs[11]]
            try_to_csv(info_path, pd.DataFrame([info]), info="info_data", mode='a')
            step += 1
            "记录标记!"
            if step >= 50 or env.done or env.assembleDepth>0.012:
                print("按x重启")
                while True:
                    if keyboard.is_pressed("x"): break
                break
        
        "记录标记!"
        if step >= 50 or env.done or env.assembleDepth>0.012:
            print("按x重启")
            while True:
                if keyboard.is_pressed("x"): break
            break
    
    
    # if env.success_flag is True:
    #     print("本回合成功")
    # else:
    #     print("本回合失败")
        
    "记录标记!"
    # print("按 e 结束本回合")
    # while True:
    #     env._get_obs(isPrintInfo=False)
    #     if keyboard.is_pressed("e"): break
            
            
    print("---------------------------------------------------------------------")
    print("---------------------------------------------------------------------")
    
    
    
    
    # env.rob.close()
    # break

def restart_kernel():
    display(Javascript('IPython.notebook.kernel.restart()'))
    os._exit(0)

restart_kernel()


