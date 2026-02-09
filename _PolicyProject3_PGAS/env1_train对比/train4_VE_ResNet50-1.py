
#@markdown ### **Imports**
# file import
import DiffusionPolicy_Networks
# module import
from random import uniform,choice
import numpy as np
import time
import os
import subprocess
import sys
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
"记录标记!"
dataset_path =           (".\\_PolicyProject3_PGAS\\_trained_models_env1\\_teach1_dataset_amplify1")
MinMax_dir_path =        (".\\_PolicyProject3_PGAS\\_trained_models_env1\\MinMax")

ckpt_path =              (".\\_PolicyProject3_PGAS\\env1_train对比\\4_VE_ResNet50-1\\model.ckpt")
output_dir =             (".\\_PolicyProject3_PGAS\\env1_train对比\\4_VE_ResNet50-1")
training_loss_csv_path = (".\\_PolicyProject3_PGAS\\env1_train对比\\4_VE_ResNet50-1\\loss.csv")
max_iterations = 50000  # 目标最大迭代次数
"记录标记!"

img_height = [256,256]
img_width = [256,256]
img_channel = 3

agent_obs_dim = 12
action_dim = 7

pred_horizon = 8
obs_horizon = 1
action_horizon = 4

from utils_file import try_read_csv, try_to_csv
                
                
                
# device transfer
device = torch.device('cuda')
print("示教数据文件夹的路径:\t", dataset_path)


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
    batch_size=16,
    # num_workers=4,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    # persistent_workers=True
)

# visualize data in batch
batch = next(iter(dataloader))
print("batch['image_1'].shape:", batch['image_1'].shape)
print("batch['image_2'].shape:", batch['image_2'].shape)
print("batch['action'].shape", batch['action'].shape)


import DiffusionPolicy_Networks as nets
import DiffusionPolicy_Networks_VE1 as nets_VE1


#@markdown ### **Network Demo**

# construct ResNet18 encoder
# if you have multiple camera views, use seperate encoder weights for each view.
# vision_encoder_1 = nets.get_resnet_with_attention('resnet18')
# vision_encoder_2 = nets.get_resnet_with_attention('resnet18')
vision_encoder_1 = nets_VE1.ResNet50VisionEncoder()
vision_encoder_2 = nets_VE1.ResNet50VisionEncoder()
"可选择冻结预训练权重"
for param in vision_encoder_1.parameters():
    param.requires_grad = False
for param in vision_encoder_2.parameters():
    param.requires_grad = False
# IMPORTANT!
# replace all BatchNorm with GroupNorm to work with EMA
# performance will tank if you forget to do this!
vision_encoder_1 = nets.replace_bn_with_gn(vision_encoder_1)
vision_encoder_2 = nets.replace_bn_with_gn(vision_encoder_2)
print("视觉编码器1的形状:\n", vision_encoder_1)
print("视觉编码器2的形状:\n", vision_encoder_2)

# ResNet18 has output dim of 512
vision_feature_dim = 512
obs_dim = vision_feature_dim + 0

# create network object
noise_pred_net = nets.ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*2*obs_horizon
)

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


#@markdown ### **Training**
#@markdown
#@markdown Takes about 2.5 hours. If you don't want to wait, skip to the next cell
#@markdown to load pre-trained weights


# def restart_notebook():
#     os.execv(sys.argv[0], sys.argv)


"记录标记!"
# 在训练前添加读取已有训练迭代次数的逻辑
def get_total_iterations(training_loss_csv_path):
    if not os.path.exists(training_loss_csv_path):
        # 创建一个空的 CSV 文件
        with open(training_loss_csv_path, 'w') as f:
            pass  # 创建空文件
        print(f"未找到训练损失 CSV 文件，已创建空文件: {training_loss_csv_path}")
    # 读取训练损失 CSV 文件
    loss_data = try_read_csv(training_loss_csv_path, info="训练损失", header=None, isPrintInfo=False)
    # 获取CSV的行数
    if not loss_data.empty:
        # CSV文件中每一行是一次迭代的记录，我们根据行数来判断迭代次数
        return len(loss_data)
    return 0  # 如果没有找到CSV文件或文件为空，返回0
# 获取当前已训练的迭代次数
current_iterations = get_total_iterations(training_loss_csv_path)
print(f"当前迭代次数: {current_iterations}")
remain_iterations = max_iterations - current_iterations
"记录标记!"

def Train_Model():
    num_epochs = 1000
    "记录标记!"
    global remain_iterations
    "记录标记!"
    
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights

    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=1e-4, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    loss_info = []
    try:
        with tqdm(range(num_epochs), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for batch_idx, nbatch in enumerate(tepoch):
                        try:
                            # data normalized in dataset
                            # device transfer
                            nimage_1 = nbatch['image_1'][:,:obs_horizon].to(device)
                            nimage_2 = nbatch['image_2'][:,:obs_horizon].to(device)
                            # # 拼接 nimage_1 和 nimage_2，在 obs_horizon 维度上进行拼接
                            # nimage_combined = torch.cat((nimage_1, nimage_2), dim=1)
                            naction = nbatch['action'].to(device)

                            # encoder vision features
                            image_features_1 = nets['vision_encoder_1'](
                                nimage_1.flatten(end_dim=1))
                            image_features_1 = image_features_1.reshape(
                                *nimage_1.shape[:2],-1)
                            B = image_features_1.shape[0]
                            # (B,obs_horizon,D)
                            image_features_2 = nets['vision_encoder_2'](
                                nimage_2.flatten(end_dim=1))
                            image_features_2 = image_features_2.reshape(
                                *nimage_2.shape[:2],-1)

                            # concatenate vision feature and low-dim obs
                            obs_features = torch.cat([image_features_1, image_features_2], dim=-1)
                            obs_cond = obs_features.flatten(start_dim=1)
                            # (B, obs_horizon * obs_dim)

                            # sample noise to add to actions
                            noise = torch.randn(naction.shape, device=device)

                            # sample a diffusion iteration for each data point
                            timesteps = torch.randint(
                                0, noise_scheduler.config.num_train_timesteps,
                                (B,), device=device
                            ).long()

                            # add noise to the clean images according to the noise magnitude at each diffusion iteration
                            # (this is the forward diffusion process)
                            noisy_actions = noise_scheduler.add_noise(
                                naction, noise, timesteps)

                            # predict the noise residual
                            noise_pred = noise_pred_net(
                                noisy_actions, timesteps, global_cond=obs_cond)

                            # L2 loss
                            loss = nn.functional.mse_loss(noise_pred, noise)

                            # optimize
                            loss.backward()
                            optimizer.step()
                            optimizer.zero_grad()
                            # step lr scheduler every batch
                            # this is different from standard pytorch behavior
                            lr_scheduler.step()

                            # update Exponential Moving Average of the model weights
                            """ 给的代码和环境中diffusers(0.11.1)的方法用法不一致，需要输入整个模型而非仅参数
                            ema.step(noise_pred_net.parameters())
                            """
                            ema.step(noise_pred_net)

                            # logging
                            loss_cpu = loss.item()
                            epoch_loss.append(loss_cpu)
                            tepoch.set_postfix(loss=loss_cpu)
                            
                            loss_info.append([epoch_idx + 1, batch_idx + 1, loss_cpu])
                            
                            "记录标记!"
                            remain_iterations = remain_iterations - 1
                            if remain_iterations <= 0:
                                print("迭代次数已达上限")
                                return
                            "记录标记!"
                            
                            # 每batch迭代保存模型
                            if (batch_idx + 1) % 50 == 0:
                                save_model(num_epochs, optimizer, lr_scheduler, ema)
                                # 将当前损失写入 CSV 文件 
                                try_to_csv(training_loss_csv_path, 
                                        pd.DataFrame(np.array(loss_info).reshape(-1,3)),
                                        info="训练损失", index=False, header=False, mode='a', isPrintInfo=True)
                                loss_info = []
                                "记录标记!"
                                print("剩余迭代次数: ", remain_iterations)
                                "记录标记!"
                                
                        
                        except Exception as e:
                            print(f"Batch {batch_idx} failed with error: {e}")
                            raise  # 重新抛出异常以触发外层的处理
                    

                tglobal.set_postfix(loss=np.mean(epoch_loss))
                
                # 每epoch保存模型
                if (epoch_idx + 1) % 1 == 0:
                    save_model(num_epochs, optimizer, lr_scheduler, ema)
                    # 将当前损失写入 CSV 文件 
                    try_to_csv(training_loss_csv_path, 
                            pd.DataFrame(np.array(loss_info).reshape(-1,3)),
                            info="训练损失", index=False, header=False, mode='a', isPrintInfo=True)
                    loss_info = []
                    "记录标记!"
                    print("剩余迭代次数: ", remain_iterations)
                    "记录标记!"
                    
        # Weights of the EMA model
        # is used for inference
        ema_noise_pred_net = noise_pred_net
        ema.copy_to(ema_noise_pred_net.parameters())
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # 重启脚本
        subprocess.Popen([sys.executable, *sys.argv])
        sys.exit(1)


# 训练
Train_Model()
