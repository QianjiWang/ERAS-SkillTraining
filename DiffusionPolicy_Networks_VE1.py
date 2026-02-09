from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
# import collections
# import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
# from tqdm.auto import tqdm
# import cv2
import timm
from transformers import AutoImageProcessor, AutoModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# from matplotlib import pyplot as plt
# import cv2





#@markdown ### **Vision Encoder**
#@markdown
#@markdown Defines helper functions:
#@markdown - `get_resnet` to initialize standard ResNet vision encoder
#@markdown - `replace_bn_with_gn` to replace all BatchNorm layers with GroupNorm

def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module







"""个人add模块"""

"1_ViT"
# def get_vit(weights=None, **kwargs) -> nn.Module:
#     """
#     Initialize a Vision Transformer model.
#     """
#     vit = torchvision.models.vit_b_16(weights=weights, **kwargs)
#     # Remove the final classification layer if necessary
#     vit.heads = torch.nn.Identity()
#     return vit
def get_vit(weights=None, **kwargs) -> nn.Module:
    """
    Initialize a Vision Transformer model with adaptive pooling.
    """
    # Initialize the Vision Transformer model
    vit = torchvision.models.vit_b_16(weights=weights, **kwargs)
    # Remove the final classification layer if necessary
    vit.heads = nn.Identity()

    # Create a new model that includes adaptive pooling
    class ViTWithAdaptivePooling(nn.Module):
        def __init__(self, vit_model):
            super(ViTWithAdaptivePooling, self).__init__()
            self.vit = vit_model
            self.pool = nn.AdaptiveAvgPool2d((224, 224))  # Pooling to 224x224

        def forward(self, x):
            # Pool the input to (b, c, 224, 224)
            x = self.pool(x)
            # Forward through the ViT model
            return self.vit(x)

    # Return the ViT model wrapped with adaptive pooling
    return ViTWithAdaptivePooling(vit)



"2_Attention"
class MultiHeadAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_linear = nn.Linear(in_channels, in_channels)
        self.k_linear = nn.Linear(in_channels, in_channels)
        self.v_linear = nn.Linear(in_channels, in_channels)
        self.out_linear = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        
        # Linear projections
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).contiguous().view(batch_size, -1, num_channels)
        return self.out_linear(x)

class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.multihead_attention = MultiHeadAttention(in_channels)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        attn_output = self.multihead_attention(x)
        return x + self.conv(attn_output.view(x.size()))  # Skip connection

def get_resnet_with_attention(name: str, weights=None, **kwargs) -> nn.Module:
    resnet = get_resnet(name, weights, **kwargs)

    for name, module in resnet.named_children():
        if isinstance(module, nn.Sequential):
            for block in module:
                in_channels = block.conv2.out_channels  # 获取输出通道数
                block.add_module('attention', AttentionModule(in_channels))
    return resnet



"3_lightweight_vit"
def get_lightweight_vit(backbone_type='mobilevit', pretrained=True):
    """
    轻量级ViT实现方案（需安装timm库：pip install timm）
    """
    model_config = {
        'mobilevit': 'mobilevit_s',        # 参数量5M，ImageNet Top1 78%
        'tinyvit': 'tiny_vit_5m_224',      # 参数量5M，ImageNet Top1 79%
        'efficientformer': 'efficientformer_l1', # 参数量12M，ImageNet Top1 80%
        'poolformer': 'poolformer_s12'     # 参数量12M，纯CNN架构
    }
    model = timm.create_model(
        model_config[backbone_type], 
        pretrained=pretrained,
        num_classes=0  # 移除分类头
    )
    # 添加自适应层保证输出维度为512
    return nn.Sequential(
        model,
        nn.Linear(model.num_features, 512) # 动态适配输出维度
    )
    

class FusedVisualEncoder(nn.Module):
    def __init__(self, model1='mobilevit', model2='poolformer'):
        super().__init__()
        self.encoder1 = get_lightweight_vit(model1)
        self.encoder2 = get_lightweight_vit(model2)
        self.fusion_proj = nn.Sequential(
            nn.Linear(1024, 512),  # 假设两个模型各输出512
            nn.GELU(),
            nn.LayerNorm(512)
        )
    
    def forward(self, x):
        feat1 = self.encoder1(x)
        feat2 = self.encoder2(x)
        fused = torch.cat([feat1, feat2], dim=-1)
        return self.fusion_proj(fused)
    
    


""
# Helper function to replace BatchNorm with GroupNorm
def replace_bn_with_gn(root_module: nn.Module, features_per_group: int=16) -> nn.Module:
    """
    Replace all BatchNorm layers with GroupNorm.
    """
    def replace_fn(module):
        if isinstance(module, nn.BatchNorm2d):
            return nn.GroupNorm(
                num_groups=module.num_features // features_per_group,
                num_channels=module.num_features
            )
        return module

    return replace_submodules(root_module, lambda m: isinstance(m, nn.BatchNorm2d), replace_fn)

# Define ResNet50-based encoder
class ResNet50VisionEncoder(nn.Module):
    def __init__(self, base_model='resnet50', features_per_group=16, output_dim=512):
        super(ResNet50VisionEncoder, self).__init__()
        # Initialize a standard ResNet (could be resnet50 or others)
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer
        self.resnet = replace_bn_with_gn(self.resnet, features_per_group)
        # Add a linear layer to reduce output to 512
        self.fc = nn.Linear(2048, output_dim)
    
    def forward(self, x):
        features = self.resnet(x)
        return self.fc(features)  # Reduce the output to 512 dimensions

# Define DINOv2-based encoder (using a smaller ViT model for better memory usage)
class DINOv2VisionEncoder(nn.Module):
    def __init__(self, features_per_group=16, output_dim=512, is_224=False):
        super(DINOv2VisionEncoder, self).__init__()
        # Initialize DINOv2 Backbone (replace with the correct pre-trained model if needed)
        # processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        # self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.dino = AutoModel.from_pretrained('facebook/dinov2-small')
        # self.dino = replace_bn_with_gn(self.dino, features_per_group)
        # Add a linear layer to reduce output to 512
        self.fc = nn.Linear(384, output_dim)  # DINOv2's ViT typically outputs a 768-dimensional vector
        self.is_224 = is_224
    
    def crop_center(self, x, target_size=224, visualize=False):
        """
        Crop the image centered to the target_size.
        Args:
            x (Tensor): Input image tensor of shape [batch_size, channels, height, width]
            target_size (int): The size to crop to (e.g., 224)
        Returns:
            Tensor: Cropped image tensor of shape [batch_size, channels, target_size, target_size]
        """
        # if visualize:
        #     img_np = x[0].permute(1, 2, 0).cpu().numpy()
        #     img_np = (img_np * 255).astype(np.uint8)
        #     # cv2.imwrite('original.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
        
        _, _, h, w = x.shape
        # Compute the center of the image
        top = (h - target_size) // 2
        left = (w - target_size) // 2
        # Crop the image
        cropped = x[:, :, top:top+target_size, left:left+target_size]
        
        # # Visualization
        # if visualize:
        #     cropped_np = cropped[0].permute(1, 2, 0).cpu().numpy()
        #     cropped_np = (cropped_np * 255).astype(np.uint8)
        #     cv2.imwrite('cropped.jpg', cv2.cvtColor(cropped_np, cv2.COLOR_RGB2BGR))
            
        #     # Show comparison
        #     plt.figure(figsize=(10, 5))
        #     plt.subplot(1, 2, 1)
        #     plt.imshow(img_np)
        #     plt.title(f'Original ({w}x{h})')
        #     plt.subplot(1, 2, 2)
        #     plt.imshow(cropped_np)
        #     plt.title(f'Cropped ({target_size}x{target_size})')
        #     plt.show()
        
        return cropped
    
    def forward(self, x):
        if self.is_224 and (x.shape[2] > 224 and x.shape[3] > 224):
            x = self.crop_center(x, target_size=224)
        # print(x.shape)
        # x = self.processor(images=x, return_tensors="pt").to(device)
        # print(**x.shape)
        # features = self.dino(**x).last_hidden_state  # Get the last hidden state
        features = self.dino(x).last_hidden_state  # Get the last hidden state
        # print(features)
        # print(features.shape)
        return self.fc(features[:, 0, :])  # Use the [CLS] token's features and reduce to 512


# Example of how to use the encoders
def test_encoder():
    # Example input: batch_size=1, obs_horizon=10, channels=3, height=256, width=256
    image_input = torch.zeros((1, 10, 3, 256, 256))  # Simulating an image batch

    # Choose encoder type ('ResNet50' or 'DINOv2')
    vision_encoder = ResNet50VisionEncoder()
    image_features = vision_encoder(image_input.flatten(end_dim=1))  # Flatten input as required

    print(f"Output shape of visual features: {image_features.shape}")

if __name__ == "__main__":
    test_encoder()
