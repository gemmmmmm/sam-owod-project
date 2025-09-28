# ------------------------------------------------------------------------
# SAM Integration Module for OW-DETR
# 封装Semantic-SAM的核心功能，提供可控过分割和特征提取接口
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any

# 导入Semantic-SAM的核心组件
import sys
sys.path.append('Semantic-SAM-main')

from semantic_sam.BaseModel import BaseModel
from semantic_sam import build_model
from tasks.automatic_mask_generator import SemanticSamAutomaticMaskGenerator
from tasks.interactive_predictor import SemanticSAMPredictor
from utils.arguments import load_opt_from_config_file


class SAMIntegration(nn.Module):
    """
    SAM集成模块，封装Semantic-SAM的核心功能
    提供可控过分割和特征提取接口，供OW-DETR使用
    """
    
    def __init__(self, 
                 model_type: str = 'L',  # 'T' or 'L'
                 checkpoint_path: str = '',
                 device: str = 'cuda',
                 config_path: str = 'Semantic-SAM-main/configs/semantic_sam_only_sa-1b_swinL.yaml'):
        """
        初始化SAM集成模块
        
        Args:
            model_type: 模型类型，'T'为SwinT，'L'为SwinL
            checkpoint_path: 预训练模型路径
            device: 设备类型
            config_path: 配置文件路径
        """
        super().__init__()
        
        self.model_type = model_type
        self.device = device
        self.config_path = config_path

        # DETR/OW-DETR使用的图像归一化参数，用于反归一化
        self.register_buffer("detr_pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("detr_pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        
        # 模型配置
        self.cfgs = {
            'T': "Semantic-SAM-main/configs/semantic_sam_only_sa-1b_swinT.yaml",
            'L': "Semantic-SAM-main/configs/semantic_sam_only_sa-1b_swinL.yaml"
        }
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化模型
        self.sam_model = None
        self.mask_generator = None
        self.predictor = None
        
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        config_file = self.cfgs[self.model_type]
        return load_opt_from_config_file(config_file)
    
    def load_model(self, checkpoint_path: str):
        """加载预训练模型"""
        try:
            # 构建模型
            self.sam_model = BaseModel(
                self.config, 
                build_model(self.config)
            ).from_pretrained(checkpoint_path).eval().to(self.device)
            
            # 初始化mask生成器
            self.mask_generator = SemanticSamAutomaticMaskGenerator(
                model=self.sam_model,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.92,
                box_nms_thresh=0.7,
                level=[1, 2, 3, 4, 5, 6]  # 支持多粒度分割
            )
            
            # 初始化交互式预测器
            self.predictor = SemanticSAMPredictor(self.sam_model)
            
            print(f"成功加载SAM模型: {checkpoint_path}")
            
        except Exception as e:
            print(f"加载SAM模型失败: {e}")
            self.sam_model = None
            self.mask_generator = None
            self.predictor = None
    
    def _denormalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        对来自DETR的归一化图像进行反归一化，以适配SAM的输入要求
        
        Args:
            images: 归一化的图像张量 (B, C, H, W)
            
        Returns:
            unnormalized_images: 像素值在 [0, 255] 范围内的图像张量
        """
        # 反归一化: (tensor * std) + mean
        images = images * self.detr_pixel_std + self.detr_pixel_mean
        # 将像素值范围从 [0, 1] 转换为 [0, 255]
        unnormalized_images = images * 255.0
        return unnormalized_images.clamp(0, 255)

    def prepare_image(self, image_path: str) -> Tuple[np.ndarray, torch.Tensor]:
        """
        预处理图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            image_ori: 原始图像数组
            image_tensor: 预处理后的图像张量
        """
        image = Image.open(image_path).convert('RGB')
        
        # 调整图像大小
        t = transforms.Compose([
            transforms.Resize(640, interpolation=Image.BICUBIC)
        ])
        image_ori = t(image)
        image_ori = np.asarray(image_ori)
        
        # 转换为张量
        image_tensor = torch.from_numpy(image_ori.copy()).permute(2, 0, 1).to(self.device)
        
        return image_ori, image_tensor
    
    def generate_masks(self, 
                      image_path: str, 
                      granularity_levels: List[int] = [1, 2, 3, 4, 5, 6]) -> Dict[str, Any]:
        """
        生成多粒度分割掩码
        
        Args:
            image_path: 图像路径
            granularity_levels: 分割粒度级别 [1-6]
            
        Returns:
            masks: 包含不同粒度掩码的字典
        """
        if self.mask_generator is None:
            raise ValueError("SAM模型未加载，请先调用load_model()")
        
        # 预处理图像
        image_ori, image_tensor = self.prepare_image(image_path)
        
        # 设置分割粒度
        self.mask_generator.level = granularity_levels
        
        # 生成掩码
        masks = self.mask_generator.generate(image_tensor)
        
        return {
            'masks': masks,
            'image_ori': image_ori,
            'image_tensor': image_tensor,
            'granularity_levels': granularity_levels
        }
    
    def extract_features(self, 
                        image_tensor: torch.Tensor, 
                        masks: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        从分割掩码中提取特征
        
        Args:
            image_tensor: 图像张量
            masks: 分割掩码列表
            
        Returns:
            features: 包含不同粒度特征的字典
        """
        if self.sam_model is None:
            raise ValueError("SAM模型未加载，请先调用load_model()")
        
        features = {}
        
        # 提取整体特征
        with torch.no_grad():
            # 获取图像特征
            image_features = self.sam_model.model.backbone(image_tensor.unsqueeze(0))
            
            # 为每个掩码提取特征
            for i, mask_info in enumerate(masks):
                mask = mask_info['segmentation']
                bbox = mask_info['bbox']
                
                # 提取掩码区域的特征
                mask_features = self._extract_mask_features(image_features, mask, bbox)
                
                # 按粒度级别组织特征
                level = mask_info.get('level', 1)
                if f'level_{level}' not in features:
                    features[f'level_{level}'] = []
                features[f'level_{level}'].append(mask_features)
        
        return features
    
    def _extract_mask_features(self, 
                              image_features: Dict[str, torch.Tensor], 
                              mask: np.ndarray, 
                              bbox: List[int]) -> torch.Tensor:
        """
        从掩码中提取特征
        
        Args:
            image_features: 图像特征字典
            mask: 分割掩码
            bbox: 边界框
            
        Returns:
            mask_features: 掩码特征
        """
        # 将掩码转换为张量
        mask_tensor = torch.from_numpy(mask).float().to(self.device)
        
        # 从不同层级的特征中提取
        mask_features = []
        for level, feat in image_features.items():
            # 调整掩码大小以匹配特征图
            feat_h, feat_w = feat.shape[-2:]
            mask_resized = F.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0),
                size=(feat_h, feat_w),
                mode='nearest'
            ).squeeze()
            
            # 应用掩码并提取特征
            masked_feat = feat * mask_resized.unsqueeze(0)
            mask_features.append(masked_feat)
        
        # 合并不同层级的特征
        combined_features = torch.cat(mask_features, dim=1)
        
        return combined_features
    
    def get_controllable_segmentation(self, 
                                    image_path: str, 
                                    control_level: int = 3) -> Dict[str, Any]:
        """
        获取可控分割结果
        
        Args:
            image_path: 图像路径
            control_level: 控制级别 (1-6)
            
        Returns:
            segmentation_result: 分割结果
        """
        # 生成指定粒度的掩码
        masks_result = self.generate_masks(image_path, [control_level])
        
        # 提取特征
        features = self.extract_features(
            masks_result['image_tensor'], 
            masks_result['masks']
        )
        
        return {
            'masks': masks_result['masks'],
            'features': features,
            'control_level': control_level,
            'image_ori': masks_result['image_ori']
        }
    
    def get_multi_granularity_features(self, 
                                     image_path: str) -> Dict[str, Any]:
        """
        获取多粒度特征，用于OW-DETR的未知类检测
        
        Args:
            image_path: 图像路径
            
        Returns:
            multi_granularity_result: 多粒度分割和特征结果
        """
        # 生成所有粒度的掩码
        masks_result = self.generate_masks(image_path, [1, 2, 3, 4, 5, 6])
        
        # 提取特征
        features = self.extract_features(
            masks_result['image_tensor'], 
            masks_result['masks']
        )
        
        # 组织结果
        result = {
            'masks': masks_result['masks'],
            'features': features,
            'image_ori': masks_result['image_ori'],
            'image_tensor': masks_result['image_tensor']
        }
        
        return result
    
    def forward(self, batched_images: torch.Tensor, granularity_levels: List[int] = [1, 2, 3, 4, 5, 6]) -> List[Dict[str, Any]]:
        """
        前向传播，处理一批已经预处理好的图像张量
        
        Args:
            batched_images: 来自DETR数据加载器的图像张量 (B, C, H, W)，已归一化
            granularity_levels: 分割粒度级别 [1-6]
            
        Returns:
            all_results: 一个列表，每个元素是对应图像的分割和特征结果字典
        """
        if self.mask_generator is None:
            raise ValueError("SAM模型未加载，请先调用load_model()")
        
        # 1. 对图像进行反归一化，以适配SAM的输入要求
        images_unnormalized = self._denormalize_images(batched_images)
        
        # 2. 准备SAM模型需要的输入格式
        batched_inputs = []
        for img_tensor in images_unnormalized:
            # SAM需要 H, W, C 格式的输入
            # img_tensor.permute(1, 2, 0).cpu().numpy()
            batched_inputs.append({
                'image': img_tensor,
                'height': img_tensor.shape[1],
                'width': img_tensor.shape[2]
            })
            
        # 3. 设置分割粒度并生成掩码
        self.mask_generator.level = granularity_levels
        # 注意: SemanticSamAutomaticMaskGenerator 内部会遍历批次
        # 但它的generate方法似乎只接受单张图片张量。我们需要确认这一点。
        # 经过检查，其generate方法的确只处理单图，所以我们需要自己遍历
        all_results = []
        for i in range(batched_images.shape[0]):
            image_tensor = batched_images[i] # 单张归一化的图
            image_unnormalized_tensor = images_unnormalized[i] # 单张反归一化的图

            # 3.1 生成掩码
            # generator需要的是 C,H,W 的张量
            masks = self.mask_generator.generate(image_unnormalized_tensor)
            
            # 3.2 提取特征
            # 使用归一化的张量送入backbone进行特征提取
            image_features = self.sam_model.model.backbone(image_tensor.unsqueeze(0))
            
            part_features = []
            for mask_info in masks:
                mask = mask_info['segmentation']
                # 提取部件特征
                mask_features = self._extract_mask_features(image_features, mask)
                part_features.append(mask_features)
            
            # 将该图片的所有部件特征堆叠起来
            if part_features:
                part_features_tensor = torch.stack(part_features, dim=0)
            else:
                part_features_tensor = torch.empty(0, device=self.device)

            # 提取整体特征 (使用整个特征图的平均池化)
            # image_features['res5'] 的形状是 [1, C, H, W]
            whole_feature = F.adaptive_avg_pool2d(image_features['res5'], (1, 1)).squeeze()

            all_results.append({
                'masks': masks,
                'part_features': part_features_tensor, # [num_masks, feature_dim]
                'whole_feature': whole_feature,      # [feature_dim]
                'image_tensor': image_tensor
            })
            
        return all_results

    def _extract_mask_features(self, 
                              image_features: Dict[str, torch.Tensor], 
                              mask: np.ndarray) -> torch.Tensor:
        """
        从掩码中提取特征 (简化版本)
        
        Args:
            image_features: backbone输出的图像特征字典
            mask: 单个分割掩码 (H, W)
            
        Returns:
            mask_feature: 掩码区域对应的特征向量
        """
        mask_tensor = torch.from_numpy(mask).float().to(self.device)
        
        # 我们主要使用最高层级的特征 'res5'
        main_features = image_features['res5']
        
        # 调整掩码大小以匹配特征图
        feat_h, feat_w = main_features.shape[-2:]
        mask_resized = F.interpolate(
            mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(feat_h, feat_w),
            mode='bilinear',
            align_corners=False
        )
        
        # 应用掩码并进行全局平均池化来获得一个固定维度的特征向量
        # main_features shape: [1, C, H, W]
        # mask_resized shape: [1, 1, H, W]
        masked_feat = main_features * mask_resized
        
        # 计算掩码区域的平均特征
        # 添加一个小的epsilon防止除以零
        mask_sum = mask_resized.sum() + 1e-6
        mask_feature = masked_feat.sum(dim=[-1, -2]) / mask_sum
        
        return mask_feature.squeeze()


    def _forward_from_path(self, image_path: str, mode: str = 'multi_granularity') -> Dict[str, Any]:
        """
        【兼容旧版】前向传播，用于从文件路径加载图像，方便独立测试
        
        Args:
            image_path: 图像路径
            mode: 模式 ('multi_granularity', 'controllable')
            
        Returns:
            result: 分割和特征结果
        """
        if mode == 'multi_granularity':
            return self.get_multi_granularity_features(image_path)
        elif mode == 'controllable':
            return self.get_controllable_segmentation(image_path)
        else:
            raise ValueError(f"不支持的模式: {mode}")


# 便捷函数
def build_sam_integration(model_type: str = 'L', 
                         checkpoint_path: str = '',
                         device: str = 'cuda') -> SAMIntegration:
    """
    构建SAM集成模块的便捷函数
    
    Args:
        model_type: 模型类型
        checkpoint_path: 检查点路径
        device: 设备
        
    Returns:
        SAMIntegration: SAM集成模块实例
    """
    return SAMIntegration(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        device=device
    )


# 使用示例
if __name__ == "__main__":
    # 创建SAM集成模块
    sam_integration = build_sam_integration(
        model_type='L',
        checkpoint_path='path/to/checkpoint.pth',
        device='cuda'
    )
    
    # 使用示例 (旧版，通过路径)
    # image_path = "path/to/image.jpg"
    # result = sam_integration._forward_from_path(image_path, mode='multi_granularity')
    
    # 新版使用示例 (通过张量)
    # 假设我们有一个来自数据加载器的批次
    dummy_batch = torch.randn(2, 3, 640, 640).cuda() # B, C, H, W
    
    # 模拟DETR的归一化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).cuda()
    normalized_batch = (dummy_batch - mean) / std

    # 前向传播
    if sam_integration.sam_model:
        results_list = sam_integration(normalized_batch)
        
        # 打印第一张图的结果
        if results_list:
            first_result = results_list[0]
            print(f"第一张图生成了 {len(first_result['masks'])} 个掩码")
            print(f"整体特征维度: {first_result['whole_feature'].shape}")
            print(f"部件特征张量维度: {first_result['part_features'].shape}")
    else:
        print("SAM模型未加载，跳过示例运行。") 
