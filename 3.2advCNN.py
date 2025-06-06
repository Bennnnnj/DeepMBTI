import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import multiprocessing
import copy
import matplotlib.font_manager as fm
from matplotlib import rcParams
import warnings
from torch.distributions import Categorical

# 设置CUDA错误同步报告，更容易调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# 设置中文字体支持
def setup_chinese_font():
    try:
        if os.path.exists('SimHei.ttf'):
            plt.rcParams['font.sans-serif'] = ['SimHei'] 
        else:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    except Exception as e:
        print(f"设置中文字体出错: {e}")
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']

# 设置随机种子以确保可重复性
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# 更新数据路径
data_root = r"C:\Users\lnasl\Desktop\DeepMBTI\data\emotion"
split_dir = os.path.join(data_root, "split")
train_dir = os.path.join(split_dir, "train")
val_dir = os.path.join(split_dir, "val")
test_dir = os.path.join(split_dir, "test")

# 获取情绪类别
def get_emotion_categories():
    return sorted(os.listdir(train_dir))

# 计算类权重以处理不平衡问题
def calculate_class_weights(data_dir, emotion_categories, accuracy_dict=None, beta=0.9999):
    class_counts = {}
    num_classes = len(emotion_categories)
    
    # 计算每个类的样本数
    for emotion in emotion_categories:
        class_path = os.path.join(data_dir, emotion)
        if os.path.exists(class_path):
            class_counts[emotion] = len([f for f in os.listdir(class_path) 
                                         if f.endswith(('.jpg', '.jpeg', '.png'))])
        else:
            class_counts[emotion] = 0
            print(f"警告: 目录不存在: {class_path}")
    
    total_samples = sum(class_counts.values())
    
    # 如果提供了准确率字典，结合准确率和样本数量进行权重计算
    if accuracy_dict:
        # 使用有效数量采样和准确率反比进行加权
        effective_num = {emotion: 1.0 - beta**count for emotion, count in class_counts.items()}
        weights = {emotion: (1.0 - beta) / (effective_num[emotion] + 1e-10) for emotion in class_counts}
        
        # 根据准确率进一步调整权重（准确率越低，权重越高）
        weights = {emotion: weight * (1.0/(accuracy_dict.get(emotion, 0.5) + 0.05)**2) 
                  for emotion, weight in weights.items()}
    else:
        # 仅基于样本数量
        weights = {emotion: total_samples / (num_classes * count) if count > 0 else 1.0
                  for emotion, count in class_counts.items()}
    
    # 归一化权重
    weight_sum = sum(weights.values())
    norm_class_weights = {k: v / weight_sum * num_classes for k, v in weights.items()}
    
    # 转换为张量格式
    weights_tensor = torch.FloatTensor([norm_class_weights[emotion] for emotion in emotion_categories])
    
    return norm_class_weights, weights_tensor

# 获取基础和增强的数据转换
def get_transforms(mode='train', target_classes=None):
    if mode == 'train':
        # 基础变换
        base_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),
        ])
        
        # 增强变换 - 针对困难类别
        enhanced_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
        
        return base_transforms, enhanced_transforms
    else:
        # 用于验证和测试的转换
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 多处理兼容的情绪数据集（含自动删除错误图片功能）
class ReinforcementEmotionDataset(Dataset):
    def __init__(self, data_dir, emotion_categories, transform=None, enhanced_transform=None, 
                 target_classes=None, adaptive_augment=True, mixer_prob=0.3):
        self.data_dir = data_dir
        self.emotion_categories = emotion_categories
        self.transform = transform
        self.enhanced_transform = enhanced_transform
        self.target_classes = target_classes or []
        self.adaptive_augment = adaptive_augment
        self.mixer_prob = mixer_prob
        
        # 读取所有图像文件路径和相应标签
        self.samples = []
        self.targets = []
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(emotion_categories)}
        self.idx_to_emotion = {i: emotion for i, emotion in enumerate(emotion_categories)}
        
        # 按类组织样本
        self.samples_by_class = {emotion: [] for emotion in emotion_categories}
        
        # 加载并过滤数据集
        self._load_dataset()
        
        # 初始化样本重要性权重
        self.sample_weights = np.ones(len(self.samples), dtype=np.float32)
    
    def _load_dataset(self):
        """加载并验证数据集中的所有图像"""
        print("加载并验证数据集...")
        
        for emotion in self.emotion_categories:
            emotion_dir = os.path.join(self.data_dir, emotion)
            emotion_idx = self.emotion_to_idx[emotion]
            
            if not os.path.exists(emotion_dir):
                print(f"警告: 目录不存在: {emotion_dir}")
                continue
                
            image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # 验证图像完整性
            valid_files = []
            for img_path in image_files:
                try:
                    # 尝试打开图像以验证
                    with Image.open(img_path) as img:
                        img.verify()  # 验证图像文件
                    valid_files.append(img_path)
                except Exception as e:
                    print(f"验证时遇到损坏的图像，删除: {img_path}")
                    try:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                    except Exception as del_err:
                        print(f"删除失败: {del_err}")
            
            self.samples_by_class[emotion] = valid_files
            self.samples.extend(valid_files)
            self.targets.extend([emotion_idx] * len(valid_files))
            
        print(f"数据集加载完成，有效样本总数: {len(self.samples)}")
    
    def update_sample_weights(self, indices, rewards):
        """更新样本权重基于强化学习的反馈"""
        # 增加获得高奖励（难样本）的权重，减少低奖励（容易样本）的权重
        for idx, reward in zip(indices, rewards):
            if idx < len(self.sample_weights):
                # 根据误差调整权重：奖励高（误差大）则增加权重
                self.sample_weights[idx] = min(2.0, self.sample_weights[idx] * (1.0 + 0.2 * reward))
            
        # 归一化权重
        if len(self.sample_weights) > 0:
            self.sample_weights = self.sample_weights / np.mean(self.sample_weights)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        emotion = self.idx_to_emotion[target]
        weight = self.sample_weights[idx]
        
        # 尝试读取图像，如果失败则删除文件并返回占位图像
        try:
            # 读取图像
            img = Image.open(img_path).convert('RGB')
            
            # 对低准确率类别应用特殊增强
            if self.adaptive_augment and emotion in self.target_classes and self.enhanced_transform is not None:
                # 随机选择使用增强变换
                if random.random() < 0.8:
                    # 应用手动增强
                    if random.random() < 0.6:
                        # 模糊处理模拟不清晰图像
                        radius = random.uniform(0.1, 0.9)
                        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                    
                    # 调整对比度
                    if random.random() < 0.7:
                        enhancer = ImageEnhance.Contrast(img)
                        factor = random.uniform(0.6, 1.4)
                        img = enhancer.enhance(factor)
                    
                    # 调整亮度
                    if random.random() < 0.7:
                        enhancer = ImageEnhance.Brightness(img)
                        factor = random.uniform(0.6, 1.4)
                        img = enhancer.enhance(factor)
                    
                    # 旋转
                    if random.random() < 0.7:
                        angle = random.uniform(-35, 35)
                        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
                    
                    # 应用增强变换
                    img = self.enhanced_transform(img)
                else:
                    # 应用基础变换
                    img = self.transform(img)
                
                # 应用混合增强 (MixUp/CutMix)
                if random.random() < self.mixer_prob:
                    try:
                        # 混合策略：50%同类混合，50%跨类混合
                        if random.random() < 0.5:
                            # 从同一类别随机选择另一张图像
                            same_class_samples = self.samples_by_class[emotion]
                            if len(same_class_samples) > 1:
                                other_img_path = random.choice([s for s in same_class_samples if s != img_path])
                                try:
                                    other_img = Image.open(other_img_path).convert('RGB')
                                    
                                    # 为第二张图像应用相同变换
                                    if random.random() < 0.8:
                                        other_img = self.enhanced_transform(other_img)
                                    else:
                                        other_img = self.transform(other_img)
                                    
                                    # 混合两张图像 (MixUp)
                                    alpha = random.uniform(0.3, 0.7)
                                    img = alpha * img + (1 - alpha) * other_img
                                except Exception as e:
                                    # 如果第二张图像有问题，删除它并继续使用原图
                                    print(f"混合图像出错，删除文件: {other_img_path}")
                                    if os.path.exists(other_img_path):
                                        os.remove(other_img_path)
                                    # 更新数据结构，从samples_by_class和samples中移除
                                    if other_img_path in self.samples_by_class[emotion]:
                                        self.samples_by_class[emotion].remove(other_img_path)
                        else:
                            # 从不同类别随机选择图像
                            other_emotions = [e for e in self.emotion_categories if e != emotion]
                            if other_emotions:
                                other_emotion = random.choice(other_emotions)
                                other_class_samples = self.samples_by_class[other_emotion]
                                if other_class_samples:
                                    other_img_path = random.choice(other_class_samples)
                                    try:
                                        other_img = Image.open(other_img_path).convert('RGB')
                                        
                                        # 为第二张图像应用变换
                                        other_img = self.transform(other_img)
                                        
                                        # 混合策略随机选择：MixUp 或 CutMix
                                        if random.random() < 0.5:  # MixUp
                                            alpha = random.uniform(0.2, 0.4)
                                            img = alpha * img + (1 - alpha) * other_img
                                        else:  # 简化的 CutMix
                                            h, w = img.shape[1], img.shape[2]
                                            r_x, r_y = random.randint(0, w//2), random.randint(0, h//2)
                                            r_w, r_h = random.randint(w//4, w//2), random.randint(h//4, h//2)
                                            img[:, r_y:r_y+r_h, r_x:r_x+r_w] = other_img[:, r_y:r_y+r_h, r_x:r_x+r_w]
                                    except Exception as e:
                                        # 如果第二张图像有问题，删除它并继续使用原图
                                        print(f"混合图像出错，删除文件: {other_img_path}")
                                        if os.path.exists(other_img_path):
                                            os.remove(other_img_path)
                                        # 更新数据结构
                                        if other_img_path in self.samples_by_class[other_emotion]:
                                            self.samples_by_class[other_emotion].remove(other_img_path)
                    except Exception as e:
                        print(f"混合增强出错: {e}")
            else:
                # 为其他类别应用基础变换
                img = self.transform(img)
            
            # 返回图像、目标、样本权重和索引
            return img, target, weight, idx
            
        except Exception as e:
            print(f"读取图像 {img_path} 出错: {e}")
            
            # 删除问题文件
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                    print(f"已删除错误文件: {img_path}")
                    
                    # 从数据结构中移除引用
                    if img_path in self.samples_by_class[emotion]:
                        self.samples_by_class[emotion].remove(img_path)
                    
                    # 注意：我们不从self.samples和self.targets中移除，
                    # 因为这会改变索引映射并可能导致训练中断
                except Exception as delete_error:
                    print(f"删除文件 {img_path} 时出错: {delete_error}")
            
            # 返回占位图像和原始标签
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, target, weight, idx

# 在每个周期后重建数据集以移除无效文件引用
def rebuild_dataset_after_epoch(train_loader, train_dataset):
    """在每个训练周期后重建数据集，移除所有无效文件的引用"""
    print("重建数据集以移除无效文件引用...")
    
    # 获取当前的样本权重
    old_weights = train_dataset.sample_weights
    
    # 创建新的样本和目标列表
    new_samples = []
    new_targets = []
    new_weights = []
    
    # 重建列表，只包含有效文件
    for emotion in train_dataset.emotion_categories:
        emotion_idx = train_dataset.emotion_to_idx[emotion]
        valid_samples = train_dataset.samples_by_class[emotion]
        
        for sample_path in valid_samples:
            if os.path.exists(sample_path):
                new_samples.append(sample_path)
                new_targets.append(emotion_idx)
                
                # 如果是已存在的样本，保留其权重
                if sample_path in train_dataset.samples:
                    old_idx = train_dataset.samples.index(sample_path)
                    if old_idx < len(old_weights):
                        new_weights.append(old_weights[old_idx])
                    else:
                        new_weights.append(1.0)  # 默认权重
                else:
                    new_weights.append(1.0)  # 默认权重
    
    # 更新数据集的样本和目标
    train_dataset.samples = new_samples
    train_dataset.targets = new_targets
    train_dataset.sample_weights = np.array(new_weights, dtype=np.float32)
    
    # 如果使用了采样器，更新它
    if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'update'):
        train_loader.sampler.update()
    
    print(f"数据集重建完成，当前有效样本数: {len(new_samples)}")
    
    return train_loader, train_dataset

# 强化学习策略网络 - 用于决定样本重要性
class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features):
        return self.network(features)

# 高级均衡采样器
class DynamicWeightedSampler(WeightedRandomSampler):
    def __init__(self, dataset, class_accuracies=None, beta=0.9999):
        self.dataset = dataset
        
        # 获取每个类的索引
        indices_per_class = {}
        for idx, target in enumerate(dataset.targets):
            emotion = dataset.emotion_categories[target]
            if emotion not in indices_per_class:
                indices_per_class[emotion] = []
            indices_per_class[emotion].append(idx)

        # 计算类权重
        if class_accuracies is None:
            class_weights = {emotion: 1.0 for emotion in dataset.emotion_categories}
        else:
            # 使用有效数量采样基于样本数量和准确率
            n_samples = len(dataset)
            n_classes = len(dataset.emotion_categories)
            
            # 计算每个类的样本数
            class_counts = {emotion: len(indices) for emotion, indices in indices_per_class.items()}
            
            # 计算目标采样数量
            effective_num = {emotion: 1.0 - beta**count for emotion, count in class_counts.items()}
            weights = {emotion: (1.0 - beta) / (effective_num[emotion] + 1e-10) for emotion in class_counts}
            
            # 根据准确率进一步调整权重
            weights = {emotion: weight * (1.0/(class_accuracies.get(emotion, 0.5) + 0.05)**2) 
                      for emotion, weight in weights.items()}
                
            # 归一化
            total_weight = sum(weights.values())
            class_weights = {emotion: weight/total_weight * n_classes for emotion, weight in weights.items()}
            
            print("采样器类权重:")
            for emotion, weight in class_weights.items():
                print(f"{emotion}: {weight:.4f}")
        
        # 计算每个样本的权重（结合样本和类权重）
        weights = []
        for idx, target in enumerate(dataset.targets):
            emotion = dataset.emotion_categories[target]
            # 将类权重与样本自身权重相乘
            weights.append(class_weights[emotion] * dataset.sample_weights[idx])
            
        # 初始化采样器
        super().__init__(weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)
    
    def update(self):
        """在样本权重更新后更新采样器权重"""
        weights = []
        for idx, target in enumerate(self.dataset.targets):
            emotion = self.dataset.emotion_categories[target]
            weights.append(self.dataset.sample_weights[idx])
        
        self.weights = torch.DoubleTensor(weights)

# 创建数据加载器
def create_dataloaders(batch_size=32, num_workers=4, emotion_categories=None, class_accuracies=None):
    # 根据准确率确定目标类
    if class_accuracies:
        target_classes = [emotion for emotion, acc in class_accuracies.items() if acc < 0.7]
        print(f"目标优化类别: {target_classes}")
    else:
        target_classes = ["Confusion", "Contempt", "Disgust"]  # 默认目标类
    
    # 获取基础和增强变换
    base_transforms, enhanced_transforms = get_transforms(mode='train', target_classes=target_classes)
    val_test_transform = get_transforms(mode='val')
    
    # 创建数据集
    train_dataset = ReinforcementEmotionDataset(
        train_dir, emotion_categories, 
        transform=base_transforms, 
        enhanced_transform=enhanced_transforms,
        target_classes=target_classes, 
        adaptive_augment=True,
        mixer_prob=0.3  # 混合增强概率
    )
    
    val_dataset = ReinforcementEmotionDataset(
        val_dir, emotion_categories, 
        transform=val_test_transform, 
        enhanced_transform=None,
        target_classes=None, 
        adaptive_augment=False
    )
    
    test_dataset = ReinforcementEmotionDataset(
        test_dir, emotion_categories, 
        transform=val_test_transform, 
        enhanced_transform=None,
        target_classes=None, 
        adaptive_augment=False
    )
    
    # 计算训练集中的类分布
    train_class_counts = {}
    for target in train_dataset.targets:
        emotion = emotion_categories[target]
        train_class_counts[emotion] = train_class_counts.get(emotion, 0) + 1
    
    print("训练集类分布:")
    for emotion, count in sorted(train_class_counts.items()):
        print(f"{emotion}: {count} 图像")
    
    # 创建平衡采样器
    if class_accuracies:
        # 使用动态加权采样器
        sampler = DynamicWeightedSampler(train_dataset, class_accuracies)
        shuffle = False  # 使用采样器时不能使用shuffle
    else:
        sampler = None
        shuffle = True
    
    # 创建数据加载器 - 使用多处理
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, sampler, train_dataset

# 注意力模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(y)
        
        return x * sa

# 高级情绪识别模型
class AdvancedEmotionModel(nn.Module):
    def __init__(self, num_classes, dropout_rates=[0.5, 0.4, 0.3], backbone='resnet50'):
        super(AdvancedEmotionModel, self).__init__()
        
        # 选择基础模型 - 使用更可靠的骨干网络
        if backbone == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights='DEFAULT')
            last_channel = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.base_model = models.resnet50(weights='DEFAULT')
            last_channel = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 添加高级注意力模块
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 复杂分类头 - 更深的MLP，更强的正则化
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(512, num_classes)
        )
        
        # 特殊化分类头用于难以分类的类别
        self.specialized_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, num_classes//2)  # 针对一半的类别
        )
        
        # 特征提取器 - 用于强化学习
        self.feature_extractor = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, extract_features=False):
        # 特征提取 - 处理不同类型的骨干网络
        if isinstance(self.base_model, models.efficientnet.EfficientNet):
            features = self.base_model.features(x)
            features = self.cbam(features)
            features = self.avg_pool(features)
            features = torch.flatten(features, 1)
        elif isinstance(self.base_model, models.resnet.ResNet):
            x = self.base_model.conv1(x)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            features = self.base_model.layer4(x)
            
            features = self.cbam(features)
            features = self.avg_pool(features)
            features = torch.flatten(features, 1)
        else:
            try:
                # 尝试通用方法
                if hasattr(self.base_model, 'forward_features'):
                    features = self.base_model.forward_features(x)
                    if features.dim() == 4:
                        features = self.cbam(features)
                        features = self.avg_pool(features)
                        features = torch.flatten(features, 1)
                else:
                    x = self.base_model(x)
                    features = x
            except Exception as e:
                print(f"模型前向传播出错: {e}")
                # 如果所有方法都失败，使用基本卷积层
                x = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)(x)
                x = nn.BatchNorm2d(64)(x)
                x = nn.ReLU()(x)
                x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)
                # 添加一些基本层
                for _ in range(4):
                    x = nn.Conv2d(64, 64, kernel_size=3, padding=1)(x)
                    x = nn.BatchNorm2d(64)(x)
                    x = nn.ReLU()(x)
                features = nn.AdaptiveAvgPool2d(1)(x)
                features = torch.flatten(features, 1)
                print("使用备用特征提取器")
        
        # 提取中间特征用于强化学习
        extracted_features = self.feature_extractor(features)
        
        # 主分类器输出
        main_output = self.classifier(features)
        
        # 特殊分类器
        specialized_output = self.specialized_classifier(features)
        
        # 如果仅提取特征
        if extract_features:
            return main_output, extracted_features
        
        return main_output, specialized_output, extracted_features

# 改进的焦点损失实现 - 用于难分类样本
class AdvancedFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', class_weights=None):
        super(AdvancedFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        self.class_weights = class_weights  # 每个类的权重
        
    def forward(self, inputs, targets, sample_weights=None):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用样本特定权重
        if sample_weights is not None:
            focal_loss = focal_loss * sample_weights
            
        # 应用类特定权重
        if self.class_weights is not None:
            class_weights_tensor = torch.tensor([self.class_weights[t.item()] 
                                               for t in targets], device=inputs.device)
            focal_loss = focal_loss * class_weights_tensor
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 标签平滑交叉熵
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target, sample_weights=None):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
        
        loss = torch.sum(-true_dist * pred, dim=self.dim)
        
        # 应用样本权重
        if sample_weights is not None:
            loss = loss * sample_weights
            
        return loss.mean()

# 复合损失函数 - 结合焦点损失、标签平滑和自适应加权
class CompositeLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1, 
                 focal_weight=0.7, smooth_weight=0.3, rl_weight=0.2):
        super(CompositeLoss, self).__init__()
        self.focal_loss = AdvancedFocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing = LabelSmoothingLoss(classes=num_classes, smoothing=smoothing, weight=alpha)
        
        # 为特殊化分类器创建单独的损失函数（针对一半类别）
        if alpha is not None and len(alpha) == num_classes:
            # 创建适合一半类别数的权重
            specialized_alpha = alpha[:num_classes//2]  # 仅使用前一半类别的权重
        else:
            specialized_alpha = None
            
        self.specialized_focal_loss = AdvancedFocalLoss(alpha=specialized_alpha, gamma=gamma)
        self.specialized_label_smoothing = LabelSmoothingLoss(classes=num_classes//2, smoothing=smoothing, weight=specialized_alpha)
        
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight
        self.rl_weight = rl_weight  # 强化学习奖励权重
        
    def forward(self, main_outputs, specialized_outputs, targets, policy_outputs=None, sample_weights=None):
    # 主分类器损失
        focal = self.focal_loss(main_outputs, targets, sample_weights)
        smooth = self.label_smoothing(main_outputs, targets, sample_weights)
        main_loss = self.focal_weight * focal + self.smooth_weight * smooth
        
        # 特殊分类器损失（用于困难类别）
        spec_loss = 0
        if specialized_outputs is not None:
            # 将目标重新映射到特殊化分类器的范围
            specialized_targets = targets % (specialized_outputs.size(1))
            
            # 使用正确类别数的损失函数
            spec_focal = self.specialized_focal_loss(specialized_outputs, specialized_targets, sample_weights)
            spec_smooth = self.specialized_label_smoothing(specialized_outputs, specialized_targets, sample_weights)
            spec_loss = self.focal_weight * spec_focal + self.smooth_weight * spec_smooth
        
        # 强化学习损失
        rl_loss = 0
        if policy_outputs is not None:
            # 修复：正确处理 GPU 张量，避免设备不匹配
            if isinstance(sample_weights, torch.Tensor):
                # 如果已经是张量，直接使用
                target_weights = sample_weights.detach()
            else:
                # 如果不是张量，创建一个并移动到正确的设备
                target_weights = torch.tensor(sample_weights, device=policy_outputs.device, dtype=torch.float)
            
            rl_loss = F.mse_loss(policy_outputs.squeeze(), target_weights)
        
        # 合并损失
        total_loss = main_loss + 0.5 * spec_loss + self.rl_weight * rl_loss
        
        return total_loss, main_loss, spec_loss, rl_loss

# 带强化学习的梯度解冻训练函数
def train_with_reinforcement(model, policy_net, train_loader, val_loader, test_loader, 
                              criterion, device, sampler, train_dataset,
                              num_epochs=40, stages=3, patience=7, eval_interval=1,
                              rl_start_epoch=5):
    # 初始化
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    scaler = GradScaler()  # 混合精度训练
    
    # 第1阶段：仅训练分类头
    print("\n===== 第1阶段：训练分类头 =====")
    # 冻结特征提取器
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # 设置优化器
    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + 
        list(model.specialized_classifier.parameters()) + 
        list(model.cbam.parameters()) + 
        list(model.feature_extractor.parameters()), 
        lr=0.001, 
        weight_decay=0.01
    )
    
    # 策略网络优化器
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=0.0005)
    
    # 学习率调度器 - 使用OneCycleLR
    steps_per_epoch = len(train_loader)
    epochs_per_stage = num_epochs // stages
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # 训练第1阶段
    for epoch in range(epochs_per_stage):
        # 训练
        train_loss, train_acc = train_one_epoch_with_rl(
            model, policy_net, train_loader, criterion, optimizer, policy_optimizer, 
            scheduler, scaler, device, train_dataset, sampler, 
            use_rl=(epoch >= rl_start_epoch)
        )
        
        # 重建数据集以移除无效文件引用
        train_loader, train_dataset = rebuild_dataset_after_epoch(train_loader, train_dataset)
        
        # 定期验证
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # 保存历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"第1阶段 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
                  f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience} 个epoch没有改善")
                    model.load_state_dict(best_model_state)
                    break
    
    # 第2阶段：部分解冻特征提取器
    print("\n===== 第2阶段：部分解冻特征提取器 =====")
    
    # 获取要解冻的层
    if hasattr(model.base_model, 'stages'):
        # 对于ConvNeXt模型
        layers_to_unfreeze = list(model.base_model.stages[-2:])
    elif isinstance(model.base_model, models.resnet.ResNet):
        # 对于ResNet模型
        layers_to_unfreeze = [model.base_model.layer3, model.base_model.layer4]
    else:
        # 对于其他模型
        modules = list(model.base_model.modules())
        layers_to_unfreeze = modules[-20:-5]  # 解冻最后几层
    
    # 解冻选定的层
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
    
    # 设置具有较低学习率的优化器
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': 0.0005},
        {'params': model.specialized_classifier.parameters(), 'lr': 0.0005},
        {'params': model.cbam.parameters(), 'lr': 0.0005},
        {'params': model.feature_extractor.parameters(), 'lr': 0.0005}
    ]
    
    # 为解冻层添加参数组
    for i, layer in enumerate(layers_to_unfreeze):
        param_groups.append({
            'params': layer.parameters(), 
            'lr': 0.0001 if i < len(layers_to_unfreeze)//2 else 0.0002
        })
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # 更新学习率调度器
    max_lrs = [0.0005, 0.0005, 0.0005, 0.0005] + [0.0001 if i < len(layers_to_unfreeze)//2 else 0.0002 for i in range(len(layers_to_unfreeze))]
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=max_lrs, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # 重置早停计数器
    patience_counter = 0
    
    # 训练第2阶段
    for epoch in range(epochs_per_stage):
        # 训练
        train_loss, train_acc = train_one_epoch_with_rl(
            model, policy_net, train_loader, criterion, optimizer, policy_optimizer, 
            scheduler, scaler, device, train_dataset, sampler, 
            use_rl=True
        )
        
        # 重建数据集以移除无效文件引用
        train_loader, train_dataset = rebuild_dataset_after_epoch(train_loader, train_dataset)
        
        # 定期验证
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # 保存历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"第2阶段 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
                  f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience} 个epoch没有改善")
                    model.load_state_dict(best_model_state)
                    break
    
    # 第3阶段：完全解冻模型并微调
    print("\n===== 第3阶段：完全解冻模型 =====")
    # 解冻所有层
    for param in model.parameters():
        param.requires_grad = True
    
    # 设置很小学习率的优化器
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 0.0001},
        {'params': model.specialized_classifier.parameters(), 'lr': 0.0001},
        {'params': model.cbam.parameters(), 'lr': 0.0001},
        {'params': model.feature_extractor.parameters(), 'lr': 0.0001},
        {'params': model.base_model.parameters(), 'lr': 0.00002}
    ], weight_decay=0.01)
    
    # 更新学习率调度器
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0001, 0.0001, 0.0001, 0.0001, 0.00002], 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # 重置早停计数器
    patience_counter = 0
    
    # 训练第3阶段
    for epoch in range(epochs_per_stage):
        # 训练
        train_loss, train_acc = train_one_epoch_with_rl(
            model, policy_net, train_loader, criterion, optimizer, policy_optimizer, 
            scheduler, scaler, device, train_dataset, sampler, 
            use_rl=True
        )
        
        # 重建数据集以移除无效文件引用
        train_loader, train_dataset = rebuild_dataset_after_epoch(train_loader, train_dataset)
        
        # 定期验证
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # 保存历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"第3阶段 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
                  f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: {patience} 个epoch没有改善")
                    model.load_state_dict(best_model_state)
                    break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估测试集性能
    test_loss, test_acc, test_preds, test_targets, test_probs = validate(
        model, test_loader, criterion, device
    )
    print(f"\n最终测试集准确率: {test_acc:.4f}")
    
    # 测试时增强评估
    tta_acc = test_time_augmentation(model, test_loader, device, num_augmentations=10)
    print(f"测试时增强后准确率: {tta_acc:.4f}")
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}, 测试准确率: {test_acc:.4f}, TTA准确率: {tta_acc:.4f}")
    return model, history, best_val_acc, test_acc, tta_acc

# 带强化学习的单个epoch训练函数
def train_one_epoch_with_rl(model, policy_net, dataloader, criterion, optimizer, policy_optimizer, 
                            scheduler, scaler, device, train_dataset, sampler, 
                            use_rl=False, final_tuning=False):
    model.train()
    policy_net.train()
    running_loss = 0.0
    running_main_loss = 0.0
    running_spec_loss = 0.0
    running_rl_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm进度条
    progress_bar = tqdm(dataloader, desc="训练")
    
    # 收集样本错误以更新权重
    sample_losses = {}
    
    for inputs, targets, sample_weights, indices in progress_bar:
        # 移动数据到GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        
        # 混合精度前向传播
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            # 主模型前向传播
            main_outputs, specialized_outputs, features = model(inputs)
            
            # 样本权重 - 可能来自数据集或策略网络
            if use_rl:
                # 使用策略网络预测样本重要性
                policy_outputs = policy_net(features.detach())
                
                # 损失计算
                loss, main_loss, spec_loss, rl_loss = criterion(
                    main_outputs, specialized_outputs, targets, 
                    policy_outputs=policy_outputs, 
                    sample_weights=sample_weights
                )
                
                # 收集错误样本用于RL反馈
                if final_tuning:  # 在最终微调阶段更激进地关注错误
                    with torch.no_grad():
                        _, predicted = main_outputs.max(1)
                        for i, (pred, target, idx) in enumerate(zip(predicted, targets, indices)):
                            # 计算错误程度 (0 = 正确, 1 = 错误)
                            error = 0.0 if pred == target else 1.0
                            # 在错误的情况下使用softmax得分
                            if error > 0:
                                softmax_scores = F.softmax(main_outputs, dim=1)[i]
                                wrong_confidence = softmax_scores[pred].item()
                                right_confidence = softmax_scores[target].item()
                                # 置信错误更受惩罚
                                if wrong_confidence > 0.5:
                                    error = 1.5  # 增强对高置信度错误的惩罚
                                # 目标类低置信度更受惩罚    
                                if right_confidence < 0.2:
                                    error += 0.5
                            # 存储样本错误率
                            sample_losses[idx.item()] = error
                else:
                    # 正常训练期间的错误收集
                    with torch.no_grad():
                        _, predicted = main_outputs.max(1)
                        probs = F.softmax(main_outputs, dim=1)
                        for i, (pred, target, idx) in enumerate(zip(predicted, targets, indices)):
                            # 计算错误程度 (0 = 正确, >0 = 错误)
                            if pred == target:
                                target_prob = probs[i, target].item()
                                # 正确但置信度低的样本
                                error = max(0, 0.7 - target_prob)
                            else:
                                # 错误样本 - 困难程度由目标的概率决定
                                target_prob = probs[i, target].item()
                                error = 1.0 - target_prob  # 目标概率越低，错误越大
                            # 存储样本错误率
                            sample_losses[idx.item()] = error
            else:
                # 不使用RL - 标准训练
                loss, main_loss, spec_loss, rl_loss = criterion(
                    main_outputs, specialized_outputs, targets, 
                    policy_outputs=None, 
                    sample_weights=sample_weights
                )
        
        # 混合精度反向传播和优化
        optimizer.zero_grad(set_to_none=True)
        if use_rl:
            policy_optimizer.zero_grad(set_to_none=True)
            
        scaler.scale(loss).backward()
        
        # 梯度裁剪防止爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if use_rl:
            scaler.unscale_(policy_optimizer)
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        if use_rl:
            scaler.step(policy_optimizer)
            
        scaler.update()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        running_main_loss += main_loss.item() * inputs.size(0)
        running_spec_loss += spec_loss.item() * inputs.size(0)
        running_rl_loss += rl_loss.item() * inputs.size(0) if isinstance(rl_loss, torch.Tensor) else 0
        
        _, predicted = main_outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # 释放不必要的内存
        torch.cuda.empty_cache()
    
    # 更新样本权重
    if use_rl and len(sample_losses) > 0:
        # 获取样本指标
        indices = list(sample_losses.keys())
        rewards = list(sample_losses.values())
        
        # 更新数据集样本权重
        train_dataset.update_sample_weights(indices, rewards)
        
        # 如果使用了采样器，则更新采样器
        if sampler is not None:
            sampler.update()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # 输出损失细节
    print(f"总损失: {epoch_loss:.4f}, 主损失: {running_main_loss/total:.4f}, "
          f"特殊损失: {running_spec_loss/total:.4f}, RL损失: {running_rl_loss/total:.4f}")
    
    return epoch_loss, epoch_acc

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, targets, _, _ in tqdm(dataloader, desc="验证中"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播
                outputs, _, _ = model(inputs)
                loss, _, _, _ = criterion(outputs, None, targets)
                
                # 计算概率
                probs = F.softmax(outputs, dim=1)
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets, all_probs

# 测试时增强
def test_time_augmentation(model, dataloader, device, num_augmentations=10):
    model.eval()
    correct = 0
    total = 0
    
    # 创建测试时增强变换
    tta_transforms = [
        transforms.Compose([
            transforms.RandomAffine(degrees=(-5, 5), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) 
        for _ in range(num_augmentations)
    ]
    
    with torch.no_grad():
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, targets, _, _ in tqdm(dataloader, desc="测试时增强"):
                batch_size = inputs.size(0)
                targets = targets.to(device)
                
                # 直接使用原始输入进行一次预测
                inputs = inputs.to(device)
                outputs, _, _ = model(inputs)
                
                # 存储所有预测的总和
                all_outputs = outputs.clone()
                
                # 应用不同的增强并累加预测结果
                for i in range(batch_size):
                    img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                    img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
                    img = img.astype(np.uint8)
                    img = Image.fromarray(img)
                    
                    # 应用多个增强
                    for transform in tta_transforms:
                        # 应用变换
                        aug_img = transform(img).unsqueeze(0).to(device)
                        # 获取预测
                        aug_outputs, _, _ = model(aug_img)
                        # 累加预测
                        all_outputs[i] += aug_outputs[0]
                
                # 平均预测结果 (原始 + num_augmentations)
                all_outputs = all_outputs / (num_augmentations + 1)
                
                # 获取最终预测
                _, predicted = all_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    tta_acc = correct / total
    return tta_acc

# GPU兼容性测试函数
def test_gpu_compatibility():
    """测试GPU兼容性和基本模型操作"""
    print("测试GPU兼容性...")
    
    try:
        # 测试基本CUDA操作
        x = torch.randn(16, 3, 224, 224).cuda()
        conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
        y = conv(x)
        print("基本CUDA操作正常")
        
        # 测试基本模型
        resnet = models.resnet18(weights=None).cuda()
        out = resnet(x)
        print("ResNet模型正常")
        
        print("GPU兼容性测试完成")
        return True
    except Exception as e:
        print(f"GPU兼容性测试失败: {e}")
        return False

# 主函数
def main():
    # 设置中文字体
    setup_chinese_font()
    
    # 设置随机种子
    set_seed()
    
    # 测试GPU兼容性
    if not test_gpu_compatibility():
        print("请解决GPU兼容性问题后再继续")
        return
    
    # 获取情绪类别
    emotion_categories = get_emotion_categories()
    num_classes = len(emotion_categories)
    print(f"情绪类别: {emotion_categories}, 类别数量: {num_classes}")
    
    # 使用最新准确率信息优化训练
    class_accuracies = {
        'Anger': 0.7362,
        'Confusion': 0.4353,
        'Contempt': 0.4518,
        'Disgust': 0.4653,
        'Happiness': 0.8254,
        'Neutral': 0.7044,
        'Sadness': 0.6738,
        'Surprise': 0.7254
    }
    
    # 设置训练参数
    batch_size = 24  # 减小批量大小以获得更好的泛化能力
    num_workers = 4  # 使用4个工作进程
    num_epochs = 60  # 增加训练轮次
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, sampler, train_dataset = create_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers,
        emotion_categories=emotion_categories,
        class_accuracies=class_accuracies
    )
    
    # 创建模型 - 使用更兼容的骨干网络
    model = AdvancedEmotionModel(
        num_classes=num_classes,
        dropout_rates=[0.5, 0.4, 0.3],
        backbone='resnet50'  # 使用ResNet50
    )
    model = model.to(device)
    
    # 创建强化学习策略网络
    policy_net = PolicyNetwork(feature_dim=512, hidden_dim=128)
    policy_net = policy_net.to(device)
    
    # 设置基于准确率的类权重
    _, weights = calculate_class_weights(
        train_dir, 
        emotion_categories, 
        accuracy_dict=class_accuracies,
        beta=0.9999
    )
    weights = weights.to(device)
    
    # 增加对低准确率类别的关注
    gamma = 2.5  # 增加gamma值以强调难以分类的样本
    
    # 使用改进的混合损失函数
    criterion = CompositeLoss(
        num_classes=num_classes,
        alpha=weights,
        gamma=gamma,
        smoothing=0.1,
        focal_weight=0.7,
        smooth_weight=0.3,
        rl_weight=0.2
    )
    
    # 使用强化学习的分阶段训练
    model, history, best_val_acc, test_acc, tta_acc = train_with_reinforcement(
        model, policy_net, train_loader, val_loader, test_loader, criterion, device, 
        sampler, train_dataset, num_epochs=num_epochs, stages=3, patience=7, eval_interval=1,
        rl_start_epoch=5
    )
    
    # 保存最佳模型
    model_save_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\models"
    # 确保目标目录存在  
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'emotion_rl_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'policy_state_dict': policy_net.state_dict(),
        'emotion_categories': emotion_categories
    }, model_save_path)
    print(f"模型保存至: {model_save_path}")
    
    # 绘制训练历史
    plot_training_history(history, data_root)
    
    # 训练第二个模型 - 使用不同骨干网络的互补模型
    print("\n训练第二个互补模型...")
    
    model2 = AdvancedEmotionModel(
        num_classes=num_classes,
        dropout_rates=[0.5, 0.4, 0.3],
        backbone='efficientnet_v2_s'  # 使用不同骨干网络
    )
    model2 = model2.to(device)
    
    policy_net2 = PolicyNetwork(feature_dim=512, hidden_dim=128)
    policy_net2 = policy_net2.to(device)
    
    # 训练第二个模型
    model2, history2, best_val_acc2, test_acc2, tta_acc2 = train_with_reinforcement(
        model2, policy_net2, train_loader, val_loader, test_loader, criterion, device, 
        sampler, train_dataset, num_epochs=num_epochs, stages=3, patience=7, eval_interval=1,
        rl_start_epoch=5
    )
    
    # 保存第二个模型
    model2_save_path = os.path.join(model_save_dir, 'emotion_rl_model2.pth')
    torch.save({
        'model_state_dict': model2.state_dict(),
        'policy_state_dict': policy_net2.state_dict(),
        'emotion_categories': emotion_categories
    }, model2_save_path)
    print(f"第二个模型保存至: {model2_save_path}")
    
    # 最终性能报告
    print("\n最终性能汇总:")
    print(f"模型1 - 验证准确率: {best_val_acc:.4f}, 测试准确率: {test_acc:.4f}, TTA准确率: {tta_acc:.4f}")
    print(f"模型2 - 验证准确率: {best_val_acc2:.4f}, 测试准确率: {test_acc2:.4f}, TTA准确率: {tta_acc2:.4f}")

# 绘制训练历史
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='训练损失')
    plt.plot(history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('验证时的Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='训练准确率')
    plt.plot(history['val_acc'], 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('验证时的Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

# 确保多处理正确启动
if __name__ == '__main__':
    # 添加多处理支持
    multiprocessing.freeze_support()
    
    # 运行主函数
    main()