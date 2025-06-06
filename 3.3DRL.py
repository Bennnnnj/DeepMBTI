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
import psutil  # 用于监控系统资源

# 设置CUDA错误同步报告，更容易调试
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

# ==================== 基础工具函数 ====================

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

# 设置数据路径 - 根据实际情况调整
data_root = r"C:\Users\lnasl\Desktop\DeepMBTI\data\emotion"
split_dir = os.path.join(data_root, "split")
train_dir = os.path.join(split_dir, "train")
val_dir = os.path.join(split_dir, "val")
test_dir = os.path.join(split_dir, "test")

# ==================== GPU优化相关函数 ====================

# 优化GPU内存使用和CUDA配置
def optimize_gpu_settings(memory_fraction=0.85):
    """配置GPU以获得最佳性能"""
    if torch.cuda.is_available():
        # 释放缓存
        torch.cuda.empty_cache()
        
        # 设置CUDA设备
        device = torch.device("cuda")
        
        # 限制PyTorch保留的GPU内存比例
        try:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"GPU内存使用限制设为: {memory_fraction*100}%")
        except:
            print("无法设置GPU内存限制")
        
        # 启用CUDA优化
        torch.backends.cudnn.benchmark = True
        
        # 打印GPU信息
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"当前占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        return device
    else:
        print("未检测到可用的GPU，将使用CPU")
        return torch.device("cpu")

# 限制CPU使用率
def limit_cpu_usage(cpu_percent=80):
    """限制程序使用的CPU核心数，降低CPU负载"""
    try:
        cpu_count = os.cpu_count()
        cpu_limit = max(1, int(cpu_count * cpu_percent / 100))
        
        # 设置PyTorch内部使用的线程数
        torch.set_num_threads(cpu_limit)
        
        # 对于Windows系统，可以设置进程亲和性
        if os.name == 'nt':
            p = psutil.Process()
            p.cpu_affinity(list(range(cpu_limit)))
            print(f"CPU使用限制为{cpu_limit}/{cpu_count}个核心 ({cpu_percent}%)")
        elif os.name == 'posix':  # Linux/Mac
            # 降低进程优先级
            os.nice(10)
            print(f"已降低进程优先级并限制PyTorch线程数为{cpu_limit}")
    except Exception as e:
        print(f"限制CPU使用失败: {e}")

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
        
        # 测试GPU内存和利用率
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"当前占用: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
        
        # 测试CUDA流
        s1 = torch.cuda.Stream()
        s2 = torch.cuda.Stream()
        with torch.cuda.stream(s1):
            y1 = conv(x)
        with torch.cuda.stream(s2):
            y2 = conv(x)
        torch.cuda.synchronize()
        print("CUDA流操作正常")
        
        print("GPU兼容性测试完成")
        return True
    except Exception as e:
        print(f"GPU兼容性测试失败: {e}")
        return False

# ==================== 数据集优化 ====================

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

# 获取优化的数据转换 - 减少CPU开销
def get_transforms(mode='train', target_classes=None):
    """获取优化的数据转换，减少CPU密集操作"""
    if mode == 'train':
        # 减少CPU操作，只保留必要的基础变换，其余移至GPU
        base_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),  # 减少翻转概率
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 简化的增强变换
        enhanced_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return base_transforms, enhanced_transforms
    else:
        # 用于验证和测试的简化转换
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# GPU上的数据增强 - 将增强操作从CPU转移到GPU以减轻CPU负担
class GPUDataAugmentation(nn.Module):
    """在GPU上执行数据增强"""
    def __init__(self, p_flip=0.5, p_color=0.7, p_erase=0.3):
        super(GPUDataAugmentation, self).__init__()
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_erase = p_erase
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # 水平翻转
        if self.p_flip > 0:
            flip_mask = torch.rand(batch_size, device=device) < self.p_flip
            x[flip_mask] = torch.flip(x[flip_mask], dims=[-1])  # 水平翻转
        
        # 亮度和对比度调整
        if self.p_color > 0:
            brightness_mask = torch.rand(batch_size, device=device) < self.p_color
            if brightness_mask.sum() > 0:
                brightness = 0.8 + 0.4 * torch.rand(brightness_mask.sum(), 1, 1, 1, device=device)
                x[brightness_mask] = x[brightness_mask] * brightness
            
            contrast_mask = torch.rand(batch_size, device=device) < self.p_color
            if contrast_mask.sum() > 0:
                contrast = 0.8 + 0.4 * torch.rand(contrast_mask.sum(), 1, 1, 1, device=device)
                mean = torch.mean(x[contrast_mask], dim=[-3, -2, -1], keepdim=True)
                x[contrast_mask] = (x[contrast_mask] - mean) * contrast + mean
        
        # 随机擦除
        if self.p_erase > 0:
            erase_mask = torch.rand(batch_size, device=device) < self.p_erase
            if erase_mask.sum() > 0:
                h, w = x.size(-2), x.size(-1)
                for idx in torch.where(erase_mask)[0]:
                    # 随机确定擦除区域
                    erase_size_h = int(h * (0.02 + 0.18 * torch.rand(1, device=device)))
                    erase_size_w = int(w * (0.02 + 0.18 * torch.rand(1, device=device)))
                    erase_pos_h = int((h - erase_size_h) * torch.rand(1, device=device))
                    erase_pos_w = int((w - erase_size_w) * torch.rand(1, device=device))
                    
                    # 应用擦除
                    x[idx, :, erase_pos_h:erase_pos_h+erase_size_h, 
                           erase_pos_w:erase_pos_w+erase_size_w] = 0
        
        # 确保值在合理范围内
        x = torch.clamp(x, 0, 1)
        
        return x

# 优化的情绪数据集 - 减少CPU预处理负担
class OptimizedEmotionDataset(Dataset):
    def __init__(self, data_dir, emotion_categories, transform=None, enhanced_transform=None, 
                 target_classes=None, adaptive_augment=False, mixer_prob=0.0):
        self.data_dir = data_dir
        self.emotion_categories = emotion_categories
        self.transform = transform
        self.enhanced_transform = enhanced_transform
        self.target_classes = target_classes or []
        self.adaptive_augment = adaptive_augment
        self.mixer_prob = mixer_prob  # 在GPU上实现mixer，这里设为0
        
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
        """优化的数据集加载，减少文件检查开销"""
        print("加载数据集...")
        
        for emotion in self.emotion_categories:
            emotion_dir = os.path.join(self.data_dir, emotion)
            emotion_idx = self.emotion_to_idx[emotion]
            
            if not os.path.exists(emotion_dir):
                print(f"警告: 目录不存在: {emotion_dir}")
                continue
                
            # 只检查文件扩展名，不验证图像完整性（提高加载速度）
            image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            self.samples_by_class[emotion] = image_files
            self.samples.extend(image_files)
            self.targets.extend([emotion_idx] * len(image_files))
            
        print(f"数据集加载完成，样本总数: {len(self.samples)}")
    
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
        
        # 尝试读取图像，如果失败则返回占位图像
        try:
            # 读取图像
            img = Image.open(img_path).convert('RGB')
            
            # 对低准确率类别应用特殊增强
            if self.adaptive_augment and emotion in self.target_classes and self.enhanced_transform is not None:
                img = self.enhanced_transform(img)
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

# 优化的数据加载器创建函数
def create_dataloaders(batch_size=128, num_workers=4, emotion_categories=None, class_accuracies=None):
    """创建优化的数据加载器，降低CPU使用率"""
    # 根据准确率确定目标类
    if class_accuracies:
        target_classes = [emotion for emotion, acc in class_accuracies.items() if acc < 0.7]
        print(f"目标优化类别: {target_classes}")
    else:
        target_classes = ["Confusion", "Contempt", "Disgust"]  # 默认目标类
    
    # 获取基础和增强变换 - 简化CPU预处理
    base_transforms, enhanced_transforms = get_transforms(mode='train', target_classes=target_classes)
    val_test_transform = get_transforms(mode='val')
    
    # 创建优化的数据集
    train_dataset = OptimizedEmotionDataset(
        train_dir, emotion_categories, 
        transform=base_transforms, 
        enhanced_transform=enhanced_transforms,
        target_classes=target_classes, 
        adaptive_augment=False,  # 禁用CPU上的自适应增强
        mixer_prob=0.0  # 禁用CPU上的混合器
    )
    
    val_dataset = OptimizedEmotionDataset(
        val_dir, emotion_categories, 
        transform=val_test_transform, 
        enhanced_transform=None,
        target_classes=None, 
        adaptive_augment=False
    )
    
    test_dataset = OptimizedEmotionDataset(
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
    
    # 创建数据加载器 - 优化参数以平衡CPU与GPU
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,  # 减少工作线程数以降低CPU压力
        pin_memory=True,
        prefetch_factor=2,  # 减小预取因子
        persistent_workers=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2,  # 验证时可以用更大的批量
        shuffle=False, 
        num_workers=max(2, num_workers//2),  # 验证时使用更少线程
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size * 2,  # 测试时可以用更大的批量
        shuffle=False, 
        num_workers=max(2, num_workers//2),  # 测试时使用更少线程
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, sampler, train_dataset

# ==================== 模型架构 ====================

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
        
        # 选择基础模型
        if backbone == 'efficientnet_v2_s':
            self.base_model = models.efficientnet_v2_s(weights='DEFAULT')
            last_channel = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.base_model = models.resnet50(weights='DEFAULT')
            last_channel = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif backbone == 'resnet101':
            self.base_model = models.resnet101(weights='DEFAULT')
            last_channel = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 添加注意力模块
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # 全局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(512, num_classes)
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
        # 特征提取
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
            # 尝试通用方法
            features = self.base_model(x)
        
        # 提取中间特征用于强化学习
        extracted_features = self.feature_extractor(features)
        
        # 分类器输出
        outputs = self.classifier(features)
        
        # 如果仅提取特征
        if extract_features:
            return outputs, extracted_features
        
        return outputs, None, extracted_features

# Vision Transformer模型 - 新增
class EmotionViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(EmotionViT, self).__init__()
        
        # 使用预训练的ViT模型
        try:
            import timm
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
            
            # 获取分类头的输入特征维度
            self.in_features = self.base_model.head.in_features
            # 替换分类头为Identity
            self.base_model.head = nn.Identity()
        except ImportError:
            print("未安装timm库，使用替代模型")
            # 使用resnet作为替代
            self.base_model = models.resnet50(weights='DEFAULT')
            self.in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        
        # 情感分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.in_features),
            nn.Dropout(0.5),
            nn.Linear(self.in_features, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # 特征提取器 - 用于强化学习
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512)
        )
    
    def forward(self, x, extract_features=False):
        # 特征提取
        features = self.base_model(x)
        
        # 提取中间特征
        extracted_features = self.feature_extractor(features)
        
        # 分类
        outputs = self.classifier(features)
        
        if extract_features:
            return outputs, extracted_features
        
        # 与其他模型保持一致的输出格式
        return outputs, None, extracted_features

# 模型集成 - 新增
class EmotionEnsemble(nn.Module):
    def __init__(self, models, num_classes):
        super(EmotionEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = nn.Parameter(torch.ones(len(models)) / len(models))
        
        # 集合所有模型的特征维度
        feature_dim = 512 * len(models)
        
        # 集成分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        all_outputs = []
        all_features = []
        
        # 获取每个模型的输出和特征
        for model in self.models:
            output, _, feature = model(x)
            all_outputs.append(output)
            all_features.append(feature)
        
        # 加权平均预测
        weights = F.softmax(self.weights, dim=0)
        ensemble_output = sum(w * out for w, out in zip(weights, all_outputs))
        
        # 连接所有特征
        combined_features = torch.cat(all_features, dim=1)
        
        # 应用集成分类器获取最终分类结果
        final_output = self.classifier(combined_features)
        
        return final_output, None, combined_features

# ==================== 损失函数 ====================

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

# 简化的损失函数 - 优化性能
# 修改简化的损失函数以正确处理policy_outputs
class CompositeLoss(nn.Module):
    def __init__(self, num_classes, alpha=None, gamma=2.0, smoothing=0.1, 
                 focal_weight=0.7, smooth_weight=0.3):
        super(CompositeLoss, self).__init__()
        self.focal_loss = AdvancedFocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing = LabelSmoothingLoss(classes=num_classes, smoothing=smoothing, weight=alpha)
        
        self.focal_weight = focal_weight
        self.smooth_weight = smooth_weight
        
    def forward(self, outputs, targets, sample_weights=None, policy_outputs=None, policy_targets=None):
        # 主分类器损失
        focal = self.focal_loss(outputs, targets, sample_weights)
        smooth = self.label_smoothing(outputs, targets, sample_weights)
        loss = self.focal_weight * focal + self.smooth_weight * smooth
        
        # 策略网络损失（如果提供）
        policy_loss = 0.0
        if policy_outputs is not None and policy_targets is not None:
            policy_loss = F.mse_loss(policy_outputs, policy_targets)
            loss = loss + 0.2 * policy_loss
        
        return loss, focal, smooth, policy_loss
# ==================== 训练函数 ====================

# 使用CUDA流水线优化的训练函数
def train_one_epoch_optimized(model, dataloader, criterion, optimizer, 
                             scaler, device, gpu_augment):
    """优化的训练函数，使用CUDA流水线重叠数据传输和计算"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用两个CUDA流并行处理
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # 预加载第一批数据
    batch_iter = iter(dataloader)
    try:
        inputs, targets, sample_weights, indices = next(batch_iter)
    except StopIteration:
        return 0, 0  # 空数据集
    
    # 将第一批数据传输到GPU
    with torch.cuda.stream(stream1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
    
    # 创建进度条
    progress_bar = tqdm(range(len(dataloader)), desc="训练")
    
    for i in progress_bar:
        # 同步上一批数据传输完成
        torch.cuda.current_stream().wait_stream(stream1)
        
        # 准备下一批数据（如果有）
        if i + 1 < len(dataloader):
            try:
                next_inputs, next_targets, next_weights, next_indices = next(batch_iter)
                
                # 在单独的流中异步传输下一批数据
                with torch.cuda.stream(stream2):
                    next_inputs = next_inputs.to(device, non_blocking=True)
                    next_targets = next_targets.to(device, non_blocking=True)
                    next_weights = next_weights.to(device, non_blocking=True)
            except StopIteration:
                next_inputs, next_targets, next_weights, next_indices = None, None, None, None
        else:
            next_inputs, next_targets, next_weights, next_indices = None, None, None, None
        
        # 在GPU上进行数据增强
        inputs = gpu_augment(inputs)
        
        # 主流程中执行计算
        with autocast(device_type='cuda', dtype=torch.float16):
            # 前向传播
            outputs, _, _ = model(inputs)
            loss, _, _, _ = criterion(outputs, targets, sample_weights)
        
        # 反向传播和优化
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
        scaler.scale(loss).backward()
        
        # 梯度裁剪防止爆炸
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 交换流和数据引用，为下一批做准备
        stream1, stream2 = stream2, stream1
        if next_inputs is not None:
            inputs, targets, sample_weights, indices = next_inputs, next_targets, next_weights, next_indices
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': f'{running_loss/total:.4f}',
            'acc': f'{correct/total:.4f}'
        })
        
        # 定期释放未使用的缓存
        if i % 50 == 0:
            torch.cuda.empty_cache()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# 带强化学习的训练函数 - 优化版本
def train_one_epoch_with_rl(model, policy_net, dataloader, criterion, optimizer, policy_optimizer, 
                           scheduler, scaler, device, train_dataset, sampler, 
                           use_rl=False, gpu_augment=None):
    """优化的RL训练函数，将数据增强移至GPU执行"""
    model.train()
    if use_rl:
        policy_net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 收集样本错误以更新权重
    sample_losses = {}
    
    # 进度条
    progress_bar = tqdm(dataloader, desc="训练")
    
    for inputs, targets, sample_weights, indices in progress_bar:
        # 移动数据到GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        sample_weights = sample_weights.to(device, non_blocking=True)
        
        # 在GPU上执行数据增强
        if gpu_augment is not None:
            inputs = gpu_augment(inputs)
        
        # 主模型前向传播和损失计算
        with autocast(device_type='cuda', dtype=torch.float16):
            # 主模型前向传播
            outputs, _, features = model(inputs)
            
            # 主模型损失计算
            model_loss = criterion(outputs, targets, sample_weights)[0]
        
        # 主模型反向传播
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model_loss).backward()
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 主模型优化器步进
        scaler.step(optimizer)
        scaler.update()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 策略网络训练 - 使用单独的前向/反向传播
        if use_rl:
            with torch.no_grad():
                _, predicted = outputs.max(1)
                probs = F.softmax(outputs, dim=1)
                # 计算样本重要性指标（错误或低置信度样本）
                for i, (pred, target, idx) in enumerate(zip(predicted, targets, indices)):
                    if pred == target:
                        target_prob = probs[i, target].item()
                        # 正确但置信度低的样本
                        error = max(0, 0.7 - target_prob)
                    else:
                        # 错误样本
                        target_prob = probs[i, target].item()
                        error = 1.0 - target_prob  # 目标概率越低，错误越大
                    # 存储样本错误率
                    sample_losses[idx.item()] = error
                
                # 提取特征并停止梯度传播
                # 关键修复：将特征从Half转换为Float精度
                features_detached = features.detach().float()
            
            # 策略网络单独训练 - 不使用混合精度
            policy_outputs = policy_net(features_detached)
            policy_targets = torch.tensor([sample_losses.get(idx.item(), 0.5) 
                                         for idx in indices], device=device).float().unsqueeze(1)
            
            # 策略网络损失 - 均方误差
            policy_loss = F.mse_loss(policy_outputs, policy_targets)
            
            # 策略网络反向传播
            policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
            policy_optimizer.step()
        
        # 统计
        running_loss += model_loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
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
    
    # 如果使用RL，更新样本权重
    if use_rl and len(sample_losses) > 0:
        indices = list(sample_losses.keys())
        rewards = list(sample_losses.values())
        
        # 更新数据集样本权重
        train_dataset.update_sample_weights(indices, rewards)
        
        # 如果使用了采样器，则更新采样器
        if sampler is not None:
            sampler.update()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# 知识蒸馏训练函数
def train_with_knowledge_distillation(teacher_model, student_model, train_loader, val_loader,
                                     criterion, device, num_epochs=10, 
                                     temperature=3.0, alpha=0.5):
    """知识蒸馏训练 - 将大模型知识转移到小模型"""
    teacher_model.eval()  # 教师模型设为评估模式
    student_model.train() # 学生模型设为训练模式
    
    optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = GradScaler()  # 混合精度训练
    
    # 创建GPU数据增强
    gpu_augment = GPUDataAugmentation().to(device)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_loader), eta_min=1e-6)
    
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"蒸馏训练 Epoch {epoch+1}/{num_epochs}")
        
        for inputs, targets, _, _ in progress_bar:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            # 在GPU上进行数据增强
            inputs = gpu_augment(inputs)
            
            # 获取教师模型的软标签
            with torch.no_grad():
                teacher_outputs, _, _ = teacher_model(inputs)
                teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)
            
            # 学生模型前向传播
            with autocast(device_type='cuda', dtype=torch.float16):
                student_outputs, _, _ = student_model(inputs)
                
                # 硬标签损失（监督信号）
                hard_loss = F.cross_entropy(student_outputs, targets)
                
                # 软标签损失（蒸馏信号）
                soft_loss = F.kl_div(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    teacher_probs,
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # 组合损失
                loss = alpha * hard_loss + (1 - alpha) * soft_loss
            
            # 反向传播与优化
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # 更新学习率
            scheduler.step()
            
            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = student_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_loss = running_loss / total
        train_acc = correct / total
        
        # 验证
        val_loss, val_acc = evaluate(student_model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - 训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(student_model.state_dict())
            print(f"保存新的最佳模型，验证准确率: {val_acc:.4f}")
    
    # 恢复最佳模型
    student_model.load_state_dict(best_model)
    
    return student_model, best_val_acc

# 验证函数
def evaluate(model, dataloader, criterion, device):
    """模型评估函数"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            for inputs, targets, _, _ in tqdm(dataloader, desc="验证中"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # 前向传播
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, targets)[0]
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# 测试函数 - 包括混淆矩阵和类别性能
def test_model(model, test_loader, criterion, device, emotion_categories):
    """详细测试模型性能，包括每个类别的指标"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            for inputs, targets, _, _ in tqdm(test_loader, desc="测试中"):
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # 前向传播
                outputs, _, _ = model(inputs)
                loss = criterion(outputs, targets)[0]
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 收集预测和目标
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
    
    test_loss = running_loss / total
    test_acc = correct / total
    
    print(f"\n测试结果:")
    print(f"损失: {test_loss:.4f}, 准确率: {test_acc:.4f}")
    
    # 计算每个类别的性能
    print("\n类别性能:")
    class_accuracies = {}
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    
    for i, emotion in enumerate(emotion_categories):
        # 类别准确率 = 对角线元素 / 该类别的总样本数
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        class_accuracies[emotion] = class_acc
        print(f"{emotion}: {class_acc:.4f}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_categories, 
                yticklabels=emotion_categories)
    plt.xlabel('预测')
    plt.ylabel('真实')
    plt.title('情绪识别混淆矩阵')
    plt.tight_layout()
    plt.savefig(os.path.join(data_root, 'confusion_matrix.png'))
    plt.close()
    
    # 生成分类报告
    cr = classification_report(all_targets, all_preds, 
                              target_names=emotion_categories, 
                              digits=4)
    print("\n分类报告:")
    print(cr)
    
    return test_loss, test_acc, class_accuracies

# 测试时增强
def test_time_augmentation(model, dataloader, device, gpu_augment, num_augmentations=5):
    """测试时增强 - 用多个增强版本进行推理并平均结果"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        with autocast(device_type='cuda', dtype=torch.float16):
            for inputs, targets, _, _ in tqdm(dataloader, desc="测试时增强"):
                batch_size = inputs.size(0)
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 原始预测
                outputs, _, _ = model(inputs)
                all_outputs = outputs.clone()
                
                # 应用多次GPU增强并累加预测
                for _ in range(num_augmentations):
                    # 应用GPU增强
                    aug_inputs = gpu_augment(inputs)
                    # 获取增强预测
                    aug_outputs, _, _ = model(aug_inputs)
                    # 累加预测
                    all_outputs += aug_outputs
                
                # 平均预测 (原始 + 增强)
                all_outputs = all_outputs / (num_augmentations + 1)
                
                # 统计
                _, predicted = all_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
    
    tta_acc = correct / total
    print(f"测试时增强后准确率: {tta_acc:.4f}")
    
    return tta_acc

# ==================== 模型训练流程 ====================

# 分阶段训练函数 - 优化版
def train_stage(model, policy_net, train_loader, val_loader, criterion, 
               device, sampler, train_dataset, scaler, stage_name="训练阶段",
               lr=0.001, freeze_layers=None, num_epochs=10, patience=3, gpu_augment=None):
    """优化的分阶段训练函数"""
    print(f"\n===== {stage_name} =====")
    
    # 如果提供了要冻结的层，冻结它们
    if freeze_layers is not None:
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    # 设置优化器
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' in n], 'lr': lr},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and 'classifier' not in n], 
         'lr': lr * 0.1}  # 其他层使用较小的学习率
    ], weight_decay=0.01)
    
    # 策略网络优化器
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr * 0.5)
    
    # 学习率调度器
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[lr, lr * 0.1], 
        steps_per_epoch=steps_per_epoch, 
        epochs=num_epochs,
        pct_start=0.3
    )
    
    # 训练循环
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # 训练
        train_loss, train_acc = train_one_epoch_with_rl(
            model, policy_net, train_loader, criterion, optimizer, policy_optimizer, 
            scheduler, scaler, device, train_dataset, sampler, 
            use_rl=True, gpu_augment=gpu_augment
        )
        
        # 重建数据集以移除无效文件引用
        train_loader, train_dataset = rebuild_dataset_after_epoch(train_loader, train_dataset)
        
        # 验证
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # 保存历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"{stage_name} - Epoch {epoch+1}/{num_epochs} - "
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
                break
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, best_val_acc, history

# 带强化学习的三阶段训练流程
def train_with_reinforcement(model, policy_net, train_loader, val_loader, test_loader, 
                             criterion, device, sampler, train_dataset, num_epochs=30, gpu_augment=None):
    # 初始化
    best_val_acc = 0.0
    best_model_state = None
    scaler = GradScaler()  # 混合精度训练
    
    epochs_per_stage = num_epochs // 3
    
    # 第1阶段：只训练分类头
    freeze_layers = ['base_model']
    model, stage1_acc, stage1_history = train_stage(
        model, policy_net, train_loader, val_loader, criterion, 
        device, sampler, train_dataset, scaler,
        stage_name="第1阶段：训练分类头", 
        lr=0.001, 
        freeze_layers=freeze_layers,
        num_epochs=epochs_per_stage,
        patience=3,
        gpu_augment=gpu_augment
    )
    
    # 更新最佳模型
    if stage1_acc > best_val_acc:
        best_val_acc = stage1_acc
        best_model_state = copy.deepcopy(model.state_dict())
    
    # 第2阶段：部分解冻
    if isinstance(model.base_model, models.resnet.ResNet):
        freeze_layers = ['base_model.conv1', 'base_model.bn1', 'base_model.layer1', 'base_model.layer2']
    else:
        # 对于其他类型的模型，通常只冻结前1/3的层
        freeze_layers = ['base_model.0', 'base_model.1', 'base_model.2', 'base_model.3', 'base_model.4']
    
    model, stage2_acc, stage2_history = train_stage(
        model, policy_net, train_loader, val_loader, criterion, 
        device, sampler, train_dataset, scaler,
        stage_name="第2阶段：部分解冻", 
        lr=0.0005, 
        freeze_layers=freeze_layers,
        num_epochs=epochs_per_stage,
        patience=3,
        gpu_augment=gpu_augment
    )
    
    # 更新最佳模型
    if stage2_acc > best_val_acc:
        best_val_acc = stage2_acc
        best_model_state = copy.deepcopy(model.state_dict())
    
    # 第3阶段：完全解冻
    model, stage3_acc, stage3_history = train_stage(
        model, policy_net, train_loader, val_loader, criterion, 
        device, sampler, train_dataset, scaler,
        stage_name="第3阶段：完全解冻", 
        lr=0.0001, 
        freeze_layers=None,  # 没有冻结层
        num_epochs=epochs_per_stage,
        patience=3,
        gpu_augment=gpu_augment
    )
    
    # 更新最佳模型
    if stage3_acc > best_val_acc:
        best_val_acc = stage3_acc
        best_model_state = copy.deepcopy(model.state_dict())
    
    # 合并历史记录
    history = {
        'train_loss': stage1_history['train_loss'] + stage2_history['train_loss'] + stage3_history['train_loss'],
        'train_acc': stage1_history['train_acc'] + stage2_history['train_acc'] + stage3_history['train_acc'],
        'val_loss': stage1_history['val_loss'] + stage2_history['val_loss'] + stage3_history['val_loss'],
        'val_acc': stage1_history['val_acc'] + stage2_history['val_acc'] + stage3_history['val_acc']
    }
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    # 评估测试集性能
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n最终测试集准确率: {test_acc:.4f}")
    
    # 测试时增强评估
    tta_acc = test_time_augmentation(model, test_loader, device, gpu_augment)
    print(f"测试时增强后准确率: {tta_acc:.4f}")
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.4f}, 测试准确率: {test_acc:.4f}, TTA准确率: {tta_acc:.4f}")
    return model, history, best_val_acc, test_acc, tta_acc

# 绘制训练历史
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='训练损失')
    plt.plot(history['val_loss'], 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('周期')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='训练准确率')
    plt.plot(history['val_acc'], 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('周期')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history_optimized.png'))
    plt.close()

# ==================== 主函数 ====================

def main():
    # 设置中文字体
    setup_chinese_font()
    
    # 设置随机种子
    set_seed(42)
    
    # 限制CPU使用率，降低CPU负载
    limit_cpu_usage(cpu_percent=80)
    
    # 优化GPU设置
    device = optimize_gpu_settings(memory_fraction=0.85)
    
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
    
    # 动态调整批量大小和工作线程数
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # 显存大小(GB)
        
        # 根据显存大小调整批量大小
        if gpu_mem > 20:  # 高端GPU (如RTX 3090/4090等)
            batch_size = 128
            num_workers = 4
        elif gpu_mem > 10:  # 中端GPU
            batch_size = 64
            num_workers = 4
        else:  # 低端GPU
            batch_size = 32
            num_workers = 2
            
        print(f"GPU显存: {gpu_mem:.1f}GB, 设置批量大小: {batch_size}, 工作线程: {num_workers}")
    else:
        batch_size = 32
        num_workers = 2
        print("未检测到GPU，使用CPU模式，批量大小: 32, 工作线程: 2")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, sampler, train_dataset = create_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers,
        emotion_categories=emotion_categories,
        class_accuracies=class_accuracies
    )
    
    # 创建GPU数据增强
    gpu_augment = GPUDataAugmentation(p_flip=0.5, p_color=0.7, p_erase=0.3).to(device)
    
    # 创建模型
    print("创建ResNet50模型...")
    resnet_model = AdvancedEmotionModel(
        num_classes=num_classes,
        dropout_rates=[0.5, 0.4, 0.3],
        backbone='resnet50'  # 使用ResNet50而非ResNet101，降低复杂度
    ).to(device)
    
    # 创建强化学习策略网络
    policy_net = PolicyNetwork(feature_dim=512, hidden_dim=128).to(device)
    
    # 设置基于准确率的类权重
    _, weights = calculate_class_weights(
        train_dir, 
        emotion_categories, 
        accuracy_dict=class_accuracies,
        beta=0.9999
    )
    weights = weights.to(device)
    
    # 使用简化的混合损失函数
    criterion = CompositeLoss(
        num_classes=num_classes,
        alpha=weights,
        gamma=2.0,
        smoothing=0.1,
        focal_weight=0.7,
        smooth_weight=0.3
    )
    
    # 使用强化学习的分阶段训练
    print("\n开始训练ResNet50模型...")
    resnet_model, history, best_val_acc, test_acc, tta_acc = train_with_reinforcement(
        resnet_model, policy_net, train_loader, val_loader, test_loader, criterion, device, 
        sampler, train_dataset, num_epochs=30, gpu_augment=gpu_augment
    )
    
    # 保存最佳模型
    model_save_dir = os.path.join(data_root, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    model_save_path = os.path.join(model_save_dir, 'emotion_resnet50_optimized.pth')
    torch.save({
        'model_state_dict': resnet_model.state_dict(),
        'policy_state_dict': policy_net.state_dict(),
        'emotion_categories': emotion_categories
    }, model_save_path)
    print(f"ResNet50模型保存至: {model_save_path}")
    
    # 绘制训练历史
    plot_training_history(history, data_root)
    
    # 在测试集上进行详细测试
    print("\n对ResNet50模型进行详细测试...")
    test_model(resnet_model, test_loader, criterion, device, emotion_categories)
    
    # 创建ViT模型（如果可用）
    try:
        print("\n尝试创建Vision Transformer模型...")
        import timm
        
        vit_model = EmotionViT(num_classes=num_classes).to(device)
        vit_policy_net = PolicyNetwork(feature_dim=512, hidden_dim=128).to(device)
        
        # 训练ViT模型
        print("\n开始训练Vision Transformer模型...")
        vit_model, vit_history, vit_best_val_acc, vit_test_acc, vit_tta_acc = train_with_reinforcement(
            vit_model, vit_policy_net, train_loader, val_loader, test_loader, criterion, device, 
            sampler, train_dataset, num_epochs=20, gpu_augment=gpu_augment
        )
        
        # 保存ViT模型
        vit_save_path = os.path.join(model_save_dir, 'emotion_vit_optimized.pth')
        torch.save({
            'model_state_dict': vit_model.state_dict(),
            'policy_state_dict': vit_policy_net.state_dict(),
            'emotion_categories': emotion_categories
        }, vit_save_path)
        print(f"ViT模型保存至: {vit_save_path}")
        
        # 创建集成模型
        print("\n创建集成模型...")
        ensemble_model = EmotionEnsemble([resnet_model, vit_model], num_classes=num_classes).to(device)
        
        # 微调集成模型
        ensemble_optimizer = optim.AdamW(ensemble_model.parameters(), lr=0.0005, weight_decay=0.01)
        ensemble_scaler = GradScaler()
        
        # 简单微调几个epoch
        for epoch in range(5):
            ensemble_model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            # 训练进度条
            progress_bar = tqdm(train_loader, desc=f"集成训练 Epoch {epoch+1}/5")
            
            for inputs, targets, _, _ in progress_bar:
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                
                # GPU数据增强
                inputs = gpu_augment(inputs)
                
                # 前向传播
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs, _, _ = ensemble_model(inputs)
                    loss = criterion(outputs, targets)[0]
                
                # 反向传播
                ensemble_optimizer.zero_grad(set_to_none=True)
                ensemble_scaler.scale(loss).backward()
                ensemble_scaler.step(ensemble_optimizer)
                ensemble_scaler.update()
                
                # 统计
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{correct/total:.4f}'
                })
            
            # 验证
            val_loss, val_acc = evaluate(ensemble_model, val_loader, criterion, device)
            print(f"Epoch {epoch+1}/5 - 集成模型验证准确率: {val_acc:.4f}")
        
        # 保存集成模型
        ensemble_save_path = os.path.join(model_save_dir, 'emotion_ensemble_optimized.pth')
        torch.save({
            'model_state_dict': ensemble_model.state_dict(),
            'emotion_categories': emotion_categories
        }, ensemble_save_path)
        print(f"集成模型保存至: {ensemble_save_path}")
        
        # 在测试集上测试集成模型
        print("\n测试集成模型性能...")
        test_model(ensemble_model, test_loader, criterion, device, emotion_categories)
        
    except ImportError:
        print("\n未安装timm库，跳过ViT和集成模型训练。请使用 'pip install timm' 安装后再尝试。")
    
    # 尝试知识蒸馏创建轻量级模型
    try:
        print("\n尝试创建轻量级模型并使用知识蒸馏...")
        # 创建小型模型
        small_model = AdvancedEmotionModel(
            num_classes=num_classes,
            dropout_rates=[0.3, 0.2, 0.1],
            backbone='efficientnet_v2_s'  # 使用更轻量的骨干网络
        ).to(device)
        
        # 使用大模型蒸馏小模型
        small_model, small_val_acc = train_with_knowledge_distillation(
            teacher_model=resnet_model,
            student_model=small_model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            num_epochs=10,
            temperature=3.0,
            alpha=0.5
        )
        
        # 保存蒸馏模型
        distill_save_path = os.path.join(model_save_dir, 'emotion_distilled_optimized.pth')
        torch.save({
            'model_state_dict': small_model.state_dict(),
            'emotion_categories': emotion_categories
        }, distill_save_path)
        print(f"蒸馏模型保存至: {distill_save_path}")
        
        # 测试蒸馏模型
        print("\n测试蒸馏模型性能...")
        test_model(small_model, test_loader, criterion, device, emotion_categories)
        
    except Exception as e:
        print(f"\n知识蒸馏过程出错: {e}")
    
    # 性能总结
    print("\n训练完成，性能总结:")
    print(f"ResNet50模型 - 验证准确率: {best_val_acc:.4f}, 测试准确率: {test_acc:.4f}, TTA准确率: {tta_acc:.4f}")
    
    # 释放GPU内存
    torch.cuda.empty_cache()

# 确保多处理正确启动
if __name__ == '__main__':
    # 添加多处理支持
    multiprocessing.freeze_support()
    
    # 运行主函数
    main()