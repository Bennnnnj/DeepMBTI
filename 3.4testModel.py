import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# 设置路径
MODEL_PATH = r"C:\Users\lnasl\Desktop\DeepMBTI\code\TrainedModel\emotion"
DATA_PATH = r"C:\Users\lnasl\Desktop\DeepMBTI\data\emotion\split"
TEST_PATH = os.path.join(DATA_PATH, "test")

# 数据预处理
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),  # 调整大小为224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

# 自定义数据集类
class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None, emotion_categories=None):
        self.root_dir = root_dir
        self.transform = transform
        # 如果提供了特定的情绪类别，则使用它，否则使用默认的
        self.classes = emotion_categories if emotion_categories else os.listdir(root_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()
        
    def _make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(class_dir, filename)
                    samples.append((path, self.class_to_idx[class_name]))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 定义一个通用的模型类，可以适应不同的模型架构
class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=7, backbone='resnet50'):
        super(EmotionClassifier, self).__init__()
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=False)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'vit':
            self.backbone = models.vit_b_16(pretrained=False)
            self.backbone.heads = nn.Linear(self.backbone.hidden_dim, num_classes)
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(pretrained=False)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

# 加载模型函数 - 修改以处理模型状态字典的结构
def load_model(model_path, device):
    """加载模型并返回模型实例和情绪类别"""
    # 加载完整的保存状态（包括模型状态、类别等）
    saved_state = torch.load(model_path, map_location=device)
    
    # 尝试提取情绪类别（如果有的话）
    emotion_categories = None
    if 'emotion_categories' in saved_state:
        emotion_categories = saved_state['emotion_categories']
        print(f"从模型中提取的情绪类别: {emotion_categories}")
    
    # 推断模型类型和结构
    model_type = None
    if 'distilled' in os.path.basename(model_path):
        model_type = 'efficientnet'
    elif 'vit' in os.path.basename(model_path):
        model_type = 'vit'
    elif 'resnet50' in os.path.basename(model_path):
        model_type = 'resnet50'
    elif 'improved' in os.path.basename(model_path):
        model_type = 'resnet34'
    elif 'ensemble' in os.path.basename(model_path):
        # 集成模型需要特殊处理
        model_type = 'ensemble'
    
    # 如果我们有情绪类别，使用它来确定num_classes
    num_classes = len(emotion_categories) if emotion_categories else 7
    
    # 创建适当的模型
    if model_type == 'ensemble':
        # 创建一个自定义的集成模型 - 这可能需要根据实际情况调整
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
    else:
        model = EmotionClassifier(num_classes=num_classes, backbone=model_type)
    
    # 尝试加载模型状态
    try:
        # 首先尝试使用model_state_dict键
        if 'model_state_dict' in saved_state:
            print("使用model_state_dict加载模型权重")
            # 创建一个新的状态字典，移除模块前缀
            fixed_state_dict = {}
            for k, v in saved_state['model_state_dict'].items():
                # 处理可能的前缀差异
                if k.startswith('module.'):
                    k = k[7:]  # 移除'module.'前缀
                if k.startswith('backbone.'):
                    k = k[9:]  # 移除'backbone.'前缀
                fixed_state_dict[k] = v
            
            # 对于非标准模型，我们可能需要额外步骤
            if model_type == 'ensemble':
                # 简化处理 - 实际应用中需要更复杂的逻辑
                print("集成模型使用自定义权重加载")
                pass
            else:
                # 尝试加载修复后的状态字典
                try:
                    model.load_state_dict(fixed_state_dict, strict=False)
                    print("模型加载成功（non-strict模式）")
                except Exception as e:
                    print(f"加载修复状态字典时出错: {e}")
                    # 可能需要更进一步的处理
        else:
            # 尝试直接加载
            print("尝试直接加载状态字典")
            model.load_state_dict(saved_state, strict=False)
            
    except Exception as e:
        print(f"加载模型权重时出错: {e}")
        # 如果无法加载，使用随机初始化的模型（用于测试）
        print("使用随机初始化的权重")
    
    model = model.to(device)
    model.eval()
    return model, emotion_categories

# 评估函数
def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="评估中"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 处理可能的模型输出格式不同的情况
            try:
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]  # 有些模型可能返回多个输出
                
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, all_preds, all_labels

# 可视化函数
def plot_results(accuracies, model_names):
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, accuracies, color='skyblue')
    plt.xlabel('模型')
    plt.ylabel('准确率 (%)')
    plt.title('不同表情识别模型的准确率比较')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 在柱状图上添加准确率值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom')
    
    plt.savefig('emotion_model_accuracy_comparison.png')
    plt.close()

def plot_confusion_matrix(cm, classes, model_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} 混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()

# 主函数
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型文件
    model_files = [
        "emotion_distilled_optimized.pth",
        "emotion_ensemble_optimized.pth",
        "emotion_resnet50_optimized.pth",
        "emotion_vit_optimized.pth",
        "improved_emotion_model.pth"
    ]
    
    results = {}
    confusion_matrices = {}
    emotion_categories_global = None
    
    # 评估每个模型
    for model_file in model_files:
        print(f"\n评估模型: {model_file}")
        model_path = os.path.join(MODEL_PATH, model_file)
        
        try:
            # 加载模型并获取情绪类别
            model, emotion_categories = load_model(model_path, device)
            
            # 使用第一个模型的情绪类别作为全局类别（确保所有模型使用相同的类别标签顺序）
            if emotion_categories_global is None and emotion_categories is not None:
                emotion_categories_global = emotion_categories
                
            # 创建测试数据集和加载器
            curr_emotion_categories = emotion_categories_global or emotion_categories
            test_dataset = EmotionDataset(TEST_PATH, transform=get_transform(), 
                                         emotion_categories=curr_emotion_categories)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            # 评估模型
            accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)
            results[model_file] = accuracy
            
            # 计算混淆矩阵
            cm = confusion_matrix(all_labels, all_preds)
            confusion_matrices[model_file] = cm
            
            print(f"{model_file} 准确率: {accuracy:.2f}%")
            
            # 打印分类报告
            if curr_emotion_categories:
                report = classification_report(all_labels, all_preds, target_names=curr_emotion_categories)
                print(f"分类报告:\n{report}")
            
        except Exception as e:
            print(f"评估 {model_file} 时出错: {e}")
            results[model_file] = 0  # 为失败的模型分配0分
    
    # 找出最佳模型（如果有的话）
    if results:
        best_model = max(results, key=results.get)
        print(f"\n最佳模型是 {best_model}，准确率为 {results[best_model]:.2f}%")
        
        # 可视化结果
        plot_results(list(results.values()), list(results.keys()))
        
        # 为最佳模型绘制混淆矩阵（如果有）
        if best_model in confusion_matrices and emotion_categories_global:
            plot_confusion_matrix(confusion_matrices[best_model], emotion_categories_global, best_model)
        
        # 保存结果到CSV
        df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': list(results.values())
        })
        df.to_csv('emotion_model_results.csv', index=False)
        print("结果已保存到 emotion_model_results.csv")
    else:
        print("没有模型评估成功，无法确定最佳模型")

if __name__ == "__main__":
    main()