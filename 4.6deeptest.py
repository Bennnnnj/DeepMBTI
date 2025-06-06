import pandas as pd
import numpy as np
import json
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入transformers
try:
    from transformers import AutoTokenizer, AutoModel
    from transformers import DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
    HAS_TRANSFORMERS = True
except ImportError:
    print("⚠️ transformers库未安装，请安装: pip install transformers")
    HAS_TRANSFORMERS = False

# PyTorch数据集类
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float32) if self.labels is not None else None
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if labels is not None:
            result['labels'] = labels
            
        return result

# 深度学习模型定义
class MBTITransformerModel(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased', num_labels=4):
        super().__init__()
        
        if 'distilbert' in model_name:
            self.bert = DistilBertModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size
        elif 'roberta' in model_name:
            self.bert = RobertaModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size
        else:
            self.bert = AutoModel.from_pretrained(model_name)
            hidden_size = self.bert.config.hidden_size
        
        # 多层分类头
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        output = self.dropout(pooled_output)
        return self.classifier(output)

class MBTIDeepModelTester:
    def __init__(self, models_dir, data_dir):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.dimensions = ['E_I', 'S_N', 'T_F', 'J_P']
        self.dim_names = {
            'E_I': 'Extraversion vs Introversion',
            'S_N': 'Sensing vs Intuition', 
            'T_F': 'Thinking vs Feeling',
            'J_P': 'Judging vs Perceiving'
        }
        self.models = {}
        self.tokenizers = {}
        self.device = self.setup_device()
        
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🎮 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device('cpu')
            print("💻 使用CPU")
        return device
    
    def preprocess_text(self, text):
        """文本预处理（与训练时保持一致）"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        
        # 高效的文本清理
        replacements = {
            r"won't": "will not", r"can't": "cannot", r"n't": " not",
            r"i'm": "i am", r"you're": "you are", r"it's": "it is"
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        
        # 清理URL和特殊字符
        text = re.sub(r'http[s]?://\S+', ' ', text)
        text = re.sub(r'[^\w\s\.\!\?\,]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def discover_models(self):
        """自动发现可用的深度学习模型"""
        print("🔍 搜索可用的深度学习模型...")
        
        available_models = []
        
        # 检查deep_learning目录中的所有子目录
        if os.path.exists(self.models_dir):
            for item in os.listdir(self.models_dir):
                model_path = os.path.join(self.models_dir, item)
                if os.path.isdir(model_path):
                    # 检查是否有模型文件
                    model_file = os.path.join(model_path, 'model.pt')
                    config_file = os.path.join(model_path, 'config.json')
                    tokenizer_file = os.path.join(model_path, 'tokenizer.json')
                    
                    if os.path.exists(model_file):
                        available_models.append({
                            'name': item,
                            'path': model_path,
                            'model_file': model_file,
                            'has_config': os.path.exists(config_file),
                            'has_tokenizer': os.path.exists(tokenizer_file)
                        })
                        print(f"  ✅ 发现模型: {item}")
        
        if not available_models:
            print("❌ 未发现任何深度学习模型")
            return []
        
        print(f"🎉 共发现 {len(available_models)} 个深度学习模型")
        return available_models
    
    def load_deep_models(self):
        """加载深度学习模型"""
        print("\n📂 加载深度学习模型...")
        
        if not HAS_TRANSFORMERS:
            print("❌ transformers库未安装，无法加载深度学习模型")
            return False
        
        # 发现可用模型
        available_models = self.discover_models()
        if not available_models:
            return False
        
        self.loaded_models = {}
        
        for model_info in available_models:
            model_name = model_info['name']
            model_path = model_info['path']
            model_file = model_info['model_file']
            
            print(f"\n🤖 加载模型: {model_name}")
            
            try:
                # 尝试从模型名称推断tokenizer类型
                if 'distilbert' in model_name.lower():
                    base_model = 'distilbert-base-uncased'
                    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
                elif 'roberta' in model_name.lower():
                    base_model = 'roberta-base'
                    tokenizer = RobertaTokenizer.from_pretrained(model_path)
                else:
                    # 默认使用DistilBERT
                    base_model = 'distilbert-base-uncased'
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                    except:
                        print(f"  ⚠️ 无法从{model_path}加载tokenizer，尝试使用预训练tokenizer")
                        tokenizer = DistilBertTokenizer.from_pretrained(base_model)
                
                print(f"  ✅ Tokenizer加载成功")
                
                # 创建模型实例
                model = MBTITransformerModel(base_model, num_labels=4)
                
                # 加载模型权重
                state_dict = torch.load(model_file, map_location=self.device)
                
                # 如果模型是DataParallel包装的，需要去掉'module.'前缀
                if any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = key.replace('module.', '')
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
                
                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()
                
                self.loaded_models[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'base_model': base_model
                }
                
                print(f"  ✅ 模型加载成功")
                
            except Exception as e:
                print(f"  ❌ 模型加载失败: {e}")
                continue
        
        if not self.loaded_models:
            print("❌ 没有成功加载任何模型")
            return False
        
        print(f"\n🎉 成功加载 {len(self.loaded_models)} 个深度学习模型")
        return True
    
    def load_test_data(self):
        """加载测试数据"""
        print("\n📊 加载测试数据...")
        
        test_file = os.path.join(self.data_dir, 'enhanced_english_mbti_test.csv')
        if not os.path.exists(test_file):
            print(f"❌ 测试文件未找到: {test_file}")
            return False
        
        self.test_df = pd.read_csv(test_file)
        print(f"✅ 测试数据加载成功: {len(self.test_df):,} 样本")
        
        # 创建MBTI维度标签
        self.test_df['E_I'] = (self.test_df['type'].str[0] == 'E').astype(int)
        self.test_df['S_N'] = (self.test_df['type'].str[1] == 'S').astype(int)
        self.test_df['T_F'] = (self.test_df['type'].str[2] == 'T').astype(int)
        self.test_df['J_P'] = (self.test_df['type'].str[3] == 'J').astype(int)
        
        # 预处理文本
        self.test_df['processed_text'] = self.test_df['normalized_posts'].fillna('').apply(self.preprocess_text)
        
        return True
    
    def test_model(self, model_name, model_data):
        """测试单个深度学习模型"""
        print(f"\n🎯 测试模型: {model_name}")
        
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # 准备测试数据
        test_labels = self.test_df[self.dimensions].values
        test_dataset = MBTIDataset(
            self.test_df['processed_text'].tolist(),
            test_labels,
            tokenizer,
            max_length=256
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # 使用较小的批量避免内存问题
            shuffle=False,
            num_workers=2 if os.name != 'nt' else 0,  # Windows兼容性
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 预测
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        print("  🔄 进行预测...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 50 == 0:
                    print(f"    处理批次 {batch_idx}/{len(test_loader)}")
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 前向传播
                try:
                    if torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = model(input_ids, attention_mask)
                    else:
                        outputs = model(input_ids, attention_mask)
                    
                    probabilities = torch.sigmoid(outputs)
                    predictions = (probabilities > 0.5).float()
                    
                    all_predictions.append(predictions.cpu())
                    all_probabilities.append(probabilities.cpu())
                    all_labels.append(labels.cpu())
                    
                except Exception as e:
                    print(f"    ⚠️ 批次 {batch_idx} 处理失败: {e}")
                    continue
        
        if not all_predictions:
            print(f"  ❌ 模型 {model_name} 预测失败")
            return None
        
        # 合并结果
        predictions = torch.cat(all_predictions, dim=0).numpy()
        probabilities = torch.cat(all_probabilities, dim=0).numpy()
        true_labels = torch.cat(all_labels, dim=0).numpy()
        
        print("  ✅ 预测完成，计算指标...")
        
        # 计算每个维度的性能
        model_results = {}
        for i, dim_name in enumerate(self.dimensions):
            y_true = true_labels[:, i]
            y_pred = predictions[:, i]
            y_prob = probabilities[:, i]
            
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            auc = roc_auc_score(y_true, y_prob)
            cm = confusion_matrix(y_true, y_pred)
            
            model_results[dim_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'confusion_matrix': cm,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            print(f"    {dim_name}: F1={f1:.3f}, AUC={auc:.3f}")
        
        return model_results
    
    def test_all_models(self):
        """测试所有深度学习模型"""
        print("\n🚀 开始测试所有深度学习模型...")
        
        self.all_results = {}
        
        for model_name, model_data in self.loaded_models.items():
            try:
                results = self.test_model(model_name, model_data)
                if results:
                    self.all_results[model_name] = results
                    print(f"✅ {model_name} 测试完成")
                else:
                    print(f"❌ {model_name} 测试失败")
            except Exception as e:
                print(f"❌ {model_name} 测试异常: {e}")
                continue
            finally:
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return self.all_results
    
    def display_results(self):
        """显示测试结果"""
        print("\n" + "="*100)
        print("🧠 深度学习MBTI模型测试结果")
        print("="*100)
        
        if not self.all_results:
            print("❌ 没有可显示的结果")
            return
        
        # 为每个模型显示结果
        for model_name, model_results in self.all_results.items():
            print(f"\n🤖 模型: {model_name}")
            print("="*60)
            
            # 性能概览表
            print(f"{'维度':<15} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUC':<8}")
            print("-" * 65)
            
            all_f1 = []
            all_auc = []
            
            for dim in self.dimensions:
                results = model_results[dim]
                all_f1.append(results['f1'])
                all_auc.append(results['auc'])
                
                print(f"{dim:<15} {results['accuracy']:<8.3f} {results['precision']:<8.3f} "
                      f"{results['recall']:<8.3f} {results['f1']:<8.3f} {results['auc']:<8.3f}")
            
            print("-" * 65)
            print(f"{'平均':<15} {'':<8} {'':<8} {'':<8} {np.mean(all_f1):<8.3f} {np.mean(all_auc):<8.3f}")
            
            # 详细分析
            for dim in self.dimensions:
                print(f"\n📊 {dim} ({self.dim_names[dim]}) 详细分析:")
                results = model_results[dim]
                
                # 目标名称
                target_names = ['0', '1']
                if dim == 'E_I':
                    target_names = ['Introversion', 'Extraversion']
                elif dim == 'S_N':
                    target_names = ['Sensing', 'Intuition']
                elif dim == 'T_F':
                    target_names = ['Thinking', 'Feeling']
                elif dim == 'J_P':
                    target_names = ['Judging', 'Perceiving']
                
                # 分类报告
                print(classification_report(results['y_true'], results['y_pred'], 
                                          target_names=target_names, digits=3))
                
                # 混淆矩阵
                print("混淆矩阵:")
                cm = results['confusion_matrix']
                print(f"实际\\预测  {target_names[0]:<12} {target_names[1]:<12}")
                print(f"{target_names[0]:<12} {cm[0][0]:<12} {cm[0][1]:<12}")
                print(f"{target_names[1]:<12} {cm[1][0]:<12} {cm[1][1]:<12}")
        
        # 模型比较
        if len(self.all_results) > 1:
            print(f"\n📈 模型性能比较:")
            print("="*60)
            print(f"{'模型':<20} {'平均F1':<10} {'平均AUC':<10}")
            print("-" * 40)
            
            for model_name, model_results in self.all_results.items():
                avg_f1 = np.mean([model_results[dim]['f1'] for dim in self.dimensions])
                avg_auc = np.mean([model_results[dim]['auc'] for dim in self.dimensions])
                print(f"{model_name:<20} {avg_f1:<10.3f} {avg_auc:<10.3f}")
    
    def predict_sample_text(self, text, model_name=None):
        """用指定模型预测样本文本"""
        if not self.loaded_models:
            print("❌ 没有可用的模型")
            return None
        
        # 如果没有指定模型，使用第一个
        if model_name is None:
            model_name = list(self.loaded_models.keys())[0]
        
        if model_name not in self.loaded_models:
            print(f"❌ 模型 {model_name} 不存在")
            print(f"可用模型: {list(self.loaded_models.keys())}")
            return None
        
        print(f"\n🔮 使用模型 {model_name} 预测文本...")
        
        model_data = self.loaded_models[model_name]
        model = model_data['model']
        tokenizer = model_data['tokenizer']
        
        # 预处理文本
        processed_text = self.preprocess_text(text)
        
        # 创建数据集
        dataset = MBTIDataset([processed_text], None, tokenizer, max_length=256)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 预测
        model.eval()
        with torch.no_grad():
            batch = next(iter(dataloader))
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            outputs = model(input_ids, attention_mask)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
            predictions = (probabilities > 0.5).astype(int)
        
        # 构建MBTI类型
        mbti_type = ""
        mbti_type += "E" if predictions[0] == 1 else "I"
        mbti_type += "S" if predictions[1] == 1 else "N"
        mbti_type += "T" if predictions[2] == 1 else "F"
        mbti_type += "J" if predictions[3] == 1 else "P"
        
        print(f"原始文本: {text}")
        print(f"预测MBTI类型: {mbti_type}")
        print(f"各维度概率:")
        for i, dim in enumerate(self.dimensions):
            print(f"  {dim}: {probabilities[i]:.3f}")
        
        return mbti_type, probabilities
    
    def run_complete_test(self):
        """运行完整的深度学习模型测试"""
        start_time = datetime.now()
        
        print("🚀 开始深度学习MBTI模型测试...")
        
        # 1. 加载模型
        if not self.load_deep_models():
            print("❌ 模型加载失败，测试终止")
            return False
        
        # 2. 加载测试数据
        if not self.load_test_data():
            print("❌ 测试数据加载失败，测试终止")
            return False
        
        # 3. 测试所有模型
        self.test_all_models()
        
        # 4. 显示结果
        self.display_results()
        
        end_time = datetime.now()
        test_time = end_time - start_time
        
        print(f"\n🎉 深度学习模型测试完成！")
        print(f"⏱️ 总测试时间: {test_time}")
        
        return True

def main():
    # 设置路径
    models_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\TrainedModel\text\new\deep_learning"
    data_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\data\Text"
    
    # 检查路径是否存在
    if not os.path.exists(models_dir):
        print(f"❌ 深度学习模型目录不存在: {models_dir}")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建测试器
    tester = MBTIDeepModelTester(models_dir, data_dir)
    
    # 运行完整测试
    success = tester.run_complete_test()
    
    if success:
        print(f"\n💡 可以使用以下代码进行新文本预测:")
        print(f"sample_text = 'Your text here...'")
        print(f"mbti_type, probs = tester.predict_sample_text(sample_text)")
        
        # 示例预测
        if hasattr(tester, 'loaded_models') and tester.loaded_models:
            print(f"\n🔮 示例预测:")
            sample_text = "I love meeting new people and going to parties. I always speak up in meetings and enjoy being the center of attention."
            try:
                tester.predict_sample_text(sample_text)
            except Exception as e:
                print(f"示例预测失败: {e}")

if __name__ == "__main__":
    main()