import pandas as pd
import numpy as np
import json
import os
import re
from datetime import datetime
import warnings
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import requests  # 添加requests导入
warnings.filterwarnings('ignore')

# 机器学习库 - 启用所有CPU核心
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# 深度学习库 - GPU优化
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, get_scheduler
from transformers import DistilBertTokenizer, DistilBertModel, RobertaTokenizer, RobertaModel
import accelerate
from torch.cuda.amp import autocast, GradScaler

# 设置CPU核心数
CPU_CORES = 20  # 您的线程数
os.environ["OMP_NUM_THREADS"] = str(CPU_CORES)
os.environ["MKL_NUM_THREADS"] = str(CPU_CORES)

class HighPerformanceMBTITrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.device = self.setup_gpu()
        self.cpu_cores = CPU_CORES
        
        print(f"🚀 高性能MBTI训练系统初始化")
        print(f"💻 CPU核心: {self.cpu_cores}")
        print(f"🎮 GPU设备: {self.device}")
        print(f"🔥 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        
        # 模型存储
        self.models = {}
        self.deep_models = {}
        self.tokenizers = {}
        
        # MBTI维度
        self.dimensions = ['E_I', 'S_N', 'T_F', 'J_P']
        self.dim_names = {
            'E_I': 'Extraversion vs Introversion',
            'S_N': 'Sensing vs Intuition', 
            'T_F': 'Thinking vs Feeling',
            'J_P': 'Judging vs Perceiving'
        }
    
    def setup_gpu(self):
        """设置GPU优化"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🎮 检测到GPU: {torch.cuda.get_device_name(0)}")
            
            # 优化GPU设置
            torch.backends.cudnn.benchmark = True  # 优化cuDNN性能
            torch.backends.cudnn.deterministic = False  # 允许非确定性操作以提高性能
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            return device
        else:
            print("⚠️ 未检测到GPU，使用CPU")
            return torch.device('cpu')
    
    def load_data_optimized(self):
        """优化的数据加载"""
        print("\n📊 加载数据（内存优化）...")
        
        # 并行加载文件
        files = {
            'train': 'enhanced_english_mbti_train.csv',
            'val': 'enhanced_english_mbti_val.csv', 
            'test': 'enhanced_english_mbti_test.csv',
            'keywords': 'english_keywords.json'
        }
        
        def load_file(file_info):
            name, filename = file_info
            filepath = os.path.join(self.data_dir, filename)
            
            if filename.endswith('.csv'):
                # 优化pandas读取
                df = pd.read_csv(filepath, engine='c')  # 使用C引擎加速
                return name, df
            elif filename.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return name, json.load(f)
        
        # 并行加载
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load_file, files.items()))
        
        # 分配数据
        for name, data in results:
            if name == 'train':
                self.train_df = data
            elif name == 'val':
                self.val_df = data
            elif name == 'test':
                self.test_df = data
            elif name == 'keywords':
                self.keywords = data
        
        print(f"✅ 训练集: {len(self.train_df):,} 样本")
        print(f"✅ 验证集: {len(self.val_df):,} 样本") 
        print(f"✅ 测试集: {len(self.test_df):,} 样本")
        
        # 创建MBTI维度标签
        self.create_labels()
        
        # 预处理文本到内存
        self.preprocess_all_texts()
        
        return self.train_df, self.val_df, self.test_df
    
    def create_labels(self):
        """创建所有维度标签"""
        for df in [self.train_df, self.val_df, self.test_df]:
            df['E_I'] = (df['type'].str[0] == 'E').astype(int)
            df['S_N'] = (df['type'].str[1] == 'S').astype(int)
            df['T_F'] = (df['type'].str[2] == 'T').astype(int)
            df['J_P'] = (df['type'].str[3] == 'J').astype(int)
    
    def preprocess_batch(self, texts):
        """预处理一批文本"""
        return [self.preprocess_text(text) for text in texts]
    
    def preprocess_all_texts(self):
        """预处理所有文本到内存"""
        print("🔄 预处理所有文本...")
        
        # 并行预处理 - 使用ThreadPoolExecutor避免pickle问题
        batch_size = 1000
        for df_name, df in [('train', self.train_df), ('val', self.val_df), ('test', self.test_df)]:
            texts = df['normalized_posts'].fillna('').tolist()
            processed_texts = []
            
            # 分批并行处理 - 改用ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=min(self.cpu_cores, 8)) as executor:
                batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
                results = list(executor.map(self.preprocess_batch, batches))
                
                for batch_result in results:
                    processed_texts.extend(batch_result)
            
            df['processed_text'] = processed_texts
            print(f"✅ {df_name} 文本预处理完成")
    
    def preprocess_text(self, text):
        """文本预处理"""
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
    
    def create_optimized_features(self):
        """创建优化的特征"""
        print("\n🧠 特征工程（多核并行）...")
        
        # 1. 高性能TF-IDF特征
        print("📝 TF-IDF特征提取...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # 增加特征数量
            stop_words='english',
            ngram_range=(1, 3),  # 增加到3-gram
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,  # 使用sublinear TF scaling
            norm='l2'
        )
        
        # 并行TF-IDF
        train_texts = self.train_df['processed_text'].tolist()
        val_texts = self.val_df['processed_text'].tolist()
        test_texts = self.test_df['processed_text'].tolist()
        
        self.X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_texts)
        self.X_val_tfidf = self.tfidf_vectorizer.transform(val_texts)
        self.X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)
        
        print(f"✅ TF-IDF特征: {self.X_train_tfidf.shape[1]:,} 维")
        
        # 2. 并行关键词特征
        print("🔑 关键词特征提取...")
        with ThreadPoolExecutor(max_workers=min(self.cpu_cores, 8)) as executor:
            keyword_futures = [
                executor.submit(self.create_keyword_features, df, 'processed_text')
                for df in [self.train_df, self.val_df, self.test_df]
            ]
            
            keyword_results = [future.result() for future in keyword_futures]
        
        self.X_train_keywords, self.X_val_keywords, self.X_test_keywords = keyword_results
        print(f"✅ 关键词特征: {self.X_train_keywords.shape[1]} 维")
        
        # 3. 并行统计特征
        print("📊 统计特征提取...")
        with ThreadPoolExecutor(max_workers=min(self.cpu_cores, 8)) as executor:
            stats_futures = [
                executor.submit(self.create_advanced_stats_features, df, 'processed_text')
                for df in [self.train_df, self.val_df, self.test_df]
            ]
            
            stats_results = [future.result() for future in stats_futures]
        
        self.X_train_stats, self.X_val_stats, self.X_test_stats = stats_results
        print(f"✅ 统计特征: {self.X_train_stats.shape[1]} 维")
        
        # 4. 组合特征
        from scipy import sparse
        
        self.X_train = sparse.hstack([
            self.X_train_tfidf, 
            self.X_train_keywords, 
            self.X_train_stats
        ])
        self.X_val = sparse.hstack([
            self.X_val_tfidf, 
            self.X_val_keywords, 
            self.X_val_stats
        ])
        self.X_test = sparse.hstack([
            self.X_test_tfidf, 
            self.X_test_keywords, 
            self.X_test_stats
        ])
        
        print(f"🎯 总特征维度: {self.X_train.shape[1]:,}")
        
        return self.X_train, self.X_val, self.X_test
    
    def create_keyword_features(self, df, text_column):
        """创建关键词特征"""
        features = []
        
        for _, row in df.iterrows():
            text = str(row[text_column]).lower()
            feature_vector = []
            
            for dim in ['E', 'S', 'T', 'J']:
                if dim in self.keywords:
                    keywords_data = self.keywords[dim][:30]  # 使用更多关键词
                    
                    pos_score = 0
                    neg_score = 0
                    weighted_score = 0
                    
                    for kw_info in keywords_data:
                        keyword = kw_info['keyword']
                        score = kw_info.get('score', 0)
                        importance = kw_info.get('abs_score', abs(score))
                        
                        count = text.count(keyword)
                        if count > 0:
                            if score > 0:
                                pos_score += count * importance
                            else:
                                neg_score += count * importance
                            weighted_score += count * score
                    
                    feature_vector.extend([pos_score, neg_score, weighted_score, pos_score - neg_score])
                else:
                    feature_vector.extend([0, 0, 0, 0])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def create_advanced_stats_features(self, df, text_column):
        """创建高级统计特征"""
        features = []
        
        for _, row in df.iterrows():
            text = str(row[text_column])
            words = text.split()
            sentences = text.split('.')
            
            # 基础统计
            text_len = len(text)
            word_count = len(words)
            sent_count = len([s for s in sentences if s.strip()])
            
            # 高级特征
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            vocab_diversity = len(set(words)) / len(words) if words else 0
            
            # 标点和语调
            exclamation_ratio = text.count('!') / text_len if text_len > 0 else 0
            question_ratio = text.count('?') / text_len if text_len > 0 else 0
            caps_ratio = sum(1 for c in text if c.isupper()) / text_len if text_len > 0 else 0
            
            # 人称代词分析
            first_person = (text.count(' i ') + text.count('me') + text.count('my')) / word_count if word_count > 0 else 0
            second_person = (text.count('you') + text.count('your')) / word_count if word_count > 0 else 0
            third_person = (text.count('he') + text.count('she') + text.count('they')) / word_count if word_count > 0 else 0
            
            # 情感词汇
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'enjoy']
            negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'sad', 'angry']
            
            pos_sentiment = sum(text.count(word) for word in positive_words) / word_count if word_count > 0 else 0
            neg_sentiment = sum(text.count(word) for word in negative_words) / word_count if word_count > 0 else 0
            
            # 复杂度指标
            avg_sent_len = word_count / sent_count if sent_count > 0 else 0
            long_words_ratio = len([w for w in words if len(w) > 6]) / word_count if word_count > 0 else 0
            
            stats = [
                text_len, word_count, sent_count, avg_word_len, vocab_diversity,
                exclamation_ratio, question_ratio, caps_ratio,
                first_person, second_person, third_person,
                pos_sentiment, neg_sentiment, avg_sent_len, long_words_ratio
            ]
            
            features.append(stats)
        
        return np.array(features)
    
    def train_model_dimension(self, args):
        """训练单个模型的单个维度"""
        model_name, model, dim_name = args
        
        y_train = self.train_df[dim_name].values
        y_val = self.val_df[dim_name].values
        
        # 训练模型
        if model_name == 'SVM':
            # SVM使用较少的特征以提高速度
            from sklearn.feature_selection import SelectKBest, f_classif
            selector = SelectKBest(f_classif, k=min(5000, self.X_train.shape[1]))
            X_train_selected = selector.fit_transform(self.X_train, y_train)
            X_val_selected = selector.transform(self.X_val)
            
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_val_selected)
            y_prob = model.predict_proba(X_val_selected)[:, 1]
        else:
            model.fit(self.X_train, y_train)
            y_pred = model.predict(self.X_val)
            y_prob = model.predict_proba(self.X_val)[:, 1]
        
        # 计算指标
        accuracy = accuracy_score(y_val, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        auc = roc_auc_score(y_val, y_prob)
        
        return dim_name, model_name, {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'model': model
        }
    
    def train_traditional_models_parallel(self):
        """并行训练传统机器学习模型"""
        print("\n🤖 并行训练传统模型...")
        
        # 高性能模型配置
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,  # 增加树的数量
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=self.cpu_cores,  # 使用所有CPU核心
                random_state=42
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=self.cpu_cores,
                random_state=42
            ),
            'LogisticRegression': LogisticRegression(
                random_state=42,
                max_iter=2000,
                n_jobs=self.cpu_cores,
                solver='saga'  # 支持并行的求解器
            ),
            'SVM': SVC(
                random_state=42,
                probability=True,
                kernel='rbf',
                cache_size=2000  # 增加缓存
            )
        }
        
        self.traditional_results = {}
        
        # 由于pickle问题，改为串行训练但每个模型内部并行
        print(f"🚀 启动传统模型训练（每个模型内部并行）...")
        
        for dim_name in self.dimensions:
            self.traditional_results[dim_name] = {}
            
            for model_name, model in models.items():
                y_train = self.train_df[dim_name].values
                y_val = self.val_df[dim_name].values
                
                # 训练模型
                if model_name == 'SVM':
                    # SVM使用较少的特征以提高速度
                    from sklearn.feature_selection import SelectKBest, f_classif
                    selector = SelectKBest(f_classif, k=min(5000, self.X_train.shape[1]))
                    X_train_selected = selector.fit_transform(self.X_train, y_train)
                    X_val_selected = selector.transform(self.X_val)
                    
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_val_selected)
                    y_prob = model.predict_proba(X_val_selected)[:, 1]
                else:
                    model.fit(self.X_train, y_train)
                    y_pred = model.predict(self.X_val)
                    y_prob = model.predict_proba(self.X_val)[:, 1]
                
                # 计算指标
                accuracy = accuracy_score(y_val, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
                auc = roc_auc_score(y_val, y_prob)
                
                self.traditional_results[dim_name][model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'model': model
                }
                
                print(f"✅ {dim_name}-{model_name}: F1={f1:.3f}, AUC={auc:.3f}")
        
        return self.traditional_results
    
    def create_gpu_ensemble_models(self):
        """创建GPU加速的集成模型"""
        print("\n🔥 创建GPU集成模型...")
        
        self.ensemble_results = {}
        
        for dim_name in self.dimensions:
            print(f"🎯 处理 {dim_name} 维度...")
            
            # 选择最佳的3个模型
            dim_results = self.traditional_results[dim_name]
            sorted_models = sorted(dim_results.items(), key=lambda x: x[1]['f1'], reverse=True)
            top_models = sorted_models[:3]
            
            # 创建加权投票集成
            estimators = [(name, result['model']) for name, result in top_models]
            weights = [result['f1'] for _, result in top_models]  # 基于F1分数的权重
            
            ensemble = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=weights,
                n_jobs=self.cpu_cores
            )
            
            # 训练集成模型
            y_train = self.train_df[dim_name].values
            y_val = self.val_df[dim_name].values
            
            ensemble.fit(self.X_train, y_train)
            y_pred = ensemble.predict(self.X_val)
            y_prob = ensemble.predict_proba(self.X_val)[:, 1]
            
            # 评估
            accuracy = accuracy_score(y_val, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
            auc = roc_auc_score(y_val, y_prob)
            
            self.ensemble_results[dim_name] = {
                'model': ensemble,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'base_models': [name for name, _ in top_models],
                'weights': weights
            }
            
            print(f"  🎉 集成模型 F1: {f1:.3f}, AUC: {auc:.3f}")
            print(f"  📊 使用模型: {', '.join([name for name, _ in top_models])}")
        
        return self.ensemble_results

# PyTorch数据集类
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }

# GPU优化的深度学习模型
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

class HighPerformanceMBTITrainer(HighPerformanceMBTITrainer):
    def check_network_and_models(self):
        """检查网络连接和模型可用性"""
        print("\n🔍 检查网络连接和模型可用性...")
        
        try:
            import requests
            response = requests.get("https://huggingface.co", timeout=10)
            if response.status_code == 200:
                print("✅ 网络连接正常")
                return True
        except Exception as e:
            print(f"❌ 网络连接异常: {e}")
            print("⚠️ 将跳过深度学习模型训练，仅使用传统机器学习模型")
            return False
    
    def train_gpu_deep_models(self):
        """训练GPU深度学习模型"""
        print("\n🚀 训练GPU深度学习模型...")
        
        # 检查网络连接
        if not self.check_network_and_models():
            print("⚠️ 由于网络问题，跳过深度学习模型训练")
            self.deep_results = {}
            return self.deep_results
        
        # 模型配置
        model_configs = [
            ('distilbert-base-uncased', 64, 3e-5),  # DistilBERT - 快速
            ('roberta-base', 32, 2e-5),             # RoBERTa - 准确
        ]
        
        self.deep_results = {}
        
        for model_name, batch_size, learning_rate in model_configs:
            print(f"\n🤖 训练 {model_name}...")
            
            try:
                # 创建tokenizer
                if 'distilbert' in model_name:
                    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
                elif 'roberta' in model_name:
                    tokenizer = RobertaTokenizer.from_pretrained(model_name)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                print(f"✅ 成功加载 {model_name} tokenizer")
                
            except Exception as e:
                print(f"❌ 无法加载 {model_name}: {e}")
                print(f"⚠️ 跳过 {model_name} 模型训练")
                continue
            
            try:
                # 准备数据
                train_labels = self.train_df[self.dimensions].values
                val_labels = self.val_df[self.dimensions].values
                
                train_dataset = MBTIDataset(
                    self.train_df['processed_text'].tolist(),
                    train_labels,
                    tokenizer,
                    max_length=256  # 优化序列长度以提高速度
                )
                
                val_dataset = MBTIDataset(
                    self.val_df['processed_text'].tolist(),
                    val_labels,
                    tokenizer,
                    max_length=256
                )
                
                # 创建数据加载器 - 优化批量大小以充分利用5090
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=4,  # 减少worker数量避免潜在问题
                    pin_memory=True if torch.cuda.is_available() else False,
                    persistent_workers=False  # 避免一些Windows兼容性问题
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size * 2,  # 验证时可以用更大批量
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True if torch.cuda.is_available() else False,
                    persistent_workers=False
                )
                
                # 创建模型
                model = MBTITransformerModel(model_name, num_labels=4)
                
                # 使用DataParallel如果有多GPU（5090是单GPU但很强大）
                if torch.cuda.device_count() > 1:
                    model = DataParallel(model)
                
                model = model.to(self.device)
                
                # 优化器和调度器
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=learning_rate,
                    weight_decay=0.01
                )
                
                num_epochs = 3  # 减少epochs避免长时间训练
                num_training_steps = len(train_loader) * num_epochs
                scheduler = get_scheduler(
                    "linear",
                    optimizer=optimizer,
                    num_warmup_steps=int(0.1 * num_training_steps),
                    num_training_steps=num_training_steps
                )
                
                # 混合精度训练 - 充分利用5090的Tensor Cores
                scaler = GradScaler() if torch.cuda.is_available() else None
                
                # 训练循环
                model.train()
                for epoch in range(num_epochs):
                    print(f"  📅 Epoch {epoch+1}/{num_epochs}")
                    
                    total_loss = 0
                    for batch_idx, batch in enumerate(train_loader):
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        optimizer.zero_grad()
                        
                        # 混合精度前向传播
                        if scaler and torch.cuda.is_available():
                            with autocast():
                                outputs = model(input_ids, attention_mask)
                                loss = F.binary_cross_entropy_with_logits(outputs, labels)
                            
                            # 混合精度反向传播
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            outputs = model(input_ids, attention_mask)
                            loss = F.binary_cross_entropy_with_logits(outputs, labels)
                            loss.backward()
                            optimizer.step()
                        
                        scheduler.step()
                        total_loss += loss.item()
                        
                        if batch_idx % 100 == 0:
                            print(f"    📈 Batch {batch_idx}, Loss: {loss.item():.4f}")
                    
                    avg_loss = total_loss / len(train_loader)
                    print(f"  📊 Epoch {epoch+1} 平均损失: {avg_loss:.4f}")
                
                # 验证
                model.eval()
                val_predictions = []
                val_probabilities = []
                val_true_labels = []
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        if scaler and torch.cuda.is_available():
                            with autocast():
                                outputs = model(input_ids, attention_mask)
                        else:
                            outputs = model(input_ids, attention_mask)
                        
                        probabilities = torch.sigmoid(outputs)
                        predictions = (probabilities > 0.5).float()
                        
                        val_predictions.append(predictions.cpu())
                        val_probabilities.append(probabilities.cpu())
                        val_true_labels.append(labels.cpu())
                
                # 合并预测结果
                val_predictions = torch.cat(val_predictions, dim=0).numpy()
                val_probabilities = torch.cat(val_probabilities, dim=0).numpy()
                val_true_labels = torch.cat(val_true_labels, dim=0).numpy()
                
                # 计算每个维度的性能
                model_results = {}
                for i, dim_name in enumerate(self.dimensions):
                    y_true = val_true_labels[:, i]
                    y_pred = val_predictions[:, i]
                    y_prob = val_probabilities[:, i]
                    
                    accuracy = accuracy_score(y_true, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                    auc = roc_auc_score(y_true, y_prob)
                    
                    model_results[dim_name] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'auc': auc
                    }
                    
                    print(f"    ✅ {dim_name}: F1={f1:.3f}, AUC={auc:.3f}")
                
                self.deep_results[model_name] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'results': model_results
                }
                
                print(f"✅ {model_name} 训练完成")
                
            except Exception as e:
                print(f"❌ {model_name} 训练失败: {e}")
                continue
            
            finally:
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        
        if not self.deep_results:
            print("⚠️ 所有深度学习模型都无法训练，将仅使用传统机器学习模型")
        
        return self.deep_results
    
    def evaluate_all_models_on_test(self):
        """在测试集上评估所有模型"""
        print("\n🎯 测试集最终评估...")
        
        self.final_test_results = {
            'traditional_ensemble': {},
            'deep_learning': {}
        }
        
        # 1. 评估传统集成模型
        print("📊 评估传统集成模型...")
        for dim_name in self.dimensions:
            model = self.ensemble_results[dim_name]['model']
            y_test = self.test_df[dim_name].values
            
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_prob)
            
            self.final_test_results['traditional_ensemble'][dim_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }
            
            print(f"  {dim_name}: F1={f1:.3f}, AUC={auc:.3f}")
        
        # 2. 评估深度学习模型（如果存在）
        if self.deep_results:
            print("🤖 评估深度学习模型...")
            for model_name, model_data in self.deep_results.items():
                print(f"  {model_name}:")
                
                try:
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
                        batch_size=64,
                        shuffle=False,
                        num_workers=4,
                        pin_memory=True if torch.cuda.is_available() else False,
                        persistent_workers=False
                    )
                    
                    # 预测
                    model.eval()
                    test_predictions = []
                    test_probabilities = []
                    test_true_labels = []
                    
                    with torch.no_grad():
                        for batch in test_loader:
                            input_ids = batch['input_ids'].to(self.device)
                            attention_mask = batch['attention_mask'].to(self.device)
                            labels = batch['labels'].to(self.device)
                            
                            if torch.cuda.is_available():
                                with autocast():
                                    outputs = model(input_ids, attention_mask)
                            else:
                                outputs = model(input_ids, attention_mask)
                            
                            probabilities = torch.sigmoid(outputs)
                            predictions = (probabilities > 0.5).float()
                            
                            test_predictions.append(predictions.cpu())
                            test_probabilities.append(probabilities.cpu())
                            test_true_labels.append(labels.cpu())
                    
                    # 合并结果
                    test_predictions = torch.cat(test_predictions, dim=0).numpy()
                    test_probabilities = torch.cat(test_probabilities, dim=0).numpy()
                    test_true_labels = torch.cat(test_true_labels, dim=0).numpy()
                    
                    # 计算性能
                    model_test_results = {}
                    for i, dim_name in enumerate(self.dimensions):
                        y_true = test_true_labels[:, i]
                        y_pred = test_predictions[:, i]
                        y_prob = test_probabilities[:, i]
                        
                        accuracy = accuracy_score(y_true, y_pred)
                        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
                        auc = roc_auc_score(y_true, y_prob)
                        
                        model_test_results[dim_name] = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc
                        }
                        
                        print(f"    {dim_name}: F1={f1:.3f}, AUC={auc:.3f}")
                    
                    self.final_test_results['deep_learning'][model_name] = model_test_results
                    
                except Exception as e:
                    print(f"    ❌ {model_name} 评估失败: {e}")
                    continue
        else:
            print("⚠️ 没有深度学习模型需要评估")
        
        return self.final_test_results
    
    def save_all_models_optimized(self):
        """优化的模型保存"""
        print("\n💾 保存所有训练好的模型...")
        
        # 修改保存路径为用户指定路径
        models_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\TrainedModel\text\new"
        os.makedirs(models_dir, exist_ok=True)
        
        import joblib
        
        # 1. 保存传统模型
        traditional_dir = os.path.join(models_dir, 'traditional')
        os.makedirs(traditional_dir, exist_ok=True)
        
        for dim_name in self.dimensions:
            model_file = os.path.join(traditional_dir, f'mbti_{dim_name}_ensemble.pkl')
            joblib.dump(self.ensemble_results[dim_name]['model'], model_file)
            print(f"  ✅ 传统模型 {dim_name}: {model_file}")
        
        # 2. 保存深度学习模型
        deep_dir = os.path.join(models_dir, 'deep_learning')
        os.makedirs(deep_dir, exist_ok=True)
        
        for model_name, model_data in self.deep_results.items():
            model_path = os.path.join(deep_dir, f'{model_name.replace("/", "_")}')
            os.makedirs(model_path, exist_ok=True)
            
            # 保存模型
            torch.save(model_data['model'].state_dict(), os.path.join(model_path, 'model.pt'))
            
            # 保存tokenizer
            model_data['tokenizer'].save_pretrained(model_path)
            
            print(f"  ✅ 深度模型 {model_name}: {model_path}")
        
        # 3. 保存特征提取器和配置
        joblib.dump(self.tfidf_vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
        
        config = {
            'keywords': self.keywords,
            'dimensions': self.dimensions,
            'dim_names': self.dim_names,
            'final_test_results': self.final_test_results,
            'training_config': {
                'cpu_cores': self.cpu_cores,
                'gpu_device': str(self.device),
                'training_date': datetime.now().isoformat()
            }
        }
        
        with open(os.path.join(models_dir, 'training_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"🎉 所有模型已保存到: {models_dir}")
    
    def run_high_performance_training(self):
        """运行高性能训练流程"""
        print("🚀 启动高性能MBTI模型训练...")
        print(f"💪 硬件配置: {self.cpu_cores}核心CPU + {torch.cuda.get_device_name(0)}")
        
        start_time = datetime.now()
        
        # 1. 数据加载和预处理
        self.load_data_optimized()
        
        # 2. 特征工程
        self.create_optimized_features()
        
        # 3. 并行训练传统模型
        self.train_traditional_models_parallel()
        
        # 4. 创建集成模型
        self.create_gpu_ensemble_models()
        
        # 5. 训练深度学习模型
        self.train_gpu_deep_models()
        
        # 6. 测试集评估
        self.evaluate_all_models_on_test()
        
        # 7. 保存所有模型
        self.save_all_models_optimized()
        
        end_time = datetime.now()
        training_time = end_time - start_time
        
        print(f"\n🎉 高性能训练完成！")
        print(f"⏱️ 总训练时间: {training_time}")
        print(f"🚀 速度提升: 利用{self.cpu_cores}核心并行 + RTX 5090 GPU加速")
        
        # 显示最佳性能
        print(f"\n📊 最终性能总结:")
        print(f"{'='*60}")
        
        # 传统集成模型性能
        print("🤖 传统集成模型 (测试集):")
        traditional_f1_scores = []
        for dim_name in self.dimensions:
            f1 = self.final_test_results['traditional_ensemble'][dim_name]['f1']
            auc = self.final_test_results['traditional_ensemble'][dim_name]['auc']
            traditional_f1_scores.append(f1)
            print(f"  {self.dim_names[dim_name]}: F1={f1:.3f}, AUC={auc:.3f}")
        
        traditional_avg_f1 = np.mean(traditional_f1_scores)
        print(f"  平均F1分数: {traditional_avg_f1:.3f}")
        
        # 深度学习模型性能
        if self.final_test_results['deep_learning']:
            print(f"\n🧠 深度学习模型 (测试集):")
            for model_name, results in self.final_test_results['deep_learning'].items():
                print(f"  {model_name}:")
                deep_f1_scores = []
                for dim_name in self.dimensions:
                    f1 = results[dim_name]['f1']
                    auc = results[dim_name]['auc']
                    deep_f1_scores.append(f1)
                    print(f"    {self.dim_names[dim_name]}: F1={f1:.3f}, AUC={auc:.3f}")
                
                deep_avg_f1 = np.mean(deep_f1_scores)
                print(f"    平均F1分数: {deep_avg_f1:.3f}")
        else:
            print(f"\n⚠️ 深度学习模型: 由于网络问题未能训练")
        
        return self.final_test_results

def main():
    # 检查CUDA
    if not torch.cuda.is_available():
        print("⚠️ 未检测到CUDA，将无法使用GPU加速")
        return
    
    # 数据目录
    data_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\data\Text"
    
    # 检查文件
    required_files = [
        'enhanced_english_mbti_train.csv',
        'enhanced_english_mbti_val.csv',
        'enhanced_english_mbti_test.csv'
    ]
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            print(f"❌ 缺少文件: {file}")
            return
    
    print("🎮 检测到RTX 5090，启用GPU加速训练！")
    print(f"💻 CPU核心数: {CPU_CORES}")
    
    # 创建高性能训练器
    trainer = HighPerformanceMBTITrainer(data_dir)
    
    # 运行训练
    results = trainer.run_high_performance_training()
    
    print(f"\n📁 生成的文件:")
    print(f"  🤖 传统模型: C:\\Users\\lnasl\\Desktop\\DeepMBTI\\TrainedModel\\text\\new\\traditional\\")
    print(f"  🧠 深度模型: C:\\Users\\lnasl\\Desktop\\DeepMBTI\\TrainedModel\\text\\new\\deep_learning\\")
    print(f"  ⚙️ 配置文件: C:\\Users\\lnasl\\Desktop\\DeepMBTI\\TrainedModel\\text\\new\\training_config.json")

if __name__ == "__main__":
    main()