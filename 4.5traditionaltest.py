import pandas as pd
import numpy as np
import json
import os
import re
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MBTIModelTester:
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
        
    def load_models_and_resources(self):
        """加载所有训练好的模型和资源"""
        print("📂 加载训练好的模型和资源...")
        
        # 1. 加载TF-IDF向量化器
        tfidf_path = os.path.join(os.path.dirname(self.models_dir), 'tfidf_vectorizer.pkl')
        if os.path.exists(tfidf_path):
            self.tfidf_vectorizer = joblib.load(tfidf_path)
            print(f"✅ TF-IDF向量化器加载成功")
        else:
            print(f"❌ TF-IDF向量化器未找到: {tfidf_path}")
            return False
        
        # 2. 加载关键词数据
        keywords_path = os.path.join(self.data_dir, 'english_keywords.json')
        if os.path.exists(keywords_path):
            with open(keywords_path, 'r', encoding='utf-8') as f:
                self.keywords = json.load(f)
            print(f"✅ 关键词数据加载成功")
        else:
            print(f"❌ 关键词文件未找到: {keywords_path}")
            return False
        
        # 3. 加载四个MBTI维度模型
        for dim in self.dimensions:
            model_path = os.path.join(self.models_dir, f'mbti_{dim}_ensemble.pkl')
            if os.path.exists(model_path):
                self.models[dim] = joblib.load(model_path)
                print(f"✅ {dim} 模型加载成功")
            else:
                print(f"❌ {dim} 模型未找到: {model_path}")
                return False
        
        print(f"🎉 所有模型和资源加载完成！")
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
    
    def create_test_features(self):
        """为测试数据创建特征（与训练时保持一致）"""
        print("\n🧠 为测试数据创建特征...")
        
        # 1. TF-IDF特征
        print("📝 创建TF-IDF特征...")
        test_texts = self.test_df['processed_text'].tolist()
        self.X_test_tfidf = self.tfidf_vectorizer.transform(test_texts)
        print(f"✅ TF-IDF特征: {self.X_test_tfidf.shape[1]:,} 维")
        
        # 2. 关键词特征
        print("🔑 创建关键词特征...")
        self.X_test_keywords = self.create_keyword_features(self.test_df, 'processed_text')
        print(f"✅ 关键词特征: {self.X_test_keywords.shape[1]} 维")
        
        # 3. 统计特征
        print("📊 创建统计特征...")
        self.X_test_stats = self.create_advanced_stats_features(self.test_df, 'processed_text')
        print(f"✅ 统计特征: {self.X_test_stats.shape[1]} 维")
        
        # 4. 组合特征
        from scipy import sparse
        self.X_test = sparse.hstack([
            self.X_test_tfidf, 
            self.X_test_keywords, 
            self.X_test_stats
        ])
        
        print(f"🎯 总特征维度: {self.X_test.shape[1]:,}")
        return True
    
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
    
    def test_all_models(self):
        """测试所有模型"""
        print("\n🎯 开始测试所有模型...")
        
        self.test_results = {}
        
        for dim in self.dimensions:
            print(f"\n📊 测试 {dim} ({self.dim_names[dim]}) 模型...")
            
            # 获取真实标签
            y_true = self.test_df[dim].values
            
            # 获取模型预测
            model = self.models[dim]
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1]
            
            # 计算各种指标
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
            auc = roc_auc_score(y_true, y_prob)
            
            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred)
            
            self.test_results[dim] = {
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
            
            print(f"  ✅ 准确率: {accuracy:.3f}")
            print(f"  ✅ 精确率: {precision:.3f}")
            print(f"  ✅ 召回率: {recall:.3f}")
            print(f"  ✅ F1分数: {f1:.3f}")
            print(f"  ✅ AUC: {auc:.3f}")
        
        return self.test_results
    
    def display_detailed_results(self):
        """显示详细的测试结果"""
        print("\n" + "="*80)
        print("📊 MBTI模型测试结果详细报告")
        print("="*80)
        
        # 总体性能概览
        print(f"\n🎯 总体性能概览:")
        print(f"{'维度':<15} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1分数':<8} {'AUC':<8}")
        print("-" * 65)
        
        all_accuracies = []
        all_f1_scores = []
        all_aucs = []
        
        for dim in self.dimensions:
            results = self.test_results[dim]
            all_accuracies.append(results['accuracy'])
            all_f1_scores.append(results['f1'])
            all_aucs.append(results['auc'])
            
            print(f"{dim:<15} {results['accuracy']:<8.3f} {results['precision']:<8.3f} "
                  f"{results['recall']:<8.3f} {results['f1']:<8.3f} {results['auc']:<8.3f}")
        
        print("-" * 65)
        print(f"{'平均':<15} {np.mean(all_accuracies):<8.3f} {'':<8} {'':<8} "
              f"{np.mean(all_f1_scores):<8.3f} {np.mean(all_aucs):<8.3f}")
        
        # 各维度详细分析
        for dim in self.dimensions:
            print(f"\n🔍 {dim} ({self.dim_names[dim]}) 详细分析:")
            results = self.test_results[dim]
            
            # 分类报告
            print(f"\n分类报告:")
            target_names = ['0', '1']
            if dim == 'E_I':
                target_names = ['Introversion', 'Extraversion']
            elif dim == 'S_N':
                target_names = ['Sensing', 'Intuition']
            elif dim == 'T_F':
                target_names = ['Thinking', 'Feeling']
            elif dim == 'J_P':
                target_names = ['Judging', 'Perceiving']
            
            print(classification_report(results['y_true'], results['y_pred'], 
                                      target_names=target_names, digits=3))
            
            # 混淆矩阵
            print(f"混淆矩阵:")
            cm = results['confusion_matrix']
            print(f"实际\\预测  {target_names[0]:<12} {target_names[1]:<12}")
            print(f"{target_names[0]:<12} {cm[0][0]:<12} {cm[0][1]:<12}")
            print(f"{target_names[1]:<12} {cm[1][0]:<12} {cm[1][1]:<12}")
            
            # 样本分布
            total_samples = len(results['y_true'])
            positive_samples = sum(results['y_true'])
            negative_samples = total_samples - positive_samples
            
            print(f"\n样本分布:")
            print(f"  {target_names[0]}: {negative_samples} 样本 ({negative_samples/total_samples*100:.1f}%)")
            print(f"  {target_names[1]}: {positive_samples} 样本 ({positive_samples/total_samples*100:.1f}%)")
    
    def predict_mbti_type(self, sample_texts):
        """预测新文本的MBTI类型"""
        print(f"\n🔮 预测新文本的MBTI类型...")
        
        if isinstance(sample_texts, str):
            sample_texts = [sample_texts]
        
        # 预处理文本
        processed_texts = [self.preprocess_text(text) for text in sample_texts]
        
        # 创建临时DataFrame
        temp_df = pd.DataFrame({'processed_text': processed_texts})
        
        # 创建特征
        # TF-IDF
        X_tfidf = self.tfidf_vectorizer.transform(processed_texts)
        
        # 关键词特征
        X_keywords = self.create_keyword_features(temp_df, 'processed_text')
        
        # 统计特征
        X_stats = self.create_advanced_stats_features(temp_df, 'processed_text')
        
        # 组合特征
        from scipy import sparse
        X_combined = sparse.hstack([X_tfidf, X_keywords, X_stats])
        
        # 预测每个维度
        predictions = {}
        probabilities = {}
        
        for dim in self.dimensions:
            model = self.models[dim]
            pred = model.predict(X_combined)
            prob = model.predict_proba(X_combined)[:, 1]
            
            predictions[dim] = pred
            probabilities[dim] = prob
        
        # 组合成MBTI类型
        mbti_types = []
        for i in range(len(sample_texts)):
            mbti_type = ""
            mbti_type += "E" if predictions['E_I'][i] == 1 else "I"
            mbti_type += "S" if predictions['S_N'][i] == 1 else "N"
            mbti_type += "T" if predictions['T_F'][i] == 1 else "F"
            mbti_type += "J" if predictions['J_P'][i] == 1 else "P"
            mbti_types.append(mbti_type)
        
        # 显示结果
        for i, (text, mbti_type) in enumerate(zip(sample_texts, mbti_types)):
            print(f"\n样本 {i+1}:")
            print(f"文本: {text[:100]}{'...' if len(text) > 100 else ''}")
            print(f"预测MBTI类型: {mbti_type}")
            print(f"各维度概率:")
            for dim in self.dimensions:
                prob = probabilities[dim][i]
                print(f"  {dim}: {prob:.3f}")
        
        return mbti_types, probabilities
    
    def run_complete_test(self):
        """运行完整的测试流程"""
        start_time = datetime.now()
        
        print("🚀 开始MBTI模型完整测试...")
        
        # 1. 加载模型和资源
        if not self.load_models_and_resources():
            print("❌ 模型加载失败，测试终止")
            return False
        
        # 2. 加载测试数据
        if not self.load_test_data():
            print("❌ 测试数据加载失败，测试终止")
            return False
        
        # 3. 创建测试特征
        if not self.create_test_features():
            print("❌ 特征创建失败，测试终止")
            return False
        
        # 4. 测试所有模型
        self.test_all_models()
        
        # 5. 显示详细结果
        self.display_detailed_results()
        
        end_time = datetime.now()
        test_time = end_time - start_time
        
        print(f"\n🎉 测试完成！")
        print(f"⏱️ 总测试时间: {test_time}")
        
        return True

def main():
    # 设置路径
    models_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\TrainedModel\text\new\traditional"
    data_dir = r"C:\Users\lnasl\Desktop\DeepMBTI\data\Text"
    
    # 检查路径是否存在
    if not os.path.exists(models_dir):
        print(f"❌ 模型目录不存在: {models_dir}")
        return
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return
    
    # 创建测试器
    tester = MBTIModelTester(models_dir, data_dir)
    
    # 运行完整测试
    success = tester.run_complete_test()
    
    if success:
        print(f"\n💡 可以使用以下代码进行新文本预测:")
        print(f"sample_text = 'Your text here...'")
        print(f"mbti_types, probs = tester.predict_mbti_type(sample_text)")

if __name__ == "__main__":
    main()