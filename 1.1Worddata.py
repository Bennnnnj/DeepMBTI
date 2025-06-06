import pandas as pd
import numpy as np
import re
import nltk
from collections import defaultdict, Counter
import os
import json
import warnings
from datetime import datetime
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import textstat
warnings.filterwarnings('ignore')

class EnglishMBTIDatasetProcessor:
    def __init__(self):
        # 英文停用词
        try:
            from nltk.corpus import stopwords
            self.english_stopwords = set(stopwords.words('english'))
        except:
            self.english_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
        
        # MBTI英文关键词库
        self.mbti_keywords = {
            'E': {
                'social', 'party', 'people', 'outgoing', 'energetic', 'talkative', 
                'expressive', 'collaborative', 'gregarious', 'vivacious', 'animated', 
                'enthusiastic', 'spontaneous', 'assertive', 'confident', 'leadership', 
                'presentation', 'performance', 'audience', 'community', 'society', 
                'external', 'together', 'group', 'team', 'networking', 'meeting', 
                'conversation', 'chat', 'interaction', 'communicate', 'express'
            },
            'I': {
                'alone', 'quiet', 'solitude', 'private', 'reserved', 'withdrawn', 
                'shy', 'independent', 'individual', 'personal', 'calm', 'peaceful', 
                'reflection', 'thinking', 'contemplation', 'reading', 'writing', 
                'meditation', 'introspection', 'solitary', 'reclusive', 'secluded', 
                'intimate', 'inner', 'internal', 'thoughtful', 'pensive', 'reflective', 
                'deliberate', 'careful', 'selective', 'depth', 'concentrate', 'focus', 
                'privacy', 'sanctuary', 'retreat', 'recharge', 'observe', 'listen'
            },
            'S': {
                'practical', 'realistic', 'concrete', 'facts', 'details', 'specific', 
                'experience', 'present', 'actual', 'real', 'tangible', 'sequential', 
                'traditional', 'conventional', 'established', 'proven', 'tested', 
                'observable', 'measurable', 'precise', 'literal', 'factual', 'empirical', 
                'hands-on', 'technical', 'methodical', 'systematic', 'routine', 
                'procedure', 'protocol', 'evidence', 'data', 'statistics', 'measurement', 
                'calculation', 'implementation', 'application', 'utility', 'function'
            },
            'N': {
                'future', 'possibility', 'potential', 'imagination', 'creative', 
                'innovative', 'abstract', 'concept', 'theory', 'idea', 'vision', 
                'intuition', 'insight', 'pattern', 'meaning', 'symbolic', 'metaphor', 
                'inspiration', 'dream', 'fantasy', 'original', 'conceptual', 
                'theoretical', 'hypothetical', 'speculative', 'inventive', 'artistic', 
                'imaginative', 'visionary', 'trend', 'forecast', 'prediction', 
                'anticipate', 'foresee', 'breakthrough', 'revolution', 'transformation'
            },
            'T': {
                'logical', 'rational', 'analysis', 'objective', 'reason', 'logic', 
                'critical', 'systematic', 'analytical', 'problem-solving', 'efficiency', 
                'effectiveness', 'truth', 'fact', 'evidence', 'principle', 'criteria', 
                'standard', 'judge', 'evaluate', 'methodical', 'strategic', 'tactical', 
                'algorithmic', 'mathematical', 'scientific', 'research', 'investigation', 
                'diagnosis', 'solution', 'optimization', 'performance', 'metrics', 
                'benchmark', 'assessment', 'examination', 'reasoning', 'calculation'
            },
            'F': {
                'feeling', 'emotion', 'heart', 'empathy', 'compassion', 'care', 
                'love', 'harmony', 'relationship', 'personal', 'values', 'moral', 
                'ethical', 'human', 'understanding', 'sympathy', 'kindness', 'warmth', 
                'support', 'help', 'concern', 'emotional', 'sensitive', 'nurturing', 
                'supportive', 'considerate', 'diplomatic', 'cooperative', 'collaborative', 
                'consensus', 'welfare', 'wellbeing', 'happiness', 'satisfaction', 
                'fulfillment', 'motivation', 'inspiration', 'encouragement', 'appreciation'
            },
            'J': {
                'organized', 'planned', 'schedule', 'structure', 'order', 'control', 
                'decided', 'settled', 'closure', 'deadline', 'goal', 'target', 
                'systematic', 'methodical', 'routine', 'regular', 'predictable', 
                'determined', 'decisive', 'complete', 'finish', 'accomplish', 
                'disciplined', 'punctual', 'orderly', 'neat', 'tidy', 'arranged', 
                'timeline', 'milestone', 'achievement', 'completion', 'resolution', 
                'commitment', 'responsibility', 'accountability', 'reliability'
            },
            'P': {
                'flexible', 'adaptable', 'spontaneous', 'open-ended', 'casual', 
                'relaxed', 'informal', 'freedom', 'options', 'alternatives', 'explore', 
                'discover', 'experiment', 'playful', 'curious', 'improvise', 'adjust', 
                'change', 'variety', 'diverse', 'versatile', 'dynamic', 'fluid', 
                'elastic', 'malleable', 'opportunistic', 'serendipity', 'adventure', 
                'exploration', 'experimentation', 'innovation', 'creativity', 'openness'
            }
        }
        
        # 数据增强配置
        self.augmentation_methods = {
            'synonym_replacement': True,
            'random_insertion': True,
            'random_swap': True,
            'random_deletion': True,
            'sentence_reordering': True,
            'paraphrasing': True
        }
    
    def comprehensive_data_analysis(self, df):
        """全面的数据分析"""
        print("=== 执行全面数据分析 ===")
        
        analysis_report = {
            'basic_statistics': {},
            'distribution_analysis': {},
            'quality_assessment': {},
            'text_characteristics': {},
            'mbti_patterns': {},
            'recommendations': []
        }
        
        # 1. 基础统计
        analysis_report['basic_statistics'] = {
            'total_samples': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict()
        }
        
        # 2. MBTI分布分析
        if 'type' in df.columns:
            type_counts = df['type'].value_counts()
            analysis_report['distribution_analysis'] = {
                'mbti_distribution': type_counts.to_dict(),
                'most_common_type': type_counts.index[0],
                'least_common_type': type_counts.index[-1],
                'imbalance_ratio': type_counts.max() / type_counts.min(),
                'dimension_distributions': self.analyze_dimension_distributions(df)
            }
        
        # 3. 文本特征分析
        if 'posts' in df.columns:
            text_analysis = self.analyze_text_characteristics(df['posts'])
            analysis_report['text_characteristics'] = text_analysis
        
        # 4. 数据质量评估
        quality_assessment = self.assess_data_quality(df)
        analysis_report['quality_assessment'] = quality_assessment
        
        return analysis_report
    
    def analyze_dimension_distributions(self, df):
        """分析MBTI维度分布"""
        dimensions = {
            'E_vs_I': {'E': 0, 'I': 0},
            'S_vs_N': {'S': 0, 'N': 0},
            'T_vs_F': {'T': 0, 'F': 0},
            'J_vs_P': {'J': 0, 'P': 0}
        }
        
        for mbti_type in df['type']:
            if len(mbti_type) == 4:
                dimensions['E_vs_I'][mbti_type[0]] += 1
                dimensions['S_vs_N'][mbti_type[1]] += 1
                dimensions['T_vs_F'][mbti_type[2]] += 1
                dimensions['J_vs_P'][mbti_type[3]] += 1
        
        # 计算平衡度
        for dim_name, counts in dimensions.items():
            total = sum(counts.values())
            if total > 0:
                balance_score = min(counts.values()) / max(counts.values()) if max(counts.values()) > 0 else 0
                dimensions[dim_name]['balance_score'] = balance_score
                dimensions[dim_name]['imbalance_ratio'] = max(counts.values()) / min(counts.values()) if min(counts.values()) > 0 else float('inf')
        
        return dimensions
    
    def analyze_text_characteristics(self, texts):
        """分析文本特征"""
        text_features = {
            'total_texts': len(texts),
            'empty_texts': texts.isnull().sum(),
            'avg_length': texts.str.len().mean(),
            'median_length': texts.str.len().median(),
            'std_length': texts.str.len().std(),
            'min_length': texts.str.len().min(),
            'max_length': texts.str.len().max(),
            'length_distribution': {
                'very_short': (texts.str.len() < 100).sum(),
                'short': ((texts.str.len() >= 100) & (texts.str.len() < 500)).sum(),
                'medium': ((texts.str.len() >= 500) & (texts.str.len() < 2000)).sum(),
                'long': ((texts.str.len() >= 2000) & (texts.str.len() < 5000)).sum(),
                'very_long': (texts.str.len() >= 5000).sum()
            }
        }
        
        # 计算平均可读性（采样）
        readability_scores = []
        for text in texts.dropna().head(1000):
            if text and len(text) > 10:
                try:
                    readability = textstat.flesch_reading_ease(text)
                    readability_scores.append(readability)
                except:
                    continue
        
        if readability_scores:
            text_features['avg_readability'] = np.mean(readability_scores)
        
        return text_features
    
    def assess_data_quality(self, df):
        """评估数据质量"""
        quality_report = {}
        
        # 1. 完整性
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        completeness = 1 - (missing_cells / total_cells)
        quality_report['completeness'] = completeness
        
        # 2. 一致性
        consistency_issues = 0
        if 'type' in df.columns:
            valid_types = df['type'].str.match(r'^[EINS][INST][TF][JP]$')
            consistency_issues += (~valid_types).sum()
        
        consistency = 1 - (consistency_issues / len(df))
        quality_report['consistency'] = consistency
        
        # 3. 唯一性
        if 'posts' in df.columns:
            duplicates = df['posts'].duplicated().sum()
            uniqueness = 1 - (duplicates / len(df))
            quality_report['uniqueness'] = uniqueness
        
        # 4. 有效性
        validity_issues = 0
        if 'posts' in df.columns:
            too_short = (df['posts'].str.len() < 10).sum()
            too_long = (df['posts'].str.len() > 50000).sum()
            validity_issues += too_short + too_long
        
        validity = 1 - (validity_issues / len(df))
        quality_report['validity'] = validity
        
        # 总体质量分数
        quality_report['overall_quality'] = np.mean([
            completeness, consistency, uniqueness, validity
        ])
        
        return quality_report
    
    def clean_and_standardize_data(self, df):
        """清洗和标准化数据"""
        print("执行数据清洗和标准化...")
        
        cleaned_df = df.copy()
        cleaning_log = []
        
        # 1. 处理缺失值
        missing_posts = cleaned_df['posts'].isnull().sum()
        cleaned_df = cleaned_df.dropna(subset=['posts'])
        cleaning_log.append(f"删除posts缺失的行: {missing_posts} 行")
        
        missing_types = cleaned_df['type'].isnull().sum()
        cleaned_df = cleaned_df.dropna(subset=['type'])
        cleaning_log.append(f"删除type缺失的行: {missing_types} 行")
        
        # 2. 标准化MBTI类型格式
        cleaned_df['type'] = cleaned_df['type'].str.upper()
        valid_types = ['INTJ', 'INTP', 'ENTJ', 'ENTP', 'INFJ', 'INFP', 'ENFJ', 'ENFP',
                      'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ', 'ISTP', 'ISFP', 'ESTP', 'ESFP']
        
        invalid_count = (~cleaned_df['type'].isin(valid_types)).sum()
        cleaned_df = cleaned_df[cleaned_df['type'].isin(valid_types)]
        cleaning_log.append(f"删除无效MBTI类型: {invalid_count} 行")
        
        # 3. 清洗文本内容
        original_count = len(cleaned_df)
        min_length = 20
        cleaned_df = cleaned_df[cleaned_df['posts'].str.len() >= min_length]
        cleaning_log.append(f"删除过短文本(<{min_length}字符): {original_count - len(cleaned_df)} 行")
        
        max_length = 50000
        long_text_count = (cleaned_df['posts'].str.len() > max_length).sum()
        cleaned_df = cleaned_df[cleaned_df['posts'].str.len() <= max_length]
        cleaning_log.append(f"删除过长文本(>{max_length}字符): {long_text_count} 行")
        
        # 4. 标准化文本
        cleaned_df['normalized_posts'] = cleaned_df['posts'].apply(self.normalize_english_text)
        
        # 5. 去除重复数据
        duplicate_count = cleaned_df['normalized_posts'].duplicated().sum()
        cleaned_df = cleaned_df.drop_duplicates(subset=['normalized_posts'])
        cleaning_log.append(f"删除重复文本: {duplicate_count} 行")
        
        # 6. 重置索引
        cleaned_df = cleaned_df.reset_index(drop=True)
        
        print(f"数据清洗完成:")
        for log in cleaning_log:
            print(f"  - {log}")
        print(f"最终数据量: {len(cleaned_df)} 行")
        
        return cleaned_df, cleaning_log
    
    def normalize_english_text(self, text):
        """标准化英文文本"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # 1. 转换为小写
        text = text.lower()
        
        # 2. 处理缩写
        contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "i'm": "i am", "you're": "you are", "it's": "it is",
            "he's": "he is", "she's": "she is", "we're": "we are",
            "they're": "they are", "i've": "i have", "you've": "you have",
            "we've": "we have", "they've": "they have", "i'll": "i will",
            "you'll": "you will", "he'll": "he will", "she'll": "she will",
            "we'll": "we will", "they'll": "they will", "i'd": "i would",
            "you'd": "you would", "he'd": "he would", "she'd": "she would",
            "we'd": "we would", "they'd": "they would"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # 3. 移除URL
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' [URL] ', text)
        
        # 4. 清理特殊字符，保留基本标点
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # 5. 处理重复标点
        text = re.sub(r'([.!?]){2,}', r'\1', text)
        
        # 6. 清理多余空格
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def local_data_augmentation(self, df, augmentation_factor=1.5):
        """本地数据增强"""
        print("执行本地数据增强...")
        
        augmented_data = []
        original_data = df.to_dict('records')
        
        for record in original_data:
            # 保留原始数据
            augmented_data.append(record)
            
            text = record['normalized_posts']
            
            # 1. 同义词替换
            if self.augmentation_methods['synonym_replacement']:
                aug_text = self.synonym_replacement(text, n=3)
                if aug_text != text:
                    aug_record = record.copy()
                    aug_record['posts'] = aug_text
                    aug_record['normalized_posts'] = aug_text
                    aug_record['augmentation_method'] = 'synonym_replacement'
                    augmented_data.append(aug_record)
            
            # 2. 随机插入
            if self.augmentation_methods['random_insertion']:
                aug_text = self.random_insertion(text, n=2)
                if aug_text != text:
                    aug_record = record.copy()
                    aug_record['posts'] = aug_text
                    aug_record['normalized_posts'] = aug_text
                    aug_record['augmentation_method'] = 'random_insertion'
                    augmented_data.append(aug_record)
            
            # 3. 随机交换
            if self.augmentation_methods['random_swap']:
                aug_text = self.random_swap(text, n=2)
                if aug_text != text:
                    aug_record = record.copy()
                    aug_record['posts'] = aug_text
                    aug_record['normalized_posts'] = aug_text
                    aug_record['augmentation_method'] = 'random_swap'
                    augmented_data.append(aug_record)
            
            # 4. 句子重组
            if self.augmentation_methods['sentence_reordering']:
                aug_text = self.sentence_reordering(text)
                if aug_text != text:
                    aug_record = record.copy()
                    aug_record['posts'] = aug_text
                    aug_record['normalized_posts'] = aug_text
                    aug_record['augmentation_method'] = 'sentence_reordering'
                    augmented_data.append(aug_record)
        
        # 限制增强后的数据量
        target_size = int(len(df) * augmentation_factor)
        augmented_df = pd.DataFrame(augmented_data)
        
        if len(augmented_df) > target_size:
            # 保留所有原始数据，随机采样增强数据
            original_df = augmented_df[augmented_df.get('augmentation_method', 'original') == 'original']
            augmented_only = augmented_df[augmented_df.get('augmentation_method', 'original') != 'original']
            
            if len(augmented_only) > 0:
                sample_size = min(target_size - len(original_df), len(augmented_only))
                sampled_aug = augmented_only.sample(n=sample_size, random_state=42)
                final_df = pd.concat([original_df, sampled_aug], ignore_index=True)
            else:
                final_df = original_df
        else:
            final_df = augmented_df
        
        print(f"本地数据增强完成: {len(df)} -> {len(final_df)} 样本")
        return final_df.reset_index(drop=True)
    
    def synonym_replacement(self, text, n=1):
        """同义词替换"""
        words = text.split()
        new_words = words.copy()
        
        # 简单的同义词映射
        synonym_dict = {
            'good': ['great', 'excellent', 'wonderful', 'fantastic'],
            'bad': ['terrible', 'awful', 'horrible', 'poor'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'mini', 'compact'],
            'happy': ['joyful', 'cheerful', 'delighted', 'pleased'],
            'sad': ['unhappy', 'depressed', 'upset', 'disappointed'],
            'think': ['believe', 'consider', 'suppose', 'assume'],
            'like': ['enjoy', 'appreciate', 'prefer', 'favor'],
            'want': ['desire', 'wish', 'need', 'require'],
            'see': ['observe', 'notice', 'view', 'watch']
        }
        
        replacements_made = 0
        for i, word in enumerate(words):
            if replacements_made >= n:
                break
            word_lower = word.lower()
            if word_lower in synonym_dict and random.random() < 0.3:
                new_words[i] = random.choice(synonym_dict[word_lower])
                replacements_made += 1
        
        return ' '.join(new_words)
    
    def random_insertion(self, text, n=1):
        """随机插入"""
        words = text.split()
        
        # 插入的词汇
        insert_words = ['really', 'very', 'quite', 'pretty', 'somewhat', 'rather']
        
        for _ in range(n):
            if len(words) > 0:
                random_idx = random.randint(0, len(words))
                random_word = random.choice(insert_words)
                words.insert(random_idx, random_word)
        
        return ' '.join(words)
    
    def random_swap(self, text, n=1):
        """随机交换"""
        words = text.split()
        new_words = words.copy()
        
        for _ in range(n):
            if len(new_words) >= 2:
                idx1 = random.randint(0, len(new_words) - 1)
                idx2 = random.randint(0, len(new_words) - 1)
                new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def sentence_reordering(self, text):
        """句子重组"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 2:
            random.shuffle(sentences)
            return '. '.join(sentences) + '.'
        return text
    
    def balance_dataset(self, df):
        """数据平衡处理"""
        print("执行数据平衡处理...")
        
        if 'type' not in df.columns:
            return df
        
        type_counts = df['type'].value_counts()
        target_count = int(type_counts.median())
        
        balanced_dfs = []
        
        for mbti_type in type_counts.index:
            type_data = df[df['type'] == mbti_type]
            current_count = len(type_data)
            
            if current_count > target_count:
                # 下采样
                sampled_data = resample(type_data, n_samples=target_count, random_state=42)
                balanced_dfs.append(sampled_data)
            elif current_count < target_count:
                # 上采样
                upsampled_data = resample(type_data, n_samples=target_count, random_state=42, replace=True)
                balanced_dfs.append(upsampled_data)
            else:
                balanced_dfs.append(type_data)
        
        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"数据平衡完成: {len(df)} -> {len(balanced_df)} 样本")
        return balanced_df
    
    def extract_english_keywords(self, df):
        """提取英文关键词"""
        print("执行英文关键词提取...")
        
        if 'type' not in df.columns or 'normalized_posts' not in df.columns:
            return {}
        
        results = {}
        
        for dimension in ['E', 'S', 'T', 'J']:
            # 确定正负样本
            dimension_pos = {'E': 0, 'S': 1, 'T': 2, 'J': 3}[dimension]
            positive_samples = df[df['type'].str[dimension_pos] == dimension]
            negative_samples = df[df['type'].str[dimension_pos] != dimension]
            
            if len(positive_samples) == 0 or len(negative_samples) == 0:
                continue
            
            try:
                # TF-IDF分析
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=list(self.english_stopwords),
                    min_df=2,
                    max_df=0.8,
                    ngram_range=(1, 2)
                )
                
                pos_texts = positive_samples['normalized_posts'].tolist()
                neg_texts = negative_samples['normalized_posts'].tolist()
                all_texts = pos_texts + neg_texts
                
                tfidf_matrix = vectorizer.fit_transform(all_texts)
                feature_names = vectorizer.get_feature_names_out()
                
                # 计算正负样本的平均TF-IDF
                pos_mean = tfidf_matrix[:len(pos_texts)].mean(axis=0).A1
                neg_mean = tfidf_matrix[len(pos_texts):].mean(axis=0).A1
                
                # 计算差异分数
                diff_scores = pos_mean - neg_mean
                
                # 获取top关键词
                top_indices = np.argsort(np.abs(diff_scores))[-50:]
                dimension_keywords = []
                
                for idx in top_indices:
                    if abs(diff_scores[idx]) > 0.001:
                        keyword = feature_names[idx]
                        dimension_keywords.append({
                            'keyword': keyword,
                            'score': diff_scores[idx],
                            'positive_tfidf': pos_mean[idx],
                            'negative_tfidf': neg_mean[idx],
                            'abs_score': abs(diff_scores[idx]),
                            'is_predefined': keyword in self.mbti_keywords.get(dimension, set()) or 
                                           keyword in self.mbti_keywords.get({'E':'I', 'S':'N', 'T':'F', 'J':'P'}.get(dimension, ''), set())
                        })
                
                results[dimension] = sorted(dimension_keywords, key=lambda x: x['abs_score'], reverse=True)
                
            except Exception as e:
                print(f"关键词提取失败 - {dimension}: {e}")
                continue
        
        return results
    
    def create_data_splits(self, df, test_size=0.2, val_size=0.1):
        """创建数据分割"""
        print("创建数据分割...")
        
        if 'type' not in df.columns:
            return {'train': df, 'val': None, 'test': None}
        
        # 分层分割
        train_val, test = train_test_split(
            df, 
            test_size=test_size, 
            stratify=df['type'], 
            random_state=42
        )
        
        val_ratio = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=train_val['type'],
            random_state=42
        )
        
        splits = {
            'train': train.reset_index(drop=True),
            'val': val.reset_index(drop=True),
            'test': test.reset_index(drop=True)
        }
        
        print(f"数据分割完成:")
        print(f"  训练集: {len(splits['train'])} 样本")
        print(f"  验证集: {len(splits['val'])} 样本")
        print(f"  测试集: {len(splits['test'])} 样本")
        
        return splits
    
    def generate_comprehensive_report(self, original_df, final_df, keywords, output_dir):
        """生成综合报告"""
        print("生成综合分析报告...")
        
        report = {
            'processing_summary': {
                'original_samples': len(original_df),
                'final_samples': len(final_df),
                'processing_timestamp': datetime.now().isoformat(),
                'processing_type': 'English_Only_Enhancement'
            },
            'original_analysis': self.comprehensive_data_analysis(original_df),
            'final_analysis': self.comprehensive_data_analysis(final_df),
            'keyword_analysis': {
                'total_dimensions': len(keywords),
                'keywords_per_dimension': {dim: len(kws) for dim, kws in keywords.items()},
                'top_keywords_summary': {}
            },
            'quality_improvements': {},
            'recommendations': []
        }
        
        # 计算质量改进
        orig_quality = report['original_analysis']['quality_assessment']
        final_quality = report['final_analysis']['quality_assessment']
        
        for metric in orig_quality:
            if metric in final_quality:
                improvement = final_quality[metric] - orig_quality[metric]
                report['quality_improvements'][metric] = {
                    'original': orig_quality[metric],
                    'final': final_quality[metric],
                    'improvement': improvement,
                    'improvement_percentage': improvement / orig_quality[metric] * 100 if orig_quality[metric] > 0 else 0
                }
        
        # 关键词摘要
        for dimension, keyword_list in keywords.items():
            if keyword_list:
                top_5 = keyword_list[:5]
                report['keyword_analysis']['top_keywords_summary'][dimension] = [
                    {'keyword': kw['keyword'], 'score': kw['score']} for kw in top_5
                ]
        
        # 生成建议
        report['recommendations'] = [
            "数据已清洗并标准化，可直接用于模型训练",
            "建议使用balanced数据集进行训练以获得最佳性能",
            "关键词可用于特征工程和模型解释",
            "考虑使用交叉验证评估模型性能",
            "可根据keywords构建MBTI特征词典"
        ]
        
        # 保存报告
        report_file = os.path.join(output_dir, 'english_mbti_analysis_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"分析报告已保存: {report_file}")
        return report
    
    def process_english_enhancement(self, file_path):
        """完整的英文数据增强处理流程"""
        print("=== 启动英文MBTI数据集增强处理 ===")
        
        # 1. 加载原始数据
        original_df = pd.read_csv(file_path)
        print(f"原始数据加载完成: {len(original_df)} 样本")
        
        # 2. 数据清洗和标准化
        cleaned_df, cleaning_log = self.clean_and_standardize_data(original_df)
        
        # 3. 本地数据增强
        augmented_df = self.local_data_augmentation(cleaned_df)
        
        # 4. 数据平衡处理
        balanced_df = self.balance_dataset(augmented_df)
        
        # 5. 英文关键词提取
        keywords = self.extract_english_keywords(balanced_df)
        
        # 6. 创建数据分割
        data_splits = self.create_data_splits(balanced_df)
        
        # 7. 保存所有数据集
        output_dir = os.path.dirname(file_path)
        
        datasets_to_save = {
            'cleaned': cleaned_df,
            'augmented': augmented_df,
            'balanced': balanced_df,
            'train': data_splits['train'],
            'val': data_splits['val'],
            'test': data_splits['test']
        }
        
        saved_files = {}
        for name, dataset in datasets_to_save.items():
            if dataset is not None:
                filename = f'enhanced_english_mbti_{name}.csv'
                filepath = os.path.join(output_dir, filename)
                dataset.to_csv(filepath, index=False)
                saved_files[name] = filepath
                print(f"已保存 {name} 数据集: {filepath}")
        
        # 8. 保存关键词
        keywords_file = os.path.join(output_dir, 'english_keywords.json')
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump(keywords, f, indent=2, ensure_ascii=False, default=str)
        print(f"关键词已保存: {keywords_file}")
        
        # 9. 生成综合报告
        report = self.generate_comprehensive_report(original_df, balanced_df, keywords, output_dir)
        
        print("=== 英文数据增强处理完成 ===")
        
        return {
            'datasets': datasets_to_save,
            'saved_files': saved_files,
            'keywords': keywords,
            'data_splits': data_splits,
            'cleaning_log': cleaning_log,
            'analysis_report': report
        }

def main():
    file_path = r"C:\Users\lnasl\Desktop\DeepMBTI\data\Text\mbti_1.csv"
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # 下载必要的NLTK资源（如果需要）
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
    except:
        print("NLTK资源下载失败，使用内置资源")
    
    processor = EnglishMBTIDatasetProcessor()
    results = processor.process_english_enhancement(file_path)
    
    print("\n=== 英文数据集增强处理总结 ===")
    print("✅ 数据清洗和标准化")
    print("✅ 本地数据增强（同义词、重组等）")
    print("✅ MBTI类型平衡处理")
    print("✅ 英文关键词提取")
    print("✅ 数据分割（训练/验证/测试）")
    print("✅ 质量评估和改进分析")
    print("✅ 综合处理报告")
    
    print(f"\n生成的数据集文件:")
    for name, filepath in results['saved_files'].items():
        if name == 'balanced':
            print(f"  🎯 {name}: {filepath} (推荐用于训练)")
        else:
            print(f"  - {name}: {filepath}")
    
    print(f"\n关键词文件: {os.path.join(os.path.dirname(file_path), 'english_keywords.json')}")
    print(f"分析报告: {os.path.join(os.path.dirname(file_path), 'english_mbti_analysis_report.json')}")
    
    # 显示一些关键统计
    if 'keywords' in results:
        print(f"\n关键词提取结果:")
        for dim, keywords in results['keywords'].items():
            if keywords:
                top_3 = keywords[:3]
                print(f"  {dim}: {', '.join([kw['keyword'] for kw in top_3])}")

if __name__ == "__main__":
    main()