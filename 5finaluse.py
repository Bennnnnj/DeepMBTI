import os
import torch
import numpy as np
import json
import pickle
import cv2
import random
import base64
from PIL import Image
import io
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torchvision.transforms as transforms
from transformers import MarianMTModel, MarianTokenizer
from torch.nn import functional as F
import langid
import logging
import uuid
import tempfile
import datetime
import pandas as pd
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义目录和模型路径
EMOTION_MODEL_PATH = r"C:\Users\lnasl\Desktop\DeepMBTI\code\TrainedModel\emotion\emotion_vit_optimized.pth"
TEXT_MODEL_DIR = r"C:\Users\lnasl\Desktop\DeepMBTI\code\TrainedModel\text\ml"
TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-zh-en"
QUESTIONS_DIR = "questions"
UPLOAD_FOLDER = 'uploads'
RESULTS_DIR = 'results'
STATIC_DIR = 'static'
DATA_COLLECTION_DIR = 'collected_data'  # 新增：数据收集目录

# 确保目录存在
for directory in [UPLOAD_FOLDER, RESULTS_DIR, STATIC_DIR, QUESTIONS_DIR, DATA_COLLECTION_DIR]:
    os.makedirs(directory, exist_ok=True)

# 创建Flask应用
app = Flask(__name__, static_folder=STATIC_DIR)
CORS(app)  # 启用跨域支持
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB上传限制

# 情绪类别
EMOTION_CATEGORIES = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']

# MBTI维度
MBTI_DIMENSIONS = [
    ["I", "E"],  # 内向/外向
    ["N", "S"],  # 直觉/感觉
    ["T", "F"],  # 思考/情感
    ["J", "P"]   # 判断/感知
]

# 情绪与MBTI的关联映射
EMOTION_TO_MBTI_WEIGHTS = {
    # I/E维度
    'anger':     {'I': 0.3, 'E': 0.7},  # 愤怒更可能是外向表现
    'disgust':   {'I': 0.6, 'E': 0.4},  # 厌恶略微偏向内向
    'fear':      {'I': 0.7, 'E': 0.3},  # 恐惧更可能是内向表现
    'happiness': {'I': 0.3, 'E': 0.7},  # 快乐更可能是外向表现
    'neutral':   {'I': 0.5, 'E': 0.5},  # 中性情绪无明显偏向
    'sadness':   {'I': 0.7, 'E': 0.3},  # 悲伤更可能是内向表现
    'surprise':  {'I': 0.4, 'E': 0.6},  # 惊讶略微偏向外向
    
    # N/S维度
    'anger':     {'N': 0.4, 'S': 0.6},  # 愤怒略微偏向感觉
    'disgust':   {'N': 0.5, 'S': 0.5},  # 厌恶无明显偏向
    'fear':      {'N': 0.6, 'S': 0.4},  # 恐惧略微偏向直觉
    'happiness': {'N': 0.5, 'S': 0.5},  # 快乐无明显偏向
    'neutral':   {'N': 0.5, 'S': 0.5},  # 中性无明显偏向
    'sadness':   {'N': 0.6, 'S': 0.4},  # 悲伤略微偏向直觉
    'surprise':  {'N': 0.7, 'S': 0.3},  # 惊讶更可能是直觉表现
    
    # T/F维度
    'anger':     {'T': 0.5, 'F': 0.5},  # 愤怒无明显偏向
    'disgust':   {'T': 0.6, 'F': 0.4},  # 厌恶略微偏向思考
    'fear':      {'T': 0.3, 'F': 0.7},  # 恐惧更可能是情感表现
    'happiness': {'T': 0.3, 'F': 0.7},  # 快乐更可能是情感表现
    'neutral':   {'T': 0.6, 'F': 0.4},  # 中性略微偏向思考
    'sadness':   {'T': 0.3, 'F': 0.7},  # 悲伤更可能是情感表现
    'surprise':  {'T': 0.4, 'F': 0.6},  # 惊讶略微偏向情感
    
    # J/P维度
    'anger':     {'J': 0.6, 'P': 0.4},  # 愤怒略微偏向判断
    'disgust':   {'J': 0.5, 'P': 0.5},  # 厌恶无明显偏向
    'fear':      {'J': 0.4, 'P': 0.6},  # 恐惧略微偏向感知
    'happiness': {'J': 0.4, 'P': 0.6},  # 快乐略微偏向感知
    'neutral':   {'J': 0.5, 'P': 0.5},  # 中性无明显偏向
    'sadness':   {'J': 0.5, 'P': 0.5},  # 悲伤无明显偏向
    'surprise':  {'J': 0.3, 'P': 0.7},  # 惊讶更可能是感知表现
}

# MBTI类型描述
MBTI_DESCRIPTIONS = {
    "en": {
        "INTJ": "Architect: Imaginative and strategic thinkers, with a plan for everything.",
        "INTP": "Logician: Innovative inventors with an unquenchable thirst for knowledge.",
        "ENTJ": "Commander: Bold, imaginative and strong-willed leaders, always finding a way – or making one.",
        "ENTP": "Debater: Smart and curious thinkers who cannot resist an intellectual challenge.",
        "INFJ": "Advocate: Quiet and mystical, yet very inspiring and tireless idealists.",
        "INFP": "Mediator: Poetic, kind and altruistic people, always eager to help a good cause.",
        "ENFJ": "Protagonist: Charismatic and inspiring leaders, able to mesmerize their listeners.",
        "ENFP": "Campaigner: Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.",
        "ISTJ": "Logistician: Practical and fact-minded individuals, whose reliability cannot be doubted.",
        "ISFJ": "Defender: Very dedicated and warm protectors, always ready to defend their loved ones.",
        "ESTJ": "Executive: Excellent administrators, unsurpassed at managing things – or people.",
        "ESFJ": "Consul: Extraordinarily caring, social and popular people, always eager to help.",
        "ISTP": "Virtuoso: Bold and practical experimenters, masters of all kinds of tools.",
        "ISFP": "Adventurer: Flexible and charming artists, always ready to explore and experience something new.",
        "ESTP": "Entrepreneur: Smart, energetic and very perceptive people, who truly enjoy living on the edge.",
        "ESFP": "Entertainer: Spontaneous, energetic and enthusiastic entertainers – life is never boring around them."
    },
    "zh": {
        "INTJ": "建筑师：富有想象力和战略性思维的策划者，对每件事都有一个计划。",
        "INTP": "逻辑学家：创新的发明家，对知识有着不可抑制的渴望。",
        "ENTJ": "指挥官：大胆、富有想象力和意志坚强的领导者，总能找到方法或创造方法。",
        "ENTP": "辩论家：聪明好奇的思考者，无法抗拒智力挑战。",
        "INFJ": "提倡者：安静而神秘，但非常鼓舞人心且不知疲倦的理想主义者。",
        "INFP": "调停者：诗意、善良和利他的人，总是渴望帮助正当的事业。",
        "ENFJ": "主角：魅力四射且鼓舞人心的领导者，能够吸引听众。",
        "ENFP": "活动家：热情、有创造力和善于社交的自由精神，总能找到微笑的理由。",
        "ISTJ": "物流师：务实且注重事实的人，其可靠性毋庸置疑。",
        "ISFJ": "卫士：非常专注和温暖的保护者，随时准备保护他们所爱的人。",
        "ESTJ": "总裁：优秀的管理者，在管理事物或人员方面无人能及。",
        "ESFJ": "执政官：非常关心他人、善于社交且受欢迎的人，总是渴望提供帮助。",
        "ISTP": "鉴赏家：大胆而实际的实验者，各种工具的掌握者。",
        "ISFP": "探险家：灵活而迷人的艺术家，总是准备探索和体验新事物。",
        "ESTP": "企业家：聪明、精力充沛且非常敏锐的人，真正享受生活在边缘。",
        "ESFP": "表演者：自发、精力充沛和热情的表演者——他们身边的生活从不无聊。"
    }
}

# 模型定义
class EmotionRecognitionModel(torch.nn.Module):
    """情绪识别模型"""
    def __init__(self, num_classes=7):
        super(EmotionRecognitionModel, self).__init__()
        # 使用ViT模型架构
        self.backbone = torch.hub.load('facebookresearch/deit:main', 
                                    'deit_tiny_patch16_224', 
                                    pretrained=True)
        # 替换分类头
        self.backbone.head = torch.nn.Linear(self.backbone.head.in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class TranslationService:
    """中文到英文的翻译服务"""
    def __init__(self, model_name=TRANSLATION_MODEL):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 检查是否安装了 sentencepiece
        try:
            import sentencepiece
            logger.info("SentencePiece 库已找到")
        except ImportError:
            logger.warning("SentencePiece 库未安装，请使用 'pip install sentencepiece' 安装")
            self.tokenizer = None
            self.model = None
            return
            
        try:
            logger.info(f"正在加载翻译模型: {model_name}")
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            logger.info("翻译模型加载成功")
        except Exception as e:
            logger.warning(f"翻译模型加载失败: {e}")
            self.tokenizer = None
            self.model = None
            
    def translate(self, text, max_length=512):
        """将中文文本翻译为英文"""
        if not text or not text.strip():
            return ""
        
        if self.model is None or self.tokenizer is None:
            logger.warning("翻译模型不可用，返回原文")
            return text
            
        try:
            # 将长文本分成较小的块
            text_chunks = self._split_text(text)
            translated_chunks = []
            
            for chunk in text_chunks:
                # 编码文本
                inputs = self.tokenizer([chunk], return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成翻译
                translated = self.model.generate(**inputs, max_length=max_length)
                translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
                translated_chunks.append(translated_text)
            
            # 合并翻译结果
            result = " ".join(translated_chunks)
            return result
        except Exception as e:
            logger.error(f"翻译过程中出错: {e}")
            return text
    
    def _split_text(self, text, max_chars=1000):
        """将长文本分割成较小的块以便翻译"""
        import re
        sentences = re.split(r'(?<=[。！？.!?])', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            if len(current_chunk) + len(sentence) <= max_chars:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

def detect_language(text):
    """检测文本语言"""
    try:
        lang, _ = langid.classify(text)
        return lang
    except:
        # 简单的基于字符的检测
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        if chinese_chars > len(text) * 0.15:  # 如果15%以上的字符是中文
            return 'zh'
        return 'en'

# 模型加载
def load_models():
    """加载所有必要的模型"""
    models = {}
    
    # 1. 加载情绪识别模型
    try:
        # 创建设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {device}")
        
        # 加载情绪模型
        logger.info(f"加载情绪识别模型: {EMOTION_MODEL_PATH}")
        checkpoint = torch.load(EMOTION_MODEL_PATH, map_location=device)
        
        # 从检查点提取模型状态和类别
        if 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
            emotion_categories = checkpoint.get('emotion_categories', EMOTION_CATEGORIES)
        else:
            model_state = checkpoint
            emotion_categories = EMOTION_CATEGORIES
        
        # 创建模型实例
        emotion_model = EmotionRecognitionModel(num_classes=len(emotion_categories))
        
        # 加载模型权重
        emotion_model.load_state_dict(model_state, strict=False)
        emotion_model.to(device)
        emotion_model.eval()
        
        models['emotion_model'] = emotion_model
        models['emotion_categories'] = emotion_categories
        models['device'] = device
        
        logger.info("情绪识别模型加载成功")
    except Exception as e:
        logger.error(f"加载情绪识别模型时出错: {e}")
    
    # 2. 加载文本分析模型
    try:
        logger.info(f"加载文本分析模型: {TEXT_MODEL_DIR}")
        
        # 加载模型组件
        with open(os.path.join(TEXT_MODEL_DIR, 'model.pkl'), 'rb') as f:
            text_model = pickle.load(f)
        
        with open(os.path.join(TEXT_MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open(os.path.join(TEXT_MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(os.path.join(TEXT_MODEL_DIR, 'config.json'), 'r') as f:
            config = json.load(f)
        
        models['text_model'] = text_model
        models['vectorizer'] = vectorizer
        models['label_encoder'] = label_encoder
        models['text_config'] = config
        
        logger.info("文本分析模型加载成功")
    except Exception as e:
        logger.error(f"加载文本分析模型时出错: {e}")
    
    # 3. 加载翻译模型
    try:
        translation_service = TranslationService()
        models['translation_service'] = translation_service
        logger.info("翻译服务加载成功")
    except Exception as e:
        logger.error(f"加载翻译服务时出错: {e}")
        models['translation_service'] = None
    
    # 4. 加载问题库
    try:
        questions_file = os.path.join(QUESTIONS_DIR, 'questions.json')
        if not os.path.exists(questions_file):
            # 创建示例问题
            create_example_questions()
            
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
            
        models['questions'] = questions
        logger.info(f"问题库加载成功，共{sum(len(q['questions']) for q in questions['dimensions'])}个问题")
    except Exception as e:
        logger.error(f"加载问题库时出错: {e}")
        models['questions'] = create_example_questions()
    
    return models

def create_example_questions():
    """创建示例问题库"""
    questions = {
        "dimensions": [
            {
                "dimension": "E/I",
                "questions": [
                    {
                        "en": "How do you typically recharge after a long day?",
                        "zh": "在漫长的一天之后，你通常如何恢复精力？"
                    },
                    {
                        "en": "Describe a perfect weekend for you.",
                        "zh": "描述一下你心目中完美的周末。"
                    },
                    {
                        "en": "How do you feel about impromptu social gatherings?",
                        "zh": "你对即兴社交聚会有什么看法？"
                    }
                ]
            },
            {
                "dimension": "S/N",
                "questions": [
                    {
                        "en": "When solving a problem, do you prefer established methods or creating new approaches?",
                        "zh": "解决问题时，你更喜欢使用已建立的方法还是创造新的方法？"
                    },
                    {
                        "en": "How important are traditions to you?",
                        "zh": "传统对你有多重要？"
                    },
                    {
                        "en": "When reading, do you focus more on details or the overall meaning?",
                        "zh": "阅读时，你更关注细节还是整体含义？"
                    }
                ]
            },
            {
                "dimension": "T/F",
                "questions": [
                    {
                        "en": "How do you typically make important decisions?",
                        "zh": "你通常如何做出重要决定？"
                    },
                    {
                        "en": "Describe how you resolve conflicts with others.",
                        "zh": "描述一下你如何解决与他人的冲突。"
                    },
                    {
                        "en": "What is more important to you in a workplace: harmony or efficiency?",
                        "zh": "在工作场所，什么对你更重要：和谐还是效率？"
                    }
                ]
            },
            {
                "dimension": "J/P",
                "questions": [
                    {
                        "en": "How do you approach deadlines?",
                        "zh": "你如何看待截止日期？"
                    },
                    {
                        "en": "Describe your approach to planning a trip.",
                        "zh": "描述一下你计划旅行的方式。"
                    },
                    {
                        "en": "How do you feel when plans change unexpectedly?",
                        "zh": "当计划意外改变时，你有什么感受？"
                    }
                ]
            }
        ]
    }
    
    # 保存问题库
    questions_file = os.path.join(QUESTIONS_DIR, 'questions.json')
    os.makedirs(os.path.dirname(questions_file), exist_ok=True)
    
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
        
    logger.info(f"创建示例问题库成功: {questions_file}")
    return questions

# 图像预处理
def preprocess_image(image_data, target_size=(224, 224)):
    """预处理图像用于情绪识别"""
    try:
        # 解码Base64图像
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # 定义预处理转换
        preprocess = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        # 应用预处理
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
        
        return input_batch
    except Exception as e:
        logger.error(f"图像预处理出错: {e}")
        return None

def preprocess_text(text, translation_service=None):
    """预处理文本用于分析"""
    if not text:
        return "", "en"
    
    # 检测语言
    lang = detect_language(text)
    
    # 如果是中文且有翻译服务，则翻译为英文
    translated_text = None
    if lang == 'zh' and translation_service is not None:
        translated_text = translation_service.translate(text)
        logger.info(f"已将中文文本翻译为英文: {text} -> {translated_text}")
        text_for_analysis = translated_text
    else:
        text_for_analysis = text
    
    # 基础文本清理
    text_for_analysis = text_for_analysis.lower()
    
    return text_for_analysis, lang, translated_text

# 预测函数
def predict_emotion(image_data, models, save_image=False, session_id=None):
    """预测图像中的情绪，并可选择保存原始图像"""
    try:
        if 'emotion_model' not in models:
            return {'error': '情绪识别模型未加载'}
        
        # 如果需要保存，先保存原始图像
        if save_image and session_id:
            try:
                # 解码Base64图像
                image_bytes = base64.b64decode(image_data.split(',')[1])
                image = Image.open(io.BytesIO(image_bytes))
                
                # 创建用户会话目录
                user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
                os.makedirs(user_dir, exist_ok=True)
                
                # 生成时间戳文件名
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"emotion_{timestamp}.jpg"
                image_path = os.path.join(user_dir, image_filename)
                
                # 保存图像
                image.save(image_path)
                logger.info(f"已保存情绪分析图像: {image_path}")
            except Exception as e:
                logger.error(f"保存情绪分析图像时出错: {e}")
        
        # 预处理图像
        input_batch = preprocess_image(image_data)
        if input_batch is None:
            return {'error': '图像预处理失败'}
        
        # 将输入移到正确的设备
        input_batch = input_batch.to(models['device'])
        
        # 进行预测
        with torch.no_grad():
            model = models['emotion_model']
            output = model(input_batch)
            probabilities = F.softmax(output, dim=1)[0]
        
        # 获取预测结果
        emotion_categories = models.get('emotion_categories', EMOTION_CATEGORIES)
        
        # 转换为可序列化的格式
        emotions_dict = {emotion: prob.item() for emotion, prob in zip(emotion_categories, probabilities)}
        
        # 获取概率最高的情绪
        predicted_emotion = emotion_categories[torch.argmax(probabilities).item()]
        confidence = torch.max(probabilities).item()
        
        result = {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'emotions': emotions_dict,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 保存情绪分析结果
        if session_id:
            user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # 情绪数据文件路径
            emotions_file = os.path.join(user_dir, 'emotion_data.jsonl')
            
            # 追加情绪数据
            with open(emotions_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
        
        return result
    except Exception as e:
        logger.error(f"情绪预测出错: {e}")
        return {'error': f'情绪预测失败: {str(e)}'}

def predict_mbti_from_text(text, models, session_id=None):
    """从文本预测MBTI类型"""
    try:
        if 'text_model' not in models or 'vectorizer' not in models:
            return {'error': '文本分析模型未加载'}
        
        # 预处理文本
        processed_text, original_lang, translated_text = preprocess_text(text, models.get('translation_service'))
        if not processed_text:
            return {'error': '文本预处理失败或文本为空'}
        
        # 向量化文本
        text_vector = models['vectorizer'].transform([processed_text])
        
        # 预测MBTI类型
        text_model = models['text_model']
        label_encoder = models['label_encoder']
        
        # 获取预测结果
        if hasattr(text_model, 'predict_proba'):
            # 获取概率
            proba = text_model.predict_proba(text_vector)[0]
            prediction_idx = proba.argmax()
            confidence = proba[prediction_idx]
            
            # 获取前三个最可能的类型
            top_indices = proba.argsort()[-3:][::-1]
            top_predictions = [
                {
                    'type': label_encoder.inverse_transform([idx])[0],
                    'confidence': float(proba[idx])
                } for idx in top_indices
            ]
        else:
            # 不支持概率的模型
            prediction_idx = text_model.predict(text_vector)[0]
            confidence = 1.0
            top_predictions = [
                {
                    'type': label_encoder.inverse_transform([prediction_idx])[0],
                    'confidence': 1.0
                }
            ]
        
        # 获取预测的MBTI类型
        mbti_type = label_encoder.inverse_transform([prediction_idx])[0]
        
        # 进行每个维度的分析
        dimension_analysis = analyze_mbti_dimensions(processed_text)
        
        result = {
            'mbti_type': mbti_type,
            'confidence': float(confidence),
            'top_predictions': top_predictions,
            'dimension_analysis': dimension_analysis,
            'original_language': original_lang,
            'processed_text': processed_text,
            'original_text': text,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if translated_text:
            result['translated'] = True
            result['translated_text'] = translated_text
        
        # 保存文本分析结果
        if session_id:
            user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # 文本数据文件路径
            text_file = os.path.join(user_dir, 'text_data.jsonl')
            
            # 追加文本数据
            with open(text_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
        
        return result
    except Exception as e:
        logger.error(f"MBTI预测出错: {e}")
        return {'error': f'MBTI预测失败: {str(e)}'}

def analyze_mbti_dimensions(text):
    """分析文本在四个MBTI维度上的得分"""
    # 各维度的关键词和权重
    dimension_keywords = {
        'I': {'introvert': 2, 'quiet': 1, 'alone': 1, 'private': 1, 'inner': 1, 
              'depth': 1, 'focus': 1, 'thought': 1, 'peace': 1, 'solitude': 2},
        'E': {'extrovert': 2, 'social': 1, 'talk': 1, 'people': 1, 'engage': 1, 
              'outgoing': 1, 'active': 1, 'external': 1, 'interact': 1, 'energetic': 1},
        'N': {'intuitive': 2, 'abstract': 1, 'future': 1, 'imagine': 1, 'possibility': 1, 
              'pattern': 1, 'meaning': 1, 'theory': 1, 'concept': 1, 'insight': 1},
        'S': {'sensing': 2, 'detail': 1, 'present': 1, 'practical': 1, 'concrete': 1, 
              'reality': 1, 'fact': 1, 'experience': 1, 'observation': 1, 'specific': 1},
        'T': {'thinking': 2, 'logic': 1, 'analysis': 1, 'objective': 1, 'principle': 1, 
              'rational': 1, 'critique': 1, 'reason': 1, 'system': 1, 'truth': 1},
        'F': {'feeling': 2, 'value': 1, 'harmony': 1, 'empathy': 1, 'personal': 1, 
              'compassion': 1, 'ethic': 1, 'human': 1, 'subjective': 1, 'emotion': 1},
        'J': {'judging': 2, 'plan': 1, 'organize': 1, 'structure': 1, 'decide': 1, 
              'control': 1, 'certain': 1, 'schedule': 1, 'complete': 1, 'deadline': 1},
        'P': {'perceiving': 2, 'flexible': 1, 'adapt': 1, 'explore': 1, 'option': 1, 
              'spontaneous': 1, 'open': 1, 'process': 1, 'possibility': 1, 'casual': 1}
    }
    
    # 计算每个维度的得分
    ie_score = 0  # 负值偏向I，正值偏向E
    ns_score = 0  # 正值偏向N，负值偏向S
    tf_score = 0  # 负值偏向T，正值偏向F
    jp_score = 0  # 负值偏向J，正值偏向P
    
    words = text.lower().split()
    
    # 计算维度得分
    for word in words:
        # I/E维度
        for keyword, weight in dimension_keywords['I'].items():
            if keyword in word:
                ie_score -= weight
        for keyword, weight in dimension_keywords['E'].items():
            if keyword in word:
                ie_score += weight
        
        # N/S维度
        for keyword, weight in dimension_keywords['N'].items():
            if keyword in word:
                ns_score += weight
        for keyword, weight in dimension_keywords['S'].items():
            if keyword in word:
                ns_score -= weight
        
        # T/F维度
        for keyword, weight in dimension_keywords['T'].items():
            if keyword in word:
                tf_score -= weight
        for keyword, weight in dimension_keywords['F'].items():
            if keyword in word:
                tf_score += weight
        
        # J/P维度
        for keyword, weight in dimension_keywords['J'].items():
            if keyword in word:
                jp_score -= weight
        for keyword, weight in dimension_keywords['P'].items():
            if keyword in word:
                jp_score += weight
    
    # 归一化得分
    def normalize_score(score, words_count):
        if words_count == 0:
            return 0
        normalized = score / (words_count ** 0.5)  # 使用平方根缩放
        return max(min(normalized, 10), -10)  # 限制在-10到10的范围内
    
    words_count = len(words)
    ie_score = normalize_score(ie_score, words_count)
    ns_score = normalize_score(ns_score, words_count)
    tf_score = normalize_score(tf_score, words_count)
    jp_score = normalize_score(jp_score, words_count)
    
    return {
        'IE': {'score': ie_score, 'preference': 'I' if ie_score < 0 else 'E'},
        'NS': {'score': ns_score, 'preference': 'N' if ns_score > 0 else 'S'},
        'TF': {'score': tf_score, 'preference': 'T' if tf_score < 0 else 'F'},
        'JP': {'score': jp_score, 'preference': 'J' if jp_score < 0 else 'P'}
    }

def infer_mbti_dimensions_from_emotion(emotion):
    """从情绪推断可能的MBTI维度偏好"""
    # 使用情绪与MBTI维度的关联权重
    dimensions = []
    
    # 为每个维度获取情绪对应的权重
    for dim_pair in MBTI_DIMENSIONS:
        dim_weights = {dim: EMOTION_TO_MBTI_WEIGHTS[emotion][dim] for dim in dim_pair}
        dimensions.append(dim_weights)
    
    return dimensions

def integrate_predictions(emotion_result, text_result, session_emotion_history=None):
    """整合表情和文本预测结果，生成最终的MBTI类型"""
    if 'error' in emotion_result or 'error' in text_result:
        # 如果任一预测有错误，仍然尝试使用可用结果
        if 'error' in emotion_result and 'error' in text_result:
            return {'error': '表情和文本预测均失败'}
        elif 'error' in emotion_result:
            return text_result
        else:
            # 使用情绪历史增强情绪分析（如果可用）
            if session_emotion_history:
                # 计算情绪历史的平均值
                emotion_counts = {}
                total = 0
                
                for emotion_data in session_emotion_history:
                    emotion = emotion_data['predicted_emotion']
                    confidence = emotion_data['confidence']
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + confidence
                    total += confidence
                
                # 获取最主要的情绪
                if total > 0:
                    main_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
                    emotion_result['predicted_emotion'] = main_emotion
                    emotion_result['confidence'] = emotion_counts[main_emotion] / total
            
            # 仅使用表情预测
            return infer_mbti_from_emotion(emotion_result)
    
    # 两种预测均成功，进行整合
    text_mbti = text_result['mbti_type']
    
    # 获取表情预测的情绪及其置信度
    emotion = emotion_result['predicted_emotion']
    emotion_confidence = emotion_result['confidence']
    
    # 获取文本预测的MBTI及其置信度
    text_confidence = text_result['confidence']
    
    # 从情绪推断可能的MBTI维度
    emotion_dimensions = infer_mbti_dimensions_from_emotion(emotion)
    
    # 获取文本预测的MBTI维度
    text_dimensions = [dim for dim in text_mbti]
    
    # 基于置信度和权重计算最终MBTI
    # 文本分析通常更可靠，给予更高权重
    text_weight = 0.7
    emotion_weight = 0.3
    
    # 调整权重基于各自的置信度
    total_weight = text_weight * text_confidence + emotion_weight * emotion_confidence
    if total_weight > 0:
        adjusted_text_weight = (text_weight * text_confidence) / total_weight
        adjusted_emotion_weight = (emotion_weight * emotion_confidence) / total_weight
    else:
        adjusted_text_weight = 0.7
        adjusted_emotion_weight = 0.3
    
    # 整合每个维度
    final_dimensions = []
    dimension_scores = {}
    
    for i, (text_dim, emotion_dims) in enumerate(zip(text_dimensions, emotion_dimensions)):
        # 获取每个维度的可能值
        dim_options = MBTI_DIMENSIONS[i]
        
        # 计算文本模型对该维度的贡献
        text_idx = dim_options.index(text_dim)
        text_score = [0, 0]
        text_score[text_idx] = 1
        
        # 计算情绪模型对该维度的贡献
        emotion_score = [emotion_dims[dim] for dim in dim_options]
        
        # 整合得分
        integrated_score = [
            adjusted_text_weight * ts + adjusted_emotion_weight * es
            for ts, es in zip(text_score, emotion_score)
        ]
        
        # 选择得分较高的维度
        final_dim = dim_options[np.argmax(integrated_score)]
        final_dimensions.append(final_dim)
        
        # 记录维度得分
        dimension_scores[dim_options[0] + dim_options[1]] = {
            'score': integrated_score[0] - integrated_score[1],  # 第一个选项的优势
            'preference': final_dim
        }
    
    # 组合成最终MBTI类型
    final_mbti = ''.join(final_dimensions)
    
    # 准备返回结果
    result = {
        'mbti_type': final_mbti,
        'text_analysis': text_result,
        'emotion_analysis': emotion_result,
        'integration_weights': {
            'text': adjusted_text_weight,
            'emotion': adjusted_emotion_weight
        },
        'dimension_analysis': dimension_scores,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    return result

def infer_mbti_from_emotion(emotion_result):
    """仅从表情推断MBTI类型"""
    if 'error' in emotion_result:
        return {'error': '无法从表情推断MBTI类型'}
    
    emotion = emotion_result['predicted_emotion']
    
    # 从情绪推断可能的MBTI维度
    emotion_dimensions = infer_mbti_dimensions_from_emotion(emotion)
    
    # 选择每个维度中概率最高的选项
    final_dimensions = []
    for dim_pair in emotion_dimensions:
        options = list(dim_pair.keys())
        probs = list(dim_pair.values())
        final_dimensions.append(options[np.argmax(probs)])
    
    # 组合成MBTI类型
    inferred_mbti = ''.join(final_dimensions)
    
    # 准备维度分析
    dimension_analysis = {}
    for i, dim_pair in enumerate(emotion_dimensions):
        dim_name = ''.join(MBTI_DIMENSIONS[i])
        options = list(dim_pair.keys())
        probs = list(dim_pair.values())
        score = probs[0] - probs[1]  # 第一个选项的优势
        preference = options[np.argmax(probs)]
        dimension_analysis[dim_name] = {
            'score': score,
            'preference': preference
        }
    
    result = {
        'mbti_type': inferred_mbti,
        'source': 'emotion_only',
        'emotion_analysis': emotion_result,
        'dimension_analysis': dimension_analysis,
        'note': '此结果仅基于表情分析，可能不如结合文本分析准确',
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    return result

# 数据收集和导出函数
def save_user_actual_mbti(session_id, actual_mbti, confidence_level=None):
    """保存用户提供的真实MBTI类型"""
    if not session_id or not actual_mbti:
        return {'error': '会话ID和MBTI类型是必需的'}
    
    # 验证MBTI类型格式
    if not re.match(r'^[IE][NS][TF][JP]$', actual_mbti.upper()):
        return {'error': 'MBTI类型格式无效'}
    
    try:
        # 创建用户会话目录
        user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # 准备数据
        actual_mbti_data = {
            'actual_mbti': actual_mbti.upper(),
            'confidence_level': confidence_level,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # 保存到文件
        with open(os.path.join(user_dir, 'actual_mbti.json'), 'w', encoding='utf-8') as f:
            json.dump(actual_mbti_data, f, ensure_ascii=False, indent=4)
        
        # 更新会话元数据
        session_meta_path = os.path.join(user_dir, 'session_meta.json')
        
        if os.path.exists(session_meta_path):
            with open(session_meta_path, 'r', encoding='utf-8') as f:
                session_meta = json.load(f)
        else:
            session_meta = {
                'session_id': session_id,
                'created_at': datetime.datetime.now().isoformat()
            }
        
        session_meta['actual_mbti'] = actual_mbti.upper()
        session_meta['actual_mbti_confidence'] = confidence_level
        session_meta['actual_mbti_timestamp'] = datetime.datetime.now().isoformat()
        
        with open(session_meta_path, 'w', encoding='utf-8') as f:
            json.dump(session_meta, f, ensure_ascii=False, indent=4)
        
        return {'success': True, 'message': '真实MBTI类型已保存'}
    
    except Exception as e:
        logger.error(f"保存真实MBTI数据时出错: {e}")
        return {'error': f'保存真实MBTI数据失败: {str(e)}'}

def export_session_data(session_id, format='json'):
    """导出会话数据"""
    if not session_id:
        return {'error': '会话ID是必需的'}
    
    try:
        user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
        
        if not os.path.exists(user_dir):
            return {'error': '找不到会话数据'}
        
        # 收集所有数据文件
        data = {
            'session_id': session_id,
            'export_timestamp': datetime.datetime.now().isoformat(),
            'emotion_data': [],
            'text_data': [],
            'responses': [],
            'actual_mbti': None
        }
        
        # 加载情绪数据
        emotion_file = os.path.join(user_dir, 'emotion_data.jsonl')
        if os.path.exists(emotion_file):
            with open(emotion_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data['emotion_data'].append(json.loads(line))
        
        # 加载文本数据
        text_file = os.path.join(user_dir, 'text_data.jsonl')
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data['text_data'].append(json.loads(line))
        
        # 加载会话响应
        responses_file = os.path.join(user_dir, 'responses.jsonl')
        if os.path.exists(responses_file):
            with open(responses_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data['responses'].append(json.loads(line))
        
        # 加载真实MBTI数据
        actual_mbti_file = os.path.join(user_dir, 'actual_mbti.json')
        if os.path.exists(actual_mbti_file):
            with open(actual_mbti_file, 'r', encoding='utf-8') as f:
                data['actual_mbti'] = json.load(f)
        
        # 加载会话元数据
        meta_file = os.path.join(user_dir, 'session_meta.json')
        if os.path.exists(meta_file):
            with open(meta_file, 'r', encoding='utf-8') as f:
                data['meta'] = json.load(f)
        
        # 导出数据
        if format.lower() == 'csv':
            # 创建导出目录
            export_dir = os.path.join(RESULTS_DIR, 'exports', session_id)
            os.makedirs(export_dir, exist_ok=True)
            
            # 导出情绪数据
            if data['emotion_data']:
                emotion_df = pd.DataFrame(data['emotion_data'])
                emotion_df.to_csv(os.path.join(export_dir, 'emotion_data.csv'), index=False)
            
            # 导出文本数据
            if data['text_data']:
                text_df = pd.DataFrame(data['text_data'])
                text_df.to_csv(os.path.join(export_dir, 'text_data.csv'), index=False)
            
            # 导出响应数据
            if data['responses']:
                responses_df = pd.DataFrame(data['responses'])
                responses_df.to_csv(os.path.join(export_dir, 'responses.csv'), index=False)
            
            return {
                'success': True, 
                'format': 'csv', 
                'files': [
                    os.path.join(export_dir, 'emotion_data.csv'),
                    os.path.join(export_dir, 'text_data.csv'),
                    os.path.join(export_dir, 'responses.csv')
                ]
            }
        else:
            # 导出为JSON
            export_path = os.path.join(RESULTS_DIR, 'exports')
            os.makedirs(export_path, exist_ok=True)
            
            export_file = os.path.join(export_path, f"{session_id}_export.json")
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            return {'success': True, 'format': 'json', 'file': export_file}
    
    except Exception as e:
        logger.error(f"导出会话数据时出错: {e}")
        return {'error': f'导出会话数据失败: {str(e)}'}

def generate_research_dataset():
    """生成用于研究的数据集，包含所有会话数据"""
    try:
        # 检查数据收集目录是否存在
        if not os.path.exists(DATA_COLLECTION_DIR):
            return {'error': '找不到数据收集目录'}
        
        # 获取所有会话目录
        session_dirs = [d for d in os.listdir(DATA_COLLECTION_DIR) 
                        if os.path.isdir(os.path.join(DATA_COLLECTION_DIR, d))]
        
        if not session_dirs:
            return {'error': '找不到会话数据'}
        
        # 准备数据框
        all_sessions = []
        all_emotions = []
        all_texts = []
        all_responses = []
        
        for session_id in session_dirs:
            session_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            
            # 会话元数据
            meta_file = os.path.join(session_dir, 'session_meta.json')
            session_meta = None
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    session_meta = json.load(f)
                    all_sessions.append(session_meta)
            else:
                session_meta = {'session_id': session_id}
                all_sessions.append(session_meta)
            
            # 真实MBTI数据
            actual_mbti_file = os.path.join(session_dir, 'actual_mbti.json')
            if os.path.exists(actual_mbti_file):
                with open(actual_mbti_file, 'r', encoding='utf-8') as f:
                    actual_mbti_data = json.load(f)
                    session_meta.update(actual_mbti_data)
            
            # 情绪数据
            emotion_file = os.path.join(session_dir, 'emotion_data.jsonl')
            if os.path.exists(emotion_file):
                with open(emotion_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            emotion_data = json.loads(line)
                            emotion_data['session_id'] = session_id
                            all_emotions.append(emotion_data)
            
            # 文本数据
            text_file = os.path.join(session_dir, 'text_data.jsonl')
            if os.path.exists(text_file):
                with open(text_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            text_data = json.loads(line)
                            text_data['session_id'] = session_id
                            all_texts.append(text_data)
            
            # 响应数据
            responses_file = os.path.join(session_dir, 'responses.jsonl')
            if os.path.exists(responses_file):
                with open(responses_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            response_data = json.loads(line)
                            response_data['session_id'] = session_id
                            all_responses.append(response_data)
        
        # 创建导出目录
        export_dir = os.path.join(RESULTS_DIR, 'research')
        os.makedirs(export_dir, exist_ok=True)
        
        # 导出为CSV
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        
        # 会话元数据
        if all_sessions:
            sessions_df = pd.DataFrame(all_sessions)
            sessions_df.to_csv(os.path.join(export_dir, f'sessions_{timestamp}.csv'), index=False)
        
        # 情绪数据
        if all_emotions:
            emotions_df = pd.DataFrame(all_emotions)
            emotions_df.to_csv(os.path.join(export_dir, f'emotions_{timestamp}.csv'), index=False)
        
        # 文本数据
        if all_texts:
            texts_df = pd.DataFrame(all_texts)
            texts_df.to_csv(os.path.join(export_dir, f'texts_{timestamp}.csv'), index=False)
        
        # 响应数据
        if all_responses:
            responses_df = pd.DataFrame(all_responses)
            responses_df.to_csv(os.path.join(export_dir, f'responses_{timestamp}.csv'), index=False)
        
        return {
            'success': True,
            'files': [
                os.path.join(export_dir, f'sessions_{timestamp}.csv'),
                os.path.join(export_dir, f'emotions_{timestamp}.csv'),
                os.path.join(export_dir, f'texts_{timestamp}.csv'),
                os.path.join(export_dir, f'responses_{timestamp}.csv')
            ],
            'sessions_count': len(all_sessions),
            'emotions_count': len(all_emotions),
            'texts_count': len(all_texts),
            'responses_count': len(all_responses)
        }
    
    except Exception as e:
        logger.error(f"生成研究数据集时出错: {e}")
        return {'error': f'生成研究数据集失败: {str(e)}'}

# 定义API路由
@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory(STATIC_DIR, path)

@app.route('/api/predict_emotion', methods=['POST'])
def api_predict_emotion():
    """
    情绪预测API
    
    请求体：
        {
            "image_data": "data:image/jpeg;base64,...",  // Base64编码的图像数据
            "session_id": "unique_session_id",          // 可选，会话ID
            "save_image": true                          // 可选，是否保存原始图像
        }
    
    响应：
        {
            "predicted_emotion": "happiness",
            "confidence": 0.85,
            "emotions": {
                "anger": 0.05,
                "disgust": 0.02,
                ...
            },
            "timestamp": "2023-01-01T12:00:00.000Z"
        }
    """
    try:
        data = request.json
        
        if not data or 'image_data' not in data:
            return jsonify({'error': '请提供图像数据'}), 400
        
        image_data = data['image_data']
        session_id = data.get('session_id')
        save_image = data.get('save_image', False)
        
        # 获取模型
        models = app.config.get('models')
        if not models:
            models = load_models()
            app.config['models'] = models
        
        # 预测情绪
        result = predict_emotion(image_data, models, save_image, session_id)
        
        # 添加会话ID
        if session_id:
            # 记录情绪历史
            if 'predicted_emotion' in result:
                session_data = app.config.get('sessions', {})
                if session_id not in session_data:
                    session_data[session_id] = {'emotion_history': []}
                
                session_data[session_id]['emotion_history'].append({
                    'predicted_emotion': result['predicted_emotion'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
                
                app.config['sessions'] = session_data
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/predict_mbti', methods=['POST'])
def api_predict_mbti():
    """
    MBTI预测API
    
    请求体：
        {
            "text": "用户的回答文本",
            "image_data": "data:image/jpeg;base64,...",  // 可选，Base64编码的图像数据
            "session_id": "unique_session_id",          // 可选，会话ID
            "save_data": true                           // 可选，是否保存数据
        }
    
    响应：
        {
            "mbti_type": "INTJ",
            "dimension_analysis": {
                "IE": {"score": -5.2, "preference": "I"},
                ...
            },
            "text_analysis": {...},
            "emotion_analysis": {...}  // 如果提供了图像
        }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({'error': '请提供文本数据'}), 400
        
        text = data['text']
        session_id = data.get('session_id')
        save_data = data.get('save_data', True)
        
        # 获取模型
        models = app.config.get('models')
        if not models:
            models = load_models()
            app.config['models'] = models
        
        # 获取会话数据
        session_data = None
        if session_id:
            sessions = app.config.get('sessions', {})
            session_data = sessions.get(session_id, {})
        
        # 预测MBTI
        text_result = predict_mbti_from_text(text, models, session_id if save_data else None)
        
        # 如果提供了图像，也进行情绪预测
        emotion_result = {'error': '未提供图像数据'}
        if 'image_data' in data:
            emotion_result = predict_emotion(
                data['image_data'], 
                models, 
                save_image=save_data, 
                session_id=session_id if save_data else None
            )
            
            # 记录情绪历史
            if session_id and 'predicted_emotion' in emotion_result:
                if 'sessions' not in app.config:
                    app.config['sessions'] = {}
                
                if session_id not in app.config['sessions']:
                    app.config['sessions'][session_id] = {'emotion_history': []}
                
                app.config['sessions'][session_id]['emotion_history'].append({
                    'predicted_emotion': emotion_result['predicted_emotion'],
                    'confidence': emotion_result['confidence'],
                    'timestamp': datetime.datetime.now().isoformat()
                })
        
        # 整合结果
        emotion_history = session_data.get('emotion_history') if session_data else None
        result = integrate_predictions(emotion_result, text_result, emotion_history)
        
        # 保存整合结果
        if save_data and session_id:
            user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            os.makedirs(user_dir, exist_ok=True)
            
            # 整合结果文件路径
            integrated_file = os.path.join(user_dir, 'integrated_results.jsonl')
            
            # 追加整合结果
            with open(integrated_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result) + '\n')
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/get_question', methods=['GET'])
def api_get_question():
    """
    获取随机问题API
    
    请求参数：
        dimension: 可选，指定维度（E/I, S/N, T/F, J/P）
        language: 可选，指定语言（en, zh），默认en
    
    响应：
        {
            "question": {
                "en": "English question",
                "zh": "中文问题"
            },
            "dimension": "E/I"
        }
    """
    try:
        dimension = request.args.get('dimension')
        language = request.args.get('language', 'en')
        
        # 获取问题库
        models = app.config.get('models')
        if not models:
            models = load_models()
            app.config['models'] = models
        
        questions = models.get('questions', create_example_questions())
        
        # 选择问题
        if dimension:
            # 查找匹配的维度
            matching_dims = [d for d in questions['dimensions'] if d['dimension'] == dimension]
            if not matching_dims:
                return jsonify({'error': f'找不到维度: {dimension}'}), 404
            
            dim_questions = matching_dims[0]['questions']
        else:
            # 随机选择一个维度
            random_dim = random.choice(questions['dimensions'])
            dimension = random_dim['dimension']
            dim_questions = random_dim['questions']
        
        # 随机选择一个问题
        question = random.choice(dim_questions)
        
        return jsonify({
            'question': question,
            'dimension': dimension
        })
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/get_mbti_description', methods=['GET'])
def api_get_mbti_description():
    """
    获取MBTI类型描述API
    
    请求参数：
        mbti: MBTI类型（如INTJ）
        language: 可选，指定语言（en, zh），默认en
    
    响应：
        {
            "description": "类型描述文本"
        }
    """
    try:
        mbti = request.args.get('mbti')
        language = request.args.get('language', 'en')
        
        if not mbti:
            return jsonify({'error': '请提供MBTI类型'}), 400
        
        mbti = mbti.upper()
        
        # 验证MBTI类型格式
        if not re.match(r'^[IE][NS][TF][JP]$', mbti):
            return jsonify({'error': 'MBTI类型格式无效'}), 400
        
        # 获取描述
        descriptions = MBTI_DESCRIPTIONS.get(language, MBTI_DESCRIPTIONS['en'])
        description = descriptions.get(mbti, "No description available.")
        
        return jsonify({
            'description': description
        })
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/create_session', methods=['POST'])
def api_create_session():
    """
    创建新会话API
    
    响应：
        {
            "session_id": "unique_session_id"
        }
    """
    try:
        # 生成会话ID
        session_id = str(uuid.uuid4())
        
        # 初始化会话数据
        if 'sessions' not in app.config:
            app.config['sessions'] = {}
        
        app.config['sessions'][session_id] = {
            'created_at': datetime.datetime.now().isoformat(),
            'emotion_history': [],
            'responses': []
        }
        
        # 创建用户会话目录
        user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # 保存会话元数据
        meta_data = {
            'session_id': session_id,
            'created_at': datetime.datetime.now().isoformat(),
            'user_agent': request.headers.get('User-Agent', 'Unknown'),
            'ip_address': request.remote_addr
        }
        
        with open(os.path.join(user_dir, 'session_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
        
        return jsonify({
            'session_id': session_id
        })
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/save_response', methods=['POST'])
def api_save_response():
    """
    保存用户回答API
    
    请求体：
        {
            "session_id": "unique_session_id",
            "question": "问题文本",
            "response": "用户回答",
            "dimension": "E/I",
            "mbti_prediction": "MBTI预测结果",
            "emotion": "情绪预测结果"
        }
    
    响应：
        {
            "success": true
        }
    """
    try:
        data = request.json
        
        if not data or 'session_id' not in data:
            return jsonify({'error': '请提供会话ID和回答数据'}), 400
        
        session_id = data['session_id']
        
        # 获取会话数据
        if 'sessions' not in app.config:
            app.config['sessions'] = {}
        
        if session_id not in app.config['sessions']:
            app.config['sessions'][session_id] = {
                'created_at': datetime.datetime.now().isoformat(),
                'emotion_history': [],
                'responses': []
            }
        
        # 准备响应数据
        response_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'question': data.get('question', ''),
            'response': data.get('response', ''),
            'dimension': data.get('dimension', ''),
            'mbti_prediction': data.get('mbti_prediction', {}),
            'emotion': data.get('emotion', {})
        }
        
        # 保存到内存中的会话数据
        app.config['sessions'][session_id]['responses'].append(response_data)
        
        # 保存到文件
        user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
        os.makedirs(user_dir, exist_ok=True)
        
        responses_file = os.path.join(user_dir, 'responses.jsonl')
        with open(responses_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(response_data) + '\n')
        
        return jsonify({
            'success': True
        })
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/get_result', methods=['GET'])
def api_get_result():
    """
    获取会话结果API
    
    请求参数：
        session_id: 会话ID
        language: 可选，指定语言（en, zh），默认en
    
    响应：
        {
            "mbti_type": "INTJ",
            "description": "类型描述",
            "dimension_analysis": {...},
            "emotion_summary": {...}
        }
    """
    try:
        session_id = request.args.get('session_id')
        language = request.args.get('language', 'en')
        
        if not session_id:
            return jsonify({'error': '请提供会话ID'}), 400
        
        # 获取会话数据
        if 'sessions' not in app.config or session_id not in app.config['sessions']:
            # 尝试从文件中加载
            user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            if not os.path.exists(user_dir):
                return jsonify({'error': '找不到会话数据'}), 404
            
            # 加载响应数据
            responses = []
            responses_file = os.path.join(user_dir, 'responses.jsonl')
            if os.path.exists(responses_file):
                with open(responses_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            responses.append(json.loads(line))
            
            if not responses:
                return jsonify({'error': '会话中没有回答数据'}), 400
                
            # 创建会话数据
            app.config.setdefault('sessions', {})
            app.config['sessions'][session_id] = {
                'created_at': datetime.datetime.now().isoformat(),
                'emotion_history': [],
                'responses': responses
            }
        
        session_data = app.config['sessions'][session_id]
        responses = session_data.get('responses', [])
        
        if not responses:
            return jsonify({'error': '会话中没有回答数据'}), 400
        
        # 统计MBTI预测结果
        mbti_votes = {}
        dimension_scores = {
            'IE': 0,
            'NS': 0,
            'TF': 0,
            'JP': 0
        }
        
        # 情绪统计
        emotion_counts = {}
        
        for response in responses:
            mbti_prediction = response.get('mbti_prediction', {})
            emotion_data = response.get('emotion', {})
            
            # 统计MBTI
            if 'mbti_type' in mbti_prediction:
                mbti_type = mbti_prediction['mbti_type']
                confidence = mbti_prediction.get('confidence', 1.0)
                mbti_votes[mbti_type] = mbti_votes.get(mbti_type, 0) + confidence
            
            # 统计维度得分
            if 'dimension_analysis' in mbti_prediction:
                dimensions = mbti_prediction['dimension_analysis']
                for dim, data in dimensions.items():
                    if 'score' in data:
                        dimension_scores[dim] += data['score']
            
            # 统计情绪
            if 'predicted_emotion' in emotion_data:
                emotion = emotion_data['predicted_emotion']
                confidence = emotion_data.get('confidence', 1.0)
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + confidence
        
        # 确定最终MBTI类型
        if mbti_votes:
            final_mbti = max(mbti_votes.items(), key=lambda x: x[1])[0]
        else:
            # 基于维度得分构建MBTI
            final_mbti = ''
            if dimension_scores['IE'] < 0:
                final_mbti += 'I'
            else:
                final_mbti += 'E'
                
            if dimension_scores['NS'] > 0:
                final_mbti += 'N'
            else:
                final_mbti += 'S'
                
            if dimension_scores['TF'] < 0:
                final_mbti += 'T'
            else:
                final_mbti += 'F'
                
            if dimension_scores['JP'] < 0:
                final_mbti += 'J'
            else:
                final_mbti += 'P'
        
        # 获取MBTI描述
        descriptions = MBTI_DESCRIPTIONS.get(language, MBTI_DESCRIPTIONS['en'])
        description = descriptions.get(final_mbti, "No description available.")
        
        # 确定主要情绪
        main_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        # 准备维度分析结果
        dimension_analysis = {
            'IE': {
                'score': dimension_scores['IE'],
                'preference': 'I' if dimension_scores['IE'] < 0 else 'E'
            },
            'NS': {
                'score': dimension_scores['NS'],
                'preference': 'N' if dimension_scores['NS'] > 0 else 'S'
            },
            'TF': {
                'score': dimension_scores['TF'],
                'preference': 'T' if dimension_scores['TF'] < 0 else 'F'
            },
            'JP': {
                'score': dimension_scores['JP'],
                'preference': 'J' if dimension_scores['JP'] < 0 else 'P'
            }
        }
        
        # 返回结果
        result = {
            'mbti_type': final_mbti,
            'description': description,
            'dimension_analysis': dimension_analysis,
            'emotion_summary': {
                'main_emotion': main_emotion,
                'emotion_counts': emotion_counts
            }
        }
        
        # 保存最终结果
        try:
            user_dir = os.path.join(DATA_COLLECTION_DIR, session_id)
            os.makedirs(user_dir, exist_ok=True)
            
            result_file = os.path.join(user_dir, 'final_result.json')
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
        except Exception as e:
            logger.error(f"保存最终结果时出错: {e}")
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/submit_actual_mbti', methods=['POST'])
def api_submit_actual_mbti():
    """
    提交用户真实MBTI类型API
    
    请求体：
        {
            "session_id": "unique_session_id",
            "actual_mbti": "INTJ",
            "confidence_level": 5  // 可选，1-5的置信度级别
        }
    
    响应：
        {
            "success": true,
            "message": "真实MBTI类型已保存"
        }
    """
    try:
        data = request.json
        
        if not data or 'session_id' not in data or 'actual_mbti' not in data:
            return jsonify({'error': '请提供会话ID和真实MBTI类型'}), 400
        
        session_id = data['session_id']
        actual_mbti = data['actual_mbti']
        confidence_level = data.get('confidence_level')
        
        result = save_user_actual_mbti(session_id, actual_mbti, confidence_level)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/export_data', methods=['GET'])
def api_export_data():
    """
    导出会话数据API
    
    请求参数：
        session_id: 会话ID
        format: 可选，指定格式（json, csv），默认json
    
    响应：
        {
            "success": true,
            "format": "json",
            "file": "path/to/export.json"
        }
    """
    try:
        session_id = request.args.get('session_id')
        format = request.args.get('format', 'json')
        
        if not session_id:
            return jsonify({'error': '请提供会话ID'}), 400
        
        result = export_session_data(session_id, format)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

@app.route('/api/generate_dataset', methods=['GET'])
def api_generate_dataset():
    """
    生成研究数据集API
    
    响应：
        {
            "success": true,
            "files": ["path/to/file1.csv", ...],
            "sessions_count": 10,
            "emotions_count": 100,
            "texts_count": 50,
            "responses_count": 40
        }
    """
    try:
        # 身份验证检查（简单版本）
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != 'YOUR_SECRET_API_KEY':  # 替换为实际的API密钥
            return jsonify({'error': '无权访问此API'}), 403
        
        result = generate_research_dataset()
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({'error': f'处理请求时出错: {str(e)}'}), 500

# 启动服务器
if __name__ == '__main__':
    # 加载模型
    models = load_models()
    app.config['models'] = models
    
    # 初始化会话存储
    app.config['sessions'] = {}
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)