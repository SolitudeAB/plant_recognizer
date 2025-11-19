# services/vision.py
import os
import streamlit as st
from transformers import pipeline
import config # 导入配置文件

@st.cache_resource
def load_vision_model():
    """加载视觉模型 (带缓存)"""
    # 检查本地模型
    if os.path.exists(config.MODEL_PATH) and os.path.isdir(config.MODEL_PATH):
        load_source = config.MODEL_PATH
    else:
        load_source = config.ONLINE_MODEL_NAME
        st.warning(f"⚠️ 未检测到本地模型，正在下载云端模型: {config.ONLINE_MODEL_NAME}")

    try:
        # 创建分类器
        classifier = pipeline("image-classification", model=load_source)
        return classifier
    except Exception as e:
        st.error(f"❌ 视觉模型加载失败: {e}")
        return None

def predict_image(classifier, image):
    """进行图像推理"""
    results = classifier(image)
    top_result = results[0]
    return top_result['label'], top_result['score']