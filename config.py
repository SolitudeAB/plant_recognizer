# config.py
import os

# --- 模型配置 ---
MODEL_PATH = "./local_model"
ONLINE_MODEL_NAME = "google/vit-base-patch16-224"

# --- 页面配置 ---
PAGE_CONFIG = {
    "page_title": "自然之眼：AI 生物图鉴",
    "page_icon": "🌿",
    "layout": "wide"
}

# --- 提示词 (Prompt) ---
SYSTEM_PROMPT = "你是一位严格的生物学分类专家。你需要先判断物体属性，再决定是否生成科普内容。"

def get_user_prompt(visual_result, location, season):
    return f"""
    视觉AI识别到一个物体，英文标签为："{visual_result}"。
    环境信息：地点="{location}"，季节="{season}"。

    🔴 **第一步：生物审查**
    请判断该标签是否属于【生物】（动物、植物、昆虫、真菌等）。
    * **特殊通过规则**：如果标签是 'pot', 'vase', 'flowerpot' (花盆/花瓶)，请视为【生物】（默认为绿植盆栽）。
    * **严格拦截规则**：如果标签是 'car', 'screen', 'toy', 'furniture' 等非生命体，必须判定为【非生物】。

    🔴 **第二步：输出指令**
    1. 如果是【非生物】：**请只输出一个单词："NON_BIO_STOP"**。不要输出任何其他废话。
    2. 如果是【生物】：请生成一份Markdown格式的自然科普报告（包含中文名称、科属、习性、环境互动），字数400字以内，不要用代码块包裹。
    """