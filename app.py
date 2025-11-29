import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
import pickle
import io
import time
from openai import OpenAI

# =========================================================
# 🔑 0. 读取 API Key
# =========================================================
KEY_FILE = "api_key_config.txt"
API_KEY = None

if os.path.exists(KEY_FILE):
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        API_KEY = f.read().strip()

# =========================================================
# 🛠️ 1. 设置与资源加载
# =========================================================
st.set_page_config(page_title="PlantAI Pro", page_icon="🌿", layout="wide")

# 🎨 优化后的 CSS (修复 Markdown 渲染样式)
st.markdown("""
<style>
    /* 隐藏顶部默认 Header */
    header {visibility: hidden;}

    /* 主标题 */
    .main-title { 
        font-size: 2.5rem; 
        color: #2E7D32; 
        text-align: center; 
        font-weight: 800; 
        margin-bottom: 20px; 
    }

    /* 结果卡片 */
    .result-card { 
        background: white; 
        padding: 25px; 
        border-radius: 12px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.08); 
        border-left: 6px solid #2E7D32; 
        margin-bottom: 20px; 
    }

    /* 识别结果文字 */
    .latin-name { 
        font-size: 1.8rem; 
        font-weight: bold; 
        color: #1b1b1b; 
        font-family: 'Times New Roman', serif; 
        font-style: italic; 
    }

    /* 侧边栏背景 */
    section[data-testid="stSidebar"] { background-color: #f8f9fa; }

    /* 📝 修复 Markdown 报告的样式 */
    .report-container h2 {
        color: #2E7D32;
        font-size: 1.5rem;
        border-bottom: 2px solid #E8F5E9;
        padding-bottom: 8px;
        margin-top: 20px;
    }
    .report-container h3 {
        color: #388E3C;
        font-size: 1.2rem;
        margin-top: 15px;
    }
    .report-container strong {
        color: #1b5e20;
    }
    .report-container ul {
        margin-left: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 路径配置
WEIGHTS_PATH = 'data.pkl'
SPECIES_NAME_JSON = 'plantnet300K_species_id_2_name.json'
CLASS_TO_ID_JSON = 'class_idx_to_species_id.json'
NUM_CLASSES = 1081
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_resources():
    if not os.path.exists(SPECIES_NAME_JSON) or not os.path.exists(CLASS_TO_ID_JSON):
        st.error("❌ 缺少 JSON 文件")
        return None, None

    with open(CLASS_TO_ID_JSON, 'r', encoding='utf-8') as f:
        class_to_id = json.load(f)
    with open(SPECIES_NAME_JSON, 'r', encoding='utf-8') as f:
        species_id_to_name = json.load(f)

    class_names = []
    for i in range(NUM_CLASSES):
        species_id = str(class_to_id[str(i)])
        class_names.append(species_id_to_name.get(species_id, f"Unknown {species_id}"))

    if not os.path.exists(WEIGHTS_PATH):
        st.error(f"❌ 缺少权重: {WEIGHTS_PATH}")
        return None, None

    base_dir = os.path.dirname(os.path.abspath(WEIGHTS_PATH))
    data_dir = os.path.join(base_dir, 'data')

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

        def persistent_load(self, saved_id):
            if isinstance(saved_id, tuple) and saved_id[0] == 'storage':
                typename, key, _, numel = saved_id[1], saved_id[2], saved_id[3], saved_id[4]
                typename_str = typename.__name__ if isinstance(typename, type) else str(typename)
                storage_cls = torch.FloatStorage
                if 'LongStorage' in typename_str:
                    storage_cls = torch.LongStorage
                elif 'IntStorage' in typename_str:
                    storage_cls = torch.IntStorage
                data_file_path = os.path.join(data_dir, str(key))
                return storage_cls.from_file(data_file_path, shared=False, size=numel)
            return saved_id

    try:
        with open(WEIGHTS_PATH, 'rb') as f:
            checkpoint = CustomUnpickler(f).load()
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"❌ 模型加载失败: {e}")
        return None, None

    model = model.to(DEVICE)
    model.eval()
    return model, class_names


def predict_local(image, model, class_names):
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_p, top_idx = probs.topk(1)
    return class_names[top_idx.item()], top_p.item() * 100


def ask_deepseek_stream(latin_name, location, season):
    if not API_KEY:
        yield "⚠️ API Key 缺失，请检查配置。"
        return

    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    # 🔥 优化 Prompt：强制 Markdown 格式，防止缩进导致渲染失败
    prompt = f"""
    你是一位植物科普专家。请生成关于"{latin_name}"的科普报告。
    观察信息：地点-{location}，季节-{season}。

    【格式要求】
    1. 必须使用标准的 Markdown 格式。
    2. 不要使用代码块。
    3. 标题前不要有空格缩进。

    【内容大纲】
    ## 中文正名与科属
    （这里介绍中文名、别名、科属）

    ## 形态特征
    （简要描述花、叶特征）

    ## 环境与习性
    （结合{location}和{season}分析）

    ## 趣味冷知识
    （一个有趣的知识点）
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=1.3
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"❌ API Error: {e}"


# =========================================================
# 🎨 2. 界面逻辑
# =========================================================
def main():
    st.markdown('<div class="main-title">🌿 AI 植物百科全书</div>', unsafe_allow_html=True)

    with st.sidebar:
        uploaded_file = st.file_uploader("📸 上传照片", type=["jpg", "png", "jpeg"])
        location = st.text_input("📍 地点", value="公园")
        season = st.selectbox("🗓️ 季节", ["春季", "夏季", "秋季", "冬季"])
        if API_KEY:
            st.success("✅ DeepSeek API 已连接")
            if st.button("重置 Key"):
                if os.path.exists(KEY_FILE): os.remove(KEY_FILE)
                st.rerun()  # 新版 streamlit 使用 rerun
        else:
            st.error("❌ API 未配置")

    if not uploaded_file: return

    with st.spinner("🧠 正在分析..."):
        model, class_names = load_resources()
        if model:
            image = Image.open(uploaded_file).convert('RGB')
            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.image(image, use_container_width=True)

            with col2:
                name, conf = predict_local(image, model, class_names)

                # 结果卡片
                st.markdown(f"""
                <div class="result-card">
                    <div style="color: #666; font-size: 0.9em;">识别结果</div>
                    <div class="latin-name">{name}</div>
                    <div style="margin-top: 5px; color: {'#2E7D32' if conf > 80 else '#F9A825'}">
                        置信度: {conf:.2f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button("✨ 生成科普报告", type="primary"):
                    st.markdown("---")
                    res_box = st.empty()
                    full_text = ""

                    # 使用 div 包裹以应用 CSS
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)

                    for chunk in ask_deepseek_stream(name, location, season):
                        full_text += chunk
                        # 实时渲染，增加 strip() 防止开头空格
                        res_box.markdown(full_text + " ▌", unsafe_allow_html=True)

                    # 最终渲染
                    res_box.markdown(full_text, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()