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
from openai import OpenAI  # 用于调用 DeepSeek

# ==============================================================================
# 🛠️ 1. 配置与模型路径
# ==============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1081
WEIGHTS_PATH = 'data.pkl'
SPECIES_NAME_JSON = 'plantnet300K_species_id_2_name.json'
CLASS_TO_ID_JSON = 'class_idx_to_species_id.json'

# ==============================================================================
# 🧠 2. 核心逻辑：加载本地 ResNet 模型
# ==============================================================================
@st.cache_resource
def load_resources():
    """加载本地 PyTorch 模型和类别映射"""
    if not os.path.exists(SPECIES_NAME_JSON) or not os.path.exists(CLASS_TO_ID_JSON):
        st.error("❌ 缺少 JSON 配置文件。")
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
        st.error(f"❌ 缺少权重文件: {WEIGHTS_PATH}")
        return None, None

    base_dir = os.path.dirname(os.path.abspath(WEIGHTS_PATH))
    data_dir = os.path.join(base_dir, 'data')

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    # 深度兼容 Unpickler (解决旧版数据格式)
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
                if 'LongStorage' in typename_str: storage_cls = torch.LongStorage
                elif 'IntStorage' in typename_str: storage_cls = torch.IntStorage
                
                data_file_path = os.path.join(data_dir, str(key))
                # 如果 data 文件夹缺失，让它报错以便用户发现
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

    model = model.to(device)
    model.eval()
    return model, class_names

def predict_local(image, model, class_names):
    """本地模型推理，只返回拉丁名和置信度"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_p, top_idx = probs.topk(1)
    return class_names[top_idx.item()], top_p.item() * 100

# ==============================================================================
# 🤖 3. 云端逻辑：调用 DeepSeek API
# ==============================================================================
def ask_deepseek(api_key, latin_name, location, season):
    """调用 DeepSeek 获取详细科普"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # 构建 Prompt
    system_prompt = "你是一位博学的植物学家和自然教育家。请用生动、准确的中文介绍植物。"
    user_prompt = f"""
    用户上传了一张植物照片，经识别其拉丁学名为："{latin_name}"。
    用户发现它的环境信息：地点="{location}"，季节="{season}"。

    请生成一份包含以下内容的科普报告（使用Markdown格式）：
    1. **中文正名**：给出最通用的中文名称。
    2. **植物简介**：简要介绍它的科属、原产地和主要形态特征。
    3. **生长习性**：它喜欢什么样的土壤、光照和水分？
    4. **环境互动**：结合用户提供的地点（{location}）和季节（{season}），分析为什么它会出现在这里？有什么观察建议？
    5. **趣味冷知识**：关于这种植物的一个有趣事实或药用/经济价值。

    请保持语气亲切、专业，字数控制在400字以内。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.3, # 稍微高一点，让回答更生动
            stream=True      # 开启流式输出
        )
        return response
    except Exception as e:
        return f"❌ API 调用失败: {str(e)}"

# ==============================================================================
# 🎨 4. 前端界面
# ==============================================================================
st.set_page_config(page_title="植物识别 Pro (DeepSeek版)", page_icon="🌿", layout="wide")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
    .report-box { border: 2px solid #f0f2f6; padding: 20px; border-radius: 10px; background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# --- 侧边栏配置 ---
with st.sidebar:
    st.title("⚙️ 设置")
    st.markdown("本系统采用 **端云结合** 架构：")
    st.info("🖥️ **本地 ResNet**：毫秒级识别植物身份")
    st.info("☁️ **DeepSeek AI**：生成深度科普介绍")
    
    api_key = st.text_input("🔑 输入 DeepSeek API Key", type="password", placeholder="sk-...")
    if not api_key:
        st.warning("⚠️ 请先输入 API Key 才能获取详细介绍")
        st.markdown("[👉 点击申请 DeepSeek Key](https://platform.deepseek.com/)")

# --- 主界面 ---
st.title("🌿 AI 植物百科全书")
st.caption("Powered by PyTorch & DeepSeek-V3")

# 加载本地模型
with st.spinner('正在加载本地视觉模型...'):
    model, class_names = load_resources()

if not model:
    st.stop()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 拍摄/上传")
    uploaded_file = st.file_uploader("上传图片", type=["jpg", "png", "jpeg"])
    
    st.subheader("2. 环境信息 (AI将结合此信息)")
    location = st.text_input("📍 发现地点", value="")
    season = st.selectbox("🗓️ 当前季节", ["春季", "夏季", "秋季", "冬季"])
    
    identify_btn = st.button("🚀 开始识别 & 咨询 AI", type="primary")

with col2:
    if uploaded_file and identify_btn:
        # 1. 图片显示
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="待识别图像", use_container_width=True)
        
        # 2. 本地推理 (极快)
        start_time = time.time()
        latin_name, confidence = predict_local(image, model, class_names)
        local_time = time.time() - start_time
        
        st.success(f"视觉识别完成！(耗时 {local_time:.3f}s)")
        
        # 显示初步结果
        c1, c2 = st.columns(2)
        c1.metric("识别学名", latin_name)
        c2.metric("视觉置信度", f"{confidence:.1f}%")
        
        st.markdown("---")
        st.subheader("🤖 DeepSeek 科普报告")

        # 3. 调用 DeepSeek (如果填了 Key)
        if api_key:
            # 创建一个占位符用于流式输出
            report_placeholder = st.empty()
            full_response = ""
            
            # 调用流式 API
            stream = ask_deepseek(api_key, latin_name, location, season)
            
            if isinstance(stream, str): # 如果返回的是错误信息字符串
                st.error(stream)
            else:
                # 实时打印字符
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        report_placeholder.markdown(full_response + "▌") # 加个光标特效
                
                report_placeholder.markdown(full_response) # 最后显示完整内容
        else:
            st.warning("⚠️ 未检测到 API Key，无法生成中文介绍。只能显示拉丁学名。")
            st.markdown(f"**Google 翻译链接：** [点击翻译 {latin_name}](https://translate.google.com/?sl=la&tl=zh-CN&text={latin_name}&op=translate)")

    elif not uploaded_file:
        st.info("👈 请在左侧上传图片。")