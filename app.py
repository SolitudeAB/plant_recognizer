# app.py
import time
import streamlit as st
from PIL import Image

# 引入我们拆分的模块
import config
from services import vision, llm
from utils import display

# 1. 初始化页面
st.set_page_config(**config.PAGE_CONFIG)
display.load_css()

# 2. 侧边栏配置
with st.sidebar:
    st.title("⚙️ 设置")
    st.info("✨ **生物专属模式**")
    st.caption("自动拦截非生物照片")
    api_key = st.text_input("🔑 DeepSeek API Key", type="password")

# 3. 主标题
st.title("🌿 自然之眼：AI 生物图鉴")
st.caption("Powered by ViT & DeepSeek-V3 | 仅限识别自然生物")

# 4. 加载模型 (调用 vision 服务)
classifier = vision.load_vision_model()
if not classifier: st.stop()

# 5. 界面布局
col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("1. 📸 采集样本")
    uploaded_file = st.file_uploader("请上传动植物照片...", type=["jpg", "png", "jpeg", "webp"])
    
    st.subheader("2. 🌍 记录环境")
    location = st.text_input("📍 发现地点", value="野外/公园")
    season = st.selectbox("🗓️ 当前季节", ["春季", "夏季", "秋季", "冬季"])
    
    identify_btn = st.button("🔍 鉴定物种", type="primary")

with col2:
    if uploaded_file and identify_btn:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="样本图像", use_container_width=True)
        
        # --- 视觉识别阶段 ---
        start_time = time.time()
        label_en, score = vision.predict_image(classifier, image)
        
        st.success(f"👁️ 视觉特征提取完成 ({time.time() - start_time:.2f}s)")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown(f'<p class="big-font">{label_en.title()}</p>', unsafe_allow_html=True)
            st.caption("视觉底层标签")
        with c2:
            st.metric("视觉置信度", f"{score*100:.1f}%")
        
        st.markdown("---")
        st.subheader("📖 鉴定报告")

        # --- DeepSeek 分析与拦截阶段 ---
        if api_key:
            report_placeholder = st.empty()
            full_response = ""
            is_non_bio = False 
            
            with st.spinner("🧠 正在进行生物学判定..."):
                stream = llm.ask_deepseek_stream(api_key, label_en, location, season)
            
            if isinstance(stream, str):
                st.error(stream) # API 调用报错
            else:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        
                        # 拦截检测
                        if "NON_BIO_STOP" in full_response:
                            is_non_bio = True
                            break 
                        
                        # 实时渲染 (使用 display 工具清洗文本)
                        report_placeholder.markdown(display.clean_markdown(full_response) + "▌", unsafe_allow_html=True)
                
                # 最终结果展示
                if is_non_bio:
                    report_placeholder.empty()
                    st.error(f"🚫 **识别失败：目标不是生物**")
                    st.warning(f"AI 识别出图像主体为：**{label_en}** (非生物)。\n\n👉 **请放入生物图片**（动物、植物、昆虫等）再次尝试。")
                else:
                    report_placeholder.markdown(display.clean_markdown(full_response), unsafe_allow_html=True)
        else:
            st.warning("⚠️ 请输入 DeepSeek API Key 以生成鉴定报告。")

    elif not uploaded_file:
        st.info("👈 请在左侧上传照片以开始。")