import streamlit as st

def load_css():
    """加载自定义 CSS"""
    st.markdown("""
    <style>
        .stButton>button { width: 100%; border-radius: 8px; font-weight: bold; }
        .report-box { border: 2px solid #f0f2f6; padding: 20px; border-radius: 10px; background-color: #ffffff; }
        .big-font { font-size: 24px !important; color: #2E7D32; font-weight: bold; }
        .stAlert { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

def clean_markdown(text):
    """清洗 DeepSeek 可能返回的 Markdown 代码块标记"""
    if not text: return ""
    return text.replace("```markdown", "").replace("```", "").strip()