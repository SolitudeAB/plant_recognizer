# services/llm.py
from openai import OpenAI
import config

def ask_deepseek_stream(api_key, visual_result, location, season):
    """调用 DeepSeek 获取流式响应"""
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": config.get_user_prompt(visual_result, location, season)},
            ],
            temperature=1.0,
            stream=True
        )
        return response
    except Exception as e:
        return f"❌ API 调用失败: {str(e)}"