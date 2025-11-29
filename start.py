import os
import subprocess
import sys

# 配置文件名
KEY_FILE = "api_key_config.txt"


def main():
    print("\n" + "=" * 50)
    print("🌿 PlantAI Pro 启动向导")
    print("=" * 50)

    # 1. 检查是否存在 Key
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            key = f.read().strip()

        if key:
            print(f"✅ 检测到本地 API Key，准备启动...")
        else:
            get_key()
    else:
        # 文件不存在，强制输入
        get_key()

    print("🚀 正在启动网页服务器...\n")
    print("-" * 50)

    # 2. 只有拿到 Key 之后，才用代码去调用 Streamlit
    # 这一步相当于帮你在命令行敲了 "streamlit run app.py"
    try:
        subprocess.run(["streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        pass


def get_key():
    print("⚠️  未检测到配置。")
    print("👉 请输入 DeepSeek API Key (输入后回车):")

    while True:
        # 这里的 input 是纯 Python 的，绝对会在网页启动前执行
        key = input("Input Key > ").strip()
        if key:
            with open(KEY_FILE, "w", encoding="utf-8") as f:
                f.write(key)
            print("✅ Key 已保存！")
            break
        else:
            print("❌ 不能为空")


if __name__ == "__main__":
    main()