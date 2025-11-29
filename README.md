# 🌿 PlantAI Pro (植物识别与科普助手)

> **基于 ResNet-18 视觉模型 + DeepSeek-V3 大语言模型的智能植物百科系统**

本项目是一个端云结合的 AI 应用。它利用本地部署的深度学习模型（ResNet-18）实现毫秒级的植物身份识别，并结合 DeepSeek 大模型根据地理位置和季节提供生动的科普介绍。

**核心模型基于权威的 [Pl@ntNet-300K](https://zenodo.org/records/5645731) 数据集训练，涵盖 1081 种常见植物。**

------

## 🌟 核心特性 (Features)

1. **🚀 极速启动 & 安全配置**:
   - 采用独特的 `start.py` 启动器。
   - **API Key 不在网页明文传输**，首次运行时在终端安全输入并本地加密存储，后续自动读取。
2. **🧠 双引擎架构**:
   - **本地视觉 (Vision)**: ResNet-18 负责“看”，无需联网即可快速识别植物 ID。
   - **云端大脑 (LLM)**: DeepSeek-V3 负责“说”，生成结构优美的 Markdown 科普报告。
3. **🌍 环境感知 (Context-Aware)**:
   - AI 不仅仅是背书，它会结合你输入的**地点**（如北京公园）和**季节**（如冬季），分析植物为何出现在此处，并提供观察建议。
4. **🎨 现代 UI 设计**:
   - 基于 Streamlit 的响应式界面，包含卡片式结果展示、置信度动态标签和 Markdown 完美渲染。

------

## 📂 项目结构

Plaintext

```
.
├── start.py                            # [入口] 程序的唯一启动入口 (负责Key管理与服务启动)
├── app.py                              # [核心] Streamlit Web 界面与业务逻辑
├── api_key_config.txt                  # [配置] 自动生成的 Key 配置文件 (首次运行后生成)
├── data.pkl                            # [模型] ResNet-18 预训练权重
├── plantnet300K_species_id_2_name.json # [数据] 种类ID映射表
├── class_idx_to_species_id.json        # [数据] 索引映射表
├── requirements.txt                    # [环境] 依赖库列表
└── README.md                           # 说明文档
```

------


## 🛠️ 环境准备

建议使用 Conda 创建独立的虚拟环境。

### 1. 环境准备

```
# 创建虚拟环境 (Python 3.9)
conda create -n plant_ai python=3.9 # 请确保你的 Python 版本 >= 3.8。
conda activate plant_ai
```

### 2. 安装依赖

```
pip install -r requirements.txt
```

**注意**: 如果您使用 NVIDIA 显卡，建议前往 [PyTorch 官网](https://pytorch.org/get-started/locally/) 获取适配 CUDA 版本的安装命令，以获得更快的推理速度。

------

## 🚀 启动与使用 (Operation Flow)

**⚠️ 请务必通过 `start.py` 启动，不要直接运行 `streamlit run`。**

### 第一步：启动程序

在终端（Terminal/Console）中运行：

```
python start.py
```

### 第二步：配置 API Key (仅首次)

1. 如果你是**第一次运行**，程序会检测到没有配置文件。

2. **观察终端黑框**，会出现提示：

   ```
   🌿 PlantAI Pro 启动向导
   ⚠️  未检测到配置。
   👉 请输入 DeepSeek API Key (输入后回车):
   Input Key > [在此处粘贴你的Key]
   ```
   
3. 输入 Key 并回车后，系统会自动保存配置，并启动网页服务器。

### 第三步：使用 Web 界面

1. 浏览器会自动打开 `http://localhost:8501`。
2. **上传图片**: 点击左侧侧边栏上传植物照片。
3. **填写环境**: 输入发现地点（如“校园”）和季节。
4. **查看结果**:
   - 系统会立即显示 **识别学名** 和 **置信度**。
   - 点击 **“✨ 生成科普报告”**，AI 将流式输出详细介绍。

------

## ❓ 常见问题 (FAQ)

**Q1: 如何重置或更换 API Key？**

- **方法 A**: 在项目根目录下找到 `api_key_config.txt` 文件，将其删除。下次运行 `python start.py` 时会重新询问。
- **方法 B**: 在 Web 界面左侧侧边栏，点击“清除 Key”按钮（如有），然后重启程序。

**Q2: 为什么终端卡住了，网页没打开？**

- 请检查终端（黑框）是否有文字提示 `Input Key >`。程序在等待你输入 Key，输入并回车后才会继续。

**Q3: 报错 `urllib.error.URLError: SSL...`**

- 这是由于网络原因导致 PyTorch 无法下载预训练模型。
- **解决**: 请确保 `data.pkl` 权重文件已经完整下载并放置在项目根目录下。

------

## 🖼️ 致谢与数据

- **数据集**: [Pl@ntNet-300K](https://zenodo.org/records/5645731)
- **大模型**: DeepSeek-V3 / R1
- **框架**: PyTorch & Streamlit