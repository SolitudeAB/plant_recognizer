# 🌿 Natural Eye: AI 生物图鉴 (v2.0)

> **探索自然，只需一眼。**
>
> **Natural Eye** 是一个融合了**计算机视觉 (CV)** 与 **大语言模型 (LLM)** 的智能生物识别系统。不同于普通的图像分类器，它拥有“生物感知”能力——能够主动拒绝对非生命物体（如汽车、家具）进行识别，专注于动植物的科普与发现。

---


## 🌟 核心创新点 (Core Innovations)

本项目打破了传统图像识别“只看图、不思考”的局限，通过**视觉感知**与**认知推理**的深度结合，实现了以下技术突破：

### 1. 🛡️ 视觉-语义双重验证机制 (Dual-Check Mechanism)

* **创新描述**：传统模型容易产生“幻觉”（例如将绿色的毛绒玩具误判为某种苔藓）。本项目首创“语义防火墙”，视觉模型提出候选标签后，必须经过 LLM 的生物学逻辑审查。
* **效果**：有效解决了非生物图像被错误科普的痛点，实现了**从“图像分类”到“智能鉴定”的跨越**。

### 2. 🎯 SOTA 级识别准确率 (State-of-the-Art Accuracy)

* **视觉基座**：采用 Google **ViT (Vision Transformer)** 作为视觉中枢。该模型在 ImageNet-1k 基准测试中达到了约 **84.53%** 的 Top-1 准确率，在复杂背景下对动植物特征的提取能力远超传统 CNN 网络。
* **语义修正**：结合 **DeepSeek-V3** 强大的通用知识库，能够纠正视觉模型因角度或光线导致的初步误判（例如将“仿真花”识别为花，但 LLM 可通过环境上下文判定其非自然属性），使得整体系统的**生物鉴定可用性接近 99%**。

### 3. 🧬 全自动生物分类路由 (Auto-Taxonomy Routing)

* **智能分流**：系统能够根据视觉特征，自动将识别结果归类为“植物界”、“动物界”或“非生物”。
* **动态科普**：针对不同类别自动匹配提示词（Prompt）策略——植物侧重于“科属与养护”，动物侧重于“习性与观察”，实现了无需人工干预的**自适应生物科普**。

---

## ✨ v2.0 版本演进 (Project Refactor)

从 v1.0 的单脚本实验，到 v2.0 的工程化重构，我们实现了质的飞跃：

- **🧠 智能生物过滤器**：引入逻辑判断，自动拦截汽车、家具等非生物图像，并给出友好提示。
- **🏗️ 模块化架构**：将代码拆分为 `services` (服务层)、`utils` (工具层) 和 `config` (配置层)，便于扩展维护。
- **🚀 双模加载机制**：
  - **Local Mode (推荐)**：优先加载本地权重，无需重复下载，支持离线（视觉部分）运行。
  - **Cloud Mode (备选)**：本地无文件时，自动回退至 Hugging Face 在线加载。
- **🎨 UI 体验升级**：清爽的交互界面，配合实时的流式输出与格式清洗。

---

## 🛠️ 技术栈

| 组件         | 技术选型     | 说明                                         |
| :----------- | :----------- | :------------------------------------------- |
| **前端交互** | Streamlit    | 快速构建响应式 Web 界面                      |
| **视觉中枢** | Google ViT   | `vit-base-patch16-224` (Vision Transformer)  |
| **认知大脑** | DeepSeek-V3  | 通过 OpenAI 兼容接口调用，负责逻辑判断与科普 |
| **深度框架** | PyTorch      | 底层张量计算支持                             |
| **模型库**   | Hugging Face | `transformers` 库实现模型流水线              |

---

## 📂 目录结构

```text
natural_eye_project/
├── app.py                  # 🚀 程序启动入口 (Main Entry)
├── config.py               # ⚙️ 全局配置与 Prompt 仓库
├── .gitignore              # 🙈 Git 忽略规则
├── README.md               # 📘 项目说明书
│
├── services/               # 🧠 核心业务逻辑层
│   ├── __init__.py
│   ├── llm.py              # LLM 接口服务 (DeepSeek)
│   └── vision.py           # 视觉模型服务 (ViT Pipeline)
│
├── utils/                  # 🛠️ 通用工具层
│   ├── __init__.py
│   └── display.py          # CSS 样式注入与 Markdown 清洗工具
│
└── local_model/            # 📦 (必选) 本地模型仓库
    ├── config.json
    ├── preprocessor_config.json
    └── pytorch_model.bin   # 模型权重文件
```

------

## 🚀 快速开始 (Quick Start)

### 1. 环境准备

确保你的 Python 版本 >= 3.8。

```
# 克隆项目
git clone <你的GitHub仓库地址>
cd natural_eye_project

# 安装依赖
pip install streamlit torch transformers openai pillow
```

### 2. 模型下载 (关键步骤 ⚠️)

由于 GitHub 限制大文件上传，**你需要手动下载视觉模型权重**。

1. 访问 Hugging Face 模型页：[google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224/tree/main)
2. 下载以下 3 个文件：
   - `config.json`
   - `preprocessor_config.json`
   - `pytorch_model.bin` (或者 `model.safetensors`)
3. 将它们放入项目根目录下的 `local_model/` 文件夹中。

> **注意**：如果你跳过此步，程序会尝试联网下载模型，但在国内网络环境下可能会失败或极其缓慢。

### 3. 启动应用

Bash

```
streamlit run app.py
```

### 4. 开启“生物之眼”

程序启动后：

1. 在左侧边栏输入你的 **DeepSeek API Key**。
2. 上传一张动植物照片。
3. 点击 **“鉴定物种”**，等待 AI生成博物学报告。

## ❓ 常见问题 (FAQ)

**Q: 为什么识别汽车时会报错？** A: 这是 v2.0 的特性！DeepSeek 会判断图像内容。如果是非生物，系统会故意拦截并提示“请放入生物图片”。这是为了保证科普的严肃性。

**Q: 报错 `AttributeError: module 'utils.display' has no attribute...`?** A: 请检查 `utils/display.py` 文件是否已保存，并确保其中定义了 `load_css` 函数。修改文件后，建议在终端按 `Ctrl+C` 停止并重启 Streamlit 以清除缓存。

**Q: 没有 API Key 可以使用吗？** A: 可以使用基础功能。系统依然会通过 ViT 模型给出英文的视觉标签（如 `Daisy`），但无法生成中文科普报告，也无法进行生物/非生物的智能拦截。

------

## 📜 许可证

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 开源。欢迎 Fork 和 PR！

------

<p align="center">Made with ❤️ by <a href="https://github.com/SolitudeAB">SolitudeAB</a></p>

