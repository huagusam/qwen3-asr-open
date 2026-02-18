[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
\# 🎤 Qwen3-ASR gradio Open Source


基于 Qwen3 的开源语音识别系统，支持 VAD 智能分块、多语言识别、量化加速等功能。



\## ✨ 特性



\- 🔊 \*\*VAD 智能分块\*\* - 基于语音活动检测，在静音处切分，避免切断句子

\- 🌍 \*\*多语言支持\*\* - 中文、英文、日语、韩语自动检测

\- ⚡ \*\*量化加速\*\* - 支持 int8/fp8 量化，降低显存占用

\- 🚀 \*\*Flash Attention\*\* - 支持 Flash Attention 2 加速

\- 💾 \*\*模型缓存\*\* - LRU 缓存机制，快速切换模型





\## 📦 安装



\### 1. 克隆仓库



```bash

git clone https://github.com/huagusam/qwen3-asr-open.git

cd qwen3-asr-open

start_asr.bat
F:\ACE-Step-1.5\python_embeded\python.exe 🟥切换为你的python环境

把模型Qwen3-ASR-1.7B整个目录 放到 ./models 文件夹

# Qwen3-ASR 快速参考

> ⚠️ **警告**：本项目为个人使用代码，**硬编码、路径写死、无异常处理**。
> 生产环境请自行改造，或让 AI 帮你重构。

## 核心代码位置

| 功能 | 文件 | 关键行数 |
|------|------|---------|
| VAD 智能切分 | `qwen3_asr_handler.py` | 60-200 |
| 显存管理 | `qwen3_asr_handler.py` | 250-300 |
| 去重后处理 | `qwen3_asr_handler.py` | 400-500 |
| Gradio UI | `qwen3_asr_gradio.py` | 30-80 |

模型下载
1.
# 安装 huggingface-cli
pip install huggingface-hub

# 下载模型（以 1.7B 为例）
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./Qwen3-ASR-1.7B

2.
pip install modelscope

# Python 下载
from modelscope import snapshot_download
model_dir = snapshot_download("qwen/Qwen3-ASR-1.7B", cache_dir="./")

链接：https://modelscope.cn/models/qwen/Qwen3-ASR-1.7B

## 必改配置  模型路径 
```python
# qwen3_asr_gradio.py 第 101 行
        value="./models",
改为 value="G:/Comfy/ComfyUI/models/diffusion_models/Qwen3-ASR",  # ← 改这里为模型路径


