\# 🎤 Qwen3-ASR Open Source



基于 Qwen3 的开源语音识别系统，支持 VAD 智能分块、多语言识别、量化加速等功能。



\## ✨ 特性



\- 🔊 \*\*VAD 智能分块\*\* - 基于语音活动检测，在静音处切分，避免切断句子

\- 🌍 \*\*多语言支持\*\* - 中文、英文、日语、韩语自动检测

\- ⚡ \*\*量化加速\*\* - 支持 int8/fp8 量化，降低显存占用

\- 🚀 \*\*Flash Attention\*\* - 支持 Flash Attention 2 加速

\- 💾 \*\*模型缓存\*\* - LRU 缓存机制，快速切换模型

\- 📥 \*\*一键下载\*\* - 支持从 HuggingFace 自动下载模型



\## 📦 安装



\### 1. 克隆仓库



```bash

git clone https://github.com/huagusam/qwen3-asr-open.git

cd qwen3-asr-open

start_asr.bat
F:\ACE-Step-1.5\python_embeded\python.exe 切换为你的python环境

