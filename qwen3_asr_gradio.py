import os
import sys
import asyncio

# ==========================================
# Windows 环境修复 (必须在其他导入前执行)
# ==========================================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 设置默认编码为 UTF-8
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    pass
os.environ['PYTHONIOENCODING'] = 'utf-8'

# ==========================================
# 导入模块
# ==========================================
import argparse
import gradio as gr
from loguru import logger
from qwen3_asr_handler import (
    Qwen3ASRHandler,
    scan_local_models,
    GLOBAL_VRAM_CONFIG,
    auto_detect_precision,
    auto_detect_flash_attention,
    get_optimal_chunk_size
)

# ==========================================
# 全局 Handler 实例
# ==========================================
handler = None

def initialize_service(model_dir, model_name, device, precision, quantization, compile_model, flash_attention):
    """Initialize the ASR service"""
    global handler
    if quantization != "none" and compile_model:
        logger.warning("Quantization is enabled. Disabling torch.compile to improve stability and load time.")
        compile_model = False
    
    handler = Qwen3ASRHandler(model_dir)
    status, success = handler.initialize(
        model_name=model_name,
        device=device,
        precision=precision,
        quantization=quantization,
        compile_model=compile_model,
        flash_attention=flash_attention
    )
    return status

def transcribe_audio(audio_file, language, chunk_size, overlap):
    """Transcribe uploaded audio file"""
    global handler
    if handler is None or handler.model is None:
        return "", "❌ Service not initialized. Please initialize first."
    if audio_file is None:
        return "", "❌ No audio file uploaded."
    try:
        import soundfile as sf
        waveform, sr = sf.read(audio_file)
        
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        
        text, timestamps = handler.transcribe(
            audio_waveform=waveform,
            sample_rate=sr,
            language=language,
            chunk_size=chunk_size,
            overlap=overlap
        )
        
        return text, timestamps if timestamps else "Transcription complete."
    except Exception as e:
        logger.exception("Transcription failed")
        return "", f"❌ Error: {str(e)}"

def create_ui():
    """Create Gradio Interface"""
    with gr.Blocks(title="Qwen3-ASR") as demo:
        gr.Markdown("# 🎤 Qwen3-ASR 语音识别")
        gr.Markdown(f"### 🖥️ GPU: {GLOBAL_VRAM_CONFIG.tier} ({GLOBAL_VRAM_CONFIG.gpu_memory_gb:.1f} GB)")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ 配置")
                
                # 模型目录
                model_dir_input = gr.Textbox(
                    label="模型目录",
                    value="./models",
                    info="存放模型文件的文件夹路径"
                )
                
                # 刷新模型列表
                refresh_btn = gr.Button("🔄 刷新模型")
                model_list = scan_local_models(model_dir_input.value)
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    label="选择模型",
                    value=model_list[0] if model_list else None
                )
                
                # 设备配置
                device_dropdown = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="运行设备"
                )
                
                precision_dropdown = gr.Dropdown(
                    choices=["auto", "bf16", "fp16", "fp32"],
                    value="auto",
                    label="计算精度"
                )
                
                quant_dropdown = gr.Dropdown(
                    choices=["none", "int8_weight_only", "fp8_weight_only"],
                    value="int8_weight_only",
                    label="量化 (TorchAO)"
                )
                
                compile_check = gr.Checkbox(
                    label="启用 torch.compile (量化时不建议启用)", 
                    value=False
                )
                flash_check = gr.Checkbox(
                    label="启用 Flash Attention", 
                    value=auto_detect_flash_attention()
                )
                
                init_btn = gr.Button("🚀 初始化模型", variant="primary")
                init_status = gr.Textbox(label="状态", interactive=False)
            
            with gr.Column():
                gr.Markdown("### 🎵 转录")
                audio_input = gr.Audio(type="filepath", label="上传音频")
                lang_dropdown = gr.Dropdown(
                    choices=["auto", "Chinese", "English", "Japanese", "Korean"],
                    value="auto",
                    label="语言"
                )
                
                recommended_chunk = get_optimal_chunk_size()
                chunk_slider = gr.Slider(
                    0, 300, 
                    value=recommended_chunk, 
                    step=5, 
                    label=f"分块大小 (秒，0=自动，推荐：{recommended_chunk}s)"
                )
                overlap_slider = gr.Slider(0, 10, value=2, step=1, label="重叠 (秒)")
                
                transcribe_btn = gr.Button("📝 开始转录", variant="primary")
                text_output = gr.Textbox(label="转录结果", lines=10)
                timestamp_output = gr.Textbox(label="信息", lines=5)
        
        # ==========================================
        # 事件绑定
        # ==========================================
        def refresh_models(dir):
            models = scan_local_models(dir)
            return gr.Dropdown(choices=models, value=models[0] if models else None)
        
        refresh_btn.click(fn=refresh_models, inputs=[model_dir_input], outputs=[model_dropdown])
        
        init_btn.click(
            fn=initialize_service,
            inputs=[model_dir_input, model_dropdown, device_dropdown, precision_dropdown, quant_dropdown, compile_check, flash_check],
            outputs=[init_status]
        )
        
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, lang_dropdown, chunk_slider, overlap_slider],
            outputs=[text_output, timestamp_output]
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Qwen3-ASR Gradio App")
    parser.add_argument("--port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="默认模型目录"
    )
    args = parser.parse_args()
    
    logger.info(f"GPU: {GLOBAL_VRAM_CONFIG.tier} ({GLOBAL_VRAM_CONFIG.gpu_memory_gb:.1f} GB)")
    logger.info(f"默认精度：{auto_detect_precision()}")
    logger.info(f"Flash Attention: {auto_detect_flash_attention()}")
    logger.info(f"推荐分块大小：{get_optimal_chunk_size()}s")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=args.port, 
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()