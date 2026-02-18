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
    scan_comfy_models,
    scan_local_models,
    GLOBAL_VRAM_CONFIG,
    auto_detect_precision,
    auto_detect_flash_attention,
    get_optimal_chunk_size,
    download_model,
    DEFAULT_MODEL_DIR
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

def download_and_init(model_dir, model_id, device, precision, quantization, compile_model, flash_attention):
    """Download model and initialize"""
    try:
        logger.info(f"Downloading model: {model_id}")
        model_path = download_model(model_id, model_dir)
        if model_path:
            logger.info(f"Model downloaded to: {model_path}")
            return initialize_service(model_dir, os.path.basename(model_path), device, precision, quantization, compile_model, flash_attention)
        else:
            return "❌ Download failed"
    except Exception as e:
        logger.exception("Download failed")
        return f"❌ Error: {str(e)}"

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
    with gr.Blocks(title="Qwen3-ASR Open Source") as demo:
        gr.Markdown("# 🎤 Qwen3-ASR Open Source Transcriber")
        gr.Markdown(f"### 🖥️ Detected GPU: {GLOBAL_VRAM_CONFIG.tier} ({GLOBAL_VRAM_CONFIG.gpu_memory_gb:.1f} GB)")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### ⚙️ Configuration")
                
                # 模型目录
                model_dir_input = gr.Textbox(
                    label="Model Directory",
                    value=DEFAULT_MODEL_DIR,
                    info="Path to store/download models"
                )
                
                # 模型选择方式
                model_source = gr.Radio(
                    choices=["local", "huggingface"],
                    value="local",
                    label="Model Source"
                )
                
                # 本地模型
                refresh_btn = gr.Button("🔄 Refresh Local Models")
                model_list = scan_local_models(model_dir_input.value)
                model_dropdown = gr.Dropdown(
                    choices=model_list,
                    label="Local Model",
                    value=model_list[0] if model_list else None,
                    visible=True
                )
                
                # HuggingFace 模型
                hf_model_id = gr.Textbox(
                    label="HuggingFace Model ID",
                    value="Qwen/Qwen3-ASR",
                    info="e.g., Qwen/Qwen3-ASR",
                    visible=False
                )
                download_btn = gr.Button("📥 Download from HuggingFace", visible=False)
                
                # 设备配置
                device_dropdown = gr.Dropdown(
                    choices=["auto", "cuda", "cpu"],
                    value="auto",
                    label="Device"
                )
                
                precision_dropdown = gr.Dropdown(
                    choices=["auto", "bf16", "fp16", "fp32"],
                    value="auto",
                    label="Precision"
                )
                
                quant_dropdown = gr.Dropdown(
                    choices=["none", "int8_weight_only", "fp8_weight_only"],
                    value="int8_weight_only",
                    label="Quantization (TorchAO)"
                )
                
                compile_check = gr.Checkbox(
                    label="Enable torch.compile (Not recommended with Quantization)", 
                    value=False
                )
                flash_check = gr.Checkbox(
                    label="Enable Flash Attention", 
                    value=auto_detect_flash_attention()
                )
                
                init_btn = gr.Button("🚀 Initialize Service", variant="primary")
                init_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column():
                gr.Markdown("### 🎵 Transcription")
                audio_input = gr.Audio(type="filepath", label="Upload Audio")
                lang_dropdown = gr.Dropdown(
                    choices=["auto", "Chinese", "English", "Japanese", "Korean"],
                    value="auto",
                    label="Language"
                )
                
                recommended_chunk = get_optimal_chunk_size()
                chunk_slider = gr.Slider(
                    0, 300, 
                    value=recommended_chunk, 
                    step=5, 
                    label=f"Chunk Size (seconds, 0=auto, Recommended: {recommended_chunk}s)"
                )
                overlap_slider = gr.Slider(0, 10, value=2, step=1, label="Overlap (seconds)")
                
                transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
                text_output = gr.Textbox(label="Transcription Text", lines=10)
                timestamp_output = gr.Textbox(label="Info", lines=5)
        
        # ==========================================
        # 事件绑定
        # ==========================================
        def refresh_models(dir):
            models = scan_local_models(dir)
            return gr.Dropdown(choices=models, value=models[0] if models else None)
        
        def toggle_model_source(source):
            if source == "local":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        model_source.change(
            fn=toggle_model_source,
            inputs=[model_source],
            outputs=[model_dropdown, hf_model_id]
        )
        
        refresh_btn.click(fn=refresh_models, inputs=[model_dir_input], outputs=[model_dropdown])
        
        download_btn.click(
            fn=download_and_init,
            inputs=[model_dir_input, hf_model_id, device_dropdown, precision_dropdown, quant_dropdown, compile_check, flash_check],
            outputs=[init_status]
        )
        
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
    parser = argparse.ArgumentParser(description="Qwen3-ASR Open Source Gradio App")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument(
        "--model_dir",
        type=str,
        default=DEFAULT_MODEL_DIR,
        help="Default model directory"
    )
    args = parser.parse_args()
    
    logger.info(f"GPU Tier: {GLOBAL_VRAM_CONFIG.tier} ({GLOBAL_VRAM_CONFIG.gpu_memory_gb:.1f} GB)")
    logger.info(f"Default Precision: {auto_detect_precision()}")
    logger.info(f"Flash Attention Available: {auto_detect_flash_attention()}")
    logger.info(f"Recommended Chunk Size: {get_optimal_chunk_size()}s")
    logger.info(f"Model Directory: {args.model_dir}")

    demo = create_ui()
    demo.launch(
        server_name="127.0.0.1", 
        server_port=args.port, 
        share=args.share,
        inbrowser=True
    )

if __name__ == "__main__":
    main()