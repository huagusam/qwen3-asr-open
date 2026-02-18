"""
Qwen3-ASR VAD-Aware Handler (Open Source Version)
基于语音活动检测的智能分块，消除重复问题
"""
import torch
import gc
import numpy as np
import os
import torch.nn.functional as F
import time
import threading
import re
import warnings
from collections import OrderedDict
from typing import Optional, Dict, Any, Tuple, List, Union
from dataclasses import dataclass
from loguru import logger

# ==========================================
# 默认配置
# ==========================================
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".qwen3_asr", "models")

# ==========================================
# 依赖检查
# ==========================================
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    raise ImportError("qwen_asr package not found. Please install it first.")

# ==========================================
# VAD 配置与工具
# ==========================================
@dataclass
class VADConfig:
    """VAD 配置参数"""
    aggressiveness: int = 2
    frame_duration_ms: int = 30
    padding_duration_ms: int = 300
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 300
    target_chunk_sec: float = 15.0
    max_chunk_sec: float = 20.0
    min_chunk_sec: float = 5.0

class VADProcessor:
    """WebRTC VAD 包装器，支持多种备选方案"""
    def __init__(self, config: VADConfig = None):
        self.config = config or VADConfig()
        self.vad = None
        self._init_vad()

    def _init_vad(self):
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(self.config.aggressiveness)
            self.mode = "webrtc"
            logger.info(f"VAD initialized: WebRTC (aggressiveness={self.config.aggressiveness})")
        except ImportError:
            logger.warning("webrtcvad not installed, falling back to energy-based VAD")
            self.mode = "energy"
            self._energy_threshold = None

    def _frame_generator(self, audio: np.ndarray, sample_rate: int):
        n = int(sample_rate * (self.config.frame_duration_ms / 1000.0))
        offset = 0
        while offset + n <= len(audio):
            yield audio[offset:offset + n]
            offset += n

    def _energy_based_vad(self, audio: np.ndarray, sample_rate: int) -> List[bool]:
        frames = list(self._frame_generator(audio, sample_rate))
        energies = [np.sqrt(np.mean(frame**2)) for frame in frames]
        
        if self._energy_threshold is None:
            self._energy_threshold = np.percentile(energies, 20) * 2
        
        return [e > self._energy_threshold for e in energies]

    def detect_speech_regions(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int]]:
        if sample_rate not in (8000, 16000, 32000, 48000):
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000
        
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        if self.mode == "webrtc":
            frames = list(self._frame_generator(audio, sample_rate))
            is_speech = []
            for frame in frames:
                try:
                    speech = self.vad.is_speech(frame.tobytes(), sample_rate)
                    is_speech.append(speech)
                except:
                    is_speech.append(False)
        else:
            is_speech = self._energy_based_vad(audio, sample_rate)
        
        regions = []
        frame_samples = int(sample_rate * self.config.frame_duration_ms / 1000)
        
        i = 0
        while i < len(is_speech):
            while i < len(is_speech) and not is_speech[i]:
                i += 1
            if i >= len(is_speech):
                break
            
            start_frame = i
            silence_count = 0
            while i < len(is_speech):
                if is_speech[i]:
                    silence_count = 0
                else:
                    silence_count += 1
                    silence_duration = silence_count * self.config.frame_duration_ms
                    if silence_duration >= self.config.min_silence_duration_ms:
                        break
                i += 1
            
            end_frame = i - silence_count
            start_sample = start_frame * frame_samples
            end_sample = min(end_frame * frame_samples, len(audio))
            
            duration_ms = (end_sample - start_sample) / sample_rate * 1000
            if duration_ms >= self.config.min_speech_duration_ms:
                regions.append((start_sample, end_sample))
        
        return regions

    def find_optimal_split_points(self, audio: np.ndarray, sample_rate: int) -> List[int]:
        regions = self.detect_speech_regions(audio, sample_rate)
        if not regions:
            return []
        
        split_points = [0]
        target_samples = int(self.config.target_chunk_sec * sample_rate)
        max_samples = int(self.config.max_chunk_sec * sample_rate)
        min_samples = int(self.config.min_chunk_sec * sample_rate)
        
        current_start = 0
        
        for i in range(len(regions) - 1):
            current_region_end = regions[i][1]
            next_region_start = regions[i + 1][0]
            silence_start = current_region_end
            silence_end = next_region_start
            
            if current_region_end - current_start >= target_samples:
                if silence_end > silence_start:
                    split_point = (silence_start + silence_end) // 2
                    if split_point - current_start <= max_samples:
                        split_points.append(split_point)
                        current_start = split_point
        
        if regions[-1][1] - current_start > max_samples:
            last_start, last_end = regions[-1]
            sub_splits = self._force_split_in_region(
                audio[last_start:last_end],  
                sample_rate, 
                current_start
            )
            split_points.extend(sub_splits)
        
        split_points.append(len(audio))
        return split_points

    def _force_split_in_region(self, region_audio: np.ndarray, sample_rate: int, 
                               offset: int) -> List[int]:
        window_size = int(sample_rate * 0.5)
        hop_size = int(sample_rate * 0.1)
        
        energies = []
        positions = []
        
        for i in range(0, len(region_audio) - window_size, hop_size):
            window = region_audio[i:i+window_size]
            energy = np.sqrt(np.mean(window**2))
            energies.append(energy)
            positions.append(i + window_size // 2)
        
        if not energies:
            return []
        
        splits = []
        current_pos = 0
        target = int(self.config.target_chunk_sec * sample_rate)
        
        for i, (pos, energy) in enumerate(zip(positions, energies)):
            if offset + pos - current_pos >= target:
                window_start = max(0, i - 5)
                window_end = min(len(energies), i + 5)
                local_min_idx = window_start + np.argmin(energies[window_start:window_end])
                split_pos = offset + positions[local_min_idx]
                splits.append(split_pos)
                current_pos = split_pos
        
        return splits

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        indices = np.linspace(0, len(audio) - 1, new_length)
        indices = np.clip(indices, 0, len(audio) - 1)
        indices_floor = indices.astype(np.int32)
        indices_ceil = np.minimum(indices_floor + 1, len(audio) - 1)
        fractions = indices - indices_floor
        
        return audio[indices_floor] * (1 - fractions) + audio[indices_ceil] * fractions

    def create_chunks(self, audio: np.ndarray, sample_rate: int) -> List[Tuple[int, int, np.ndarray]]:
        split_points = self.find_optimal_split_points(audio, sample_rate)
        chunks = []
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            
            pad_samples = int(self.config.padding_duration_ms / 1000 * sample_rate)
            pad_start = max(0, start - pad_samples // 2)
            pad_end = min(len(audio), end + pad_samples // 2)
            
            chunk_audio = audio[pad_start:pad_end]
            chunks.append((start, end, chunk_audio))
        
        return chunks

# ==========================================
# 全局配置
# ==========================================
class VRAMConfig:
    """GPU Memory Configuration"""
    def __init__(self):
        self.gpu_memory_gb = 0
        self.tier = "unknown"
        self._detect()
    
    def _detect(self):
        if torch.cuda.is_available():
            self.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if self.gpu_memory_gb >= 24:
                self.tier = "high"
            elif self.gpu_memory_gb >= 12:
                self.tier = "medium"
            elif self.gpu_memory_gb >= 8:
                self.tier = "low"
            else:
                self.tier = "very_low"
        else:
            self.tier = "cpu"

GLOBAL_VRAM_CONFIG = VRAMConfig()

# ==========================================
# 智能缓存 (LRU)
# ==========================================
class ThreadSafeLRUCache:
    """Thread-safe LRU cache for model management"""
    def __init__(self, max_size: int = 2):
        self._cache = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, model: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    old_key, old_model = self._cache.popitem(last=False)
                    del old_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            self._cache[key] = model

    def remove(self, key: str):
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def clear(self):
        with self._lock:
            self._cache.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self):
        return len(self._cache)

_MODEL_CACHE = ThreadSafeLRUCache(max_size=2)

# ==========================================
# 量化管理器
# ==========================================
class QuantizationManager:
    """Manages TorchAO quantization"""
    def __init__(self):
        self._available = False
        self._check()
    
    def _check(self):
        try:
            import torchao
            self._available = True
            logger.info(f"TorchAO version: {torchao.__version__}")
        except ImportError:
            self._available = False
            logger.warning("TorchAO not installed. Quantization disabled.")

    def is_available(self) -> bool:
        return self._available

    def apply_quantization(self, model: torch.nn.Module, quant_type: str) -> bool:
        if not self._available:
            return False
        try:
            from torchao.quantization import quantize_
            
            if quant_type == "int8_weight_only":
                try:
                    from torchao.quantization import Int8WeightOnlyConfig
                    config = Int8WeightOnlyConfig()
                except ImportError:
                    from torchao.quantization import int8_weight_only
                    quantize_(model, int8_weight_only())
                    logger.info("Applied legacy int8_weight_only quantization")
                    return True
            elif quant_type == "fp8_weight_only":
                try:
                    from torchao.quantization import Float8WeightOnlyConfig
                    config = Float8WeightOnlyConfig()
                except ImportError:
                    logger.warning("FP8 quantization not available in this TorchAO version")
                    return False
            else:
                logger.warning(f"Unknown quantization type: {quant_type}")
                return False
            
            logger.info(f"Applying {quant_type} quantization...")
            quantize_(model, config)
            logger.info("Quantization applied successfully")
            return True
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return False

_QUANT_MANAGER = QuantizationManager()

# ==========================================
# 辅助函数
# ==========================================
def scan_local_models(model_dir: str) -> List[str]:
    """Scan local model directory"""
    models = []
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Created model directory: {model_dir}")
        return ["No Models Found - Please download first"]
    
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path):
            models.append(item)
    
    return sorted(models) if models else ["No Models Found - Please download first"]

def scan_comfy_models(model_dir: str) -> List[str]:
    """Scan ComfyUI-style model directory for Qwen3-ASR models"""
    models = []
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return models
    for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if os.path.isdir(item_path) and "Qwen3-ASR" in item:
            models.append(item)
    return sorted(models) if models else ["No Models Found"]

def download_model(model_id: str, model_dir: str) -> Optional[str]:
    """Download model from HuggingFace"""
    try:
        from huggingface_hub import snapshot_download
        
        os.makedirs(model_dir, exist_ok=True)
        
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
        
        logger.info(f"Model downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None

def auto_detect_precision() -> str:
    if not torch.cuda.is_available():
        return "fp32"
    try:
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 8:
            return "bf16"
        elif capability[0] >= 7:
            return "fp16"
        else:
            return "fp32"
    except:
        return "fp32"

def auto_detect_flash_attention() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import flash_attn
        capability = torch.cuda.get_device_capability()
        return capability[0] >= 8
    except ImportError:
        return False

def get_optimal_chunk_size() -> int:
    tier = GLOBAL_VRAM_CONFIG.tier
    if tier == "very_low":
        return 15
    elif tier == "low":
        return 20
    elif tier == "medium":
        return 30
    else:
        return 60

# ==========================================
# 核心 Handler (VAD 版本)
# ==========================================
class Qwen3ASRHandler:
    """VAD-aware Qwen3-ASR Handler"""
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = "cpu"
        self.dtype = torch.float32
        self.model = None
        self.current_model_key = None
        self.vad_processor = VADProcessor()
    
    def initialize(
        self,
        model_name: str,
        device: str = "auto",
        precision: str = "auto",
        quantization: str = "none",
        compile_model: bool = False,
        flash_attention: bool = False
    ) -> Tuple[str, bool]:
        try:
            # 1. Device Selection
            if device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            # 2. Precision Selection
            if precision == "auto":
                precision = auto_detect_precision()
            dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
            self.dtype = dtype_map.get(precision, torch.float32)
            
            # 3. Flash Attention
            if flash_attention and self.device == "cuda":
                if not auto_detect_flash_attention():
                    logger.warning("Flash Attention requested but not available. Falling back to SDPA.")
                    flash_attention = False
            
            # 4. Cache Key
            cache_key = f"{model_name}_{self.device}_{precision}_{quantization}_{flash_attention}"
            
            # 5. Check Cache
            cached_model = _MODEL_CACHE.get(cache_key)
            if cached_model is not None:
                logger.info(f"Model loaded from cache: {cache_key}")
                self.model = cached_model
                self.current_model_key = cache_key
                return "✅ Model loaded from cache", True
            
            # 6. 强制清理显存
            logger.info("Forcing VRAM cleanup before loading new model...")
            _MODEL_CACHE.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # 7. Load Model
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                return f"❌ Model not found: {model_path}", False
            
            logger.info(f"Loading model: {model_path}")
            start_time = time.time()
            
            attn_impl = "flash_attention_2" if flash_attention else "sdpa"
            load_device = self.device
            
            model = Qwen3ASRModel.from_pretrained(
                pretrained_model_name_or_path=model_path,
                dtype=self.dtype,
                device_map=load_device,
                attn_implementation=attn_impl
            )
            
            # 8. Apply Quantization
            if quantization != "none":
                if _QUANT_MANAGER.apply_quantization(model.model, quantization):
                    logger.info(f"Quantization applied: {quantization}")
            
            # 9. Apply Compile
            if compile_model and self.device == "cuda":
                try:
                    logger.info("Compiling model with torch.compile...")
                    model.model = torch.compile(model.model)
                except Exception as e:
                    logger.warning(f"Compilation failed: {e}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            
            # 10. Cache Model
            _MODEL_CACHE.put(cache_key, model)
            self.model = model
            self.current_model_key = cache_key
            
            return f"✅ Model initialized successfully\nLoad time: {load_time:.2f}s", True
            
        except Exception as e:
            logger.exception("Initialization failed")
            return f"❌ Error: {str(e)}", False

    def transcribe(
        self,
        audio_waveform: np.ndarray,
        sample_rate: int,
        language: str = "auto",
        use_vad: bool = True,
        chunk_size: int = 0,
        overlap: int = 2
    ) -> Tuple[str, str]:
        if self.model is None:
            return "", "❌ Model not initialized"
        
        try:
            if sample_rate != 16000:
                audio_waveform = self._resample(audio_waveform, sample_rate, 16000)
                sample_rate = 16000
            
            duration = len(audio_waveform) / sample_rate
            
            if use_vad and duration > self.vad_processor.config.target_chunk_sec:
                logger.info(f"Using VAD-aware chunking for {duration:.1f}s audio")
                return self._transcribe_vad_aware(audio_waveform, sample_rate, language)
            elif chunk_size > 0 and duration > chunk_size:
                logger.info(f"Using fixed chunking ({chunk_size}s) for {duration:.1f}s audio")
                return self._transcribe_fixed_chunk(audio_waveform, sample_rate, language, chunk_size, overlap)
            else:
                logger.info(f"Single pass transcription for {duration:.1f}s audio")
                return self._transcribe_single(audio_waveform, sample_rate, language)
                
        except Exception as e:
            logger.exception("Transcription failed")
            return "", f"❌ Error: {str(e)}"

    def _transcribe_single(self, audio_waveform, sample_rate, language):
        lang_param = None if language == "auto" else language
        results = self.model.transcribe(
            audio=[(audio_waveform, sample_rate)],
            language=lang_param,
            return_time_stamps=False
        )
        text = self._post_process(results[0].text)
        return text, ""

    def _transcribe_vad_aware(self, audio_waveform, sample_rate, language):
        lang_param = None if language == "auto" else language
        
        chunks = self.vad_processor.create_chunks(audio_waveform, sample_rate)
        logger.info(f"VAD split audio into {len(chunks)} chunks")
        
        texts = []
        timestamps = []
        
        for i, (orig_start, orig_end, chunk_audio) in enumerate(chunks):
            chunk_duration = len(chunk_audio) / sample_rate
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}: {chunk_duration:.1f}s")
            
            results = self.model.transcribe(
                audio=[(chunk_audio, sample_rate)],
                language=lang_param,
                return_time_stamps=False,
            )
            
            text = results[0].text.strip()
            text = self._post_process(text)
            
            if text:
                texts.append(text)
                timestamps.append((orig_start / sample_rate, orig_end / sample_rate))
                logger.debug(f"Chunk {i+1} result: {text[:50]}...")
        
        merged_text = self._merge_vad_chunks(texts, timestamps)
        
        return merged_text, ""

    def _transcribe_fixed_chunk(self, audio_waveform, sample_rate, language, chunk_size, overlap):
        chunk_samples = chunk_size * sample_rate
        overlap_samples = overlap * sample_rate
        step_samples = chunk_samples - overlap_samples
        
        total_samples = len(audio_waveform)
        num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))
        
        texts = []
        lang_param = None if language == "auto" else language

        for i in range(num_chunks):
            start = i * step_samples
            end = min(start + chunk_samples, total_samples)
            chunk = audio_waveform[start:end]
            
            results = self.model.transcribe(
                audio=[(chunk, sample_rate)],
                language=lang_param,
                return_time_stamps=False
            )
            
            text = self._post_process(results[0].text.strip())
            
            if i == 0:
                texts.append(text)
            else:
                new_text = self._deduplicate_lcs(texts[-1], text)
                if new_text:
                    texts.append(new_text)
        
        return self._smart_join(texts), ""

    def _post_process(self, text: str) -> str:
        if not text:
            return ""
        
        text = text.strip()
        text = self._remove_consecutive_duplicates(text)
        
        hallucinations = [
            r'谢谢观看', r'感谢收听', r'字幕由\w+提供', 
            r'听写 [:：]', r'转录 [:：]', r'^\s* 谢谢\s*$',
            r'订阅.*频道', r'点击.*关注'
        ]
        for pattern in hallucinations:
            text = re.sub(pattern, '', text)
        
        text = re.sub(r'[，,]\s*[，,]+', '，', text)
        text = re.sub(r'[。.]\s*[。.]+', '。', text)
        
        return text.strip()

    def _remove_consecutive_duplicates(self, text: str, max_repeat: int = 2) -> str:
        sentences = re.split(r'([.！？.!?])', text)
        filtered = []
        prev_clean = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                punct = sentences[i + 1]
            else:
                punct = ""
            
            clean = re.sub(r'[^\w]', '', sentence)
            
            if prev_clean and self._similarity(clean, prev_clean) > 0.8:
                continue
            
            filtered.append(sentence + punct)
            prev_clean = clean
        
        text = ''.join(filtered)
        
        words = re.split(r'(\s+)', text)
        result = []
        prev_word = None
        repeat_count = 0
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word == prev_word:
                repeat_count += 1
                if repeat_count < max_repeat:
                    result.append(word)
            else:
                repeat_count = 0
                result.append(word)
                prev_word = clean_word
        
        return ''.join(result)

    def _similarity(self, s1: str, s2: str) -> float:
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0
        
        max_len = 0
        for i in range(len(s1)):
            for j in range(len(s2)):
                k = 0
                while i + k < len(s1) and j + k < len(s2) and s1[i+k] == s2[j+k]:
                    k += 1
                max_len = max(max_len, k)
        
        return 2 * max_len / (len(s1) + len(s2))

    def _deduplicate_lcs(self, prev: str, curr: str, min_overlap: int = 6) -> str:
        if not prev or not curr:
            return curr
        
        clean_prev = re.sub(r'[^\w]', '', prev)
        clean_curr = re.sub(r'[^\w]', '', curr)
        
        if clean_curr in clean_prev:
            return ""
        
        max_overlap = 0
        max_len = min(len(clean_prev), len(clean_curr), 50)
        
        for i in range(max_len, min_overlap - 1, -1):
            if clean_prev[-i:] == clean_curr[:i]:
                max_overlap = i
                break
        
        if max_overlap >= min_overlap:
            count = 0
            for idx, char in enumerate(curr):
                if char.isalnum():
                    count += 1
                if count == max_overlap:
                    return curr[idx+1:]
            return ""
        else:
            return "  " + curr if not curr.startswith("  ") else curr

    def _merge_vad_chunks(self, texts: List[str], timestamps: List[Tuple[float, float]]) -> str:
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        merged = [texts[0]]
        
        for i in range(1, len(texts)):
            prev_end = timestamps[i-1][1]
            curr_start = timestamps[i][0]
            gap = curr_start - prev_end
            
            curr_text = texts[i]
            prev_text = merged[-1]
            
            if gap > 1.0:
                if not prev_text.endswith(('。', '！', '？', '.', '!', '?')):
                    merged[-1] = prev_text + '。'
                merged.append(curr_text)
            else:
                clean_prev = re.sub(r'[^\w]', '', prev_text[-20:])
                clean_curr = re.sub(r'[^\w]', '', curr_text[:20])
                
                if clean_prev.endswith(clean_curr) or clean_curr.startswith(clean_prev):
                    overlap_len = len(clean_curr)
                    for j in range(min(overlap_len, len(prev_text)), 0, -1):
                        if re.sub(r'[^\w]', '', prev_text[-j:]) == clean_curr[:j]:
                            merged[-1] = prev_text + curr_text[j:]
                            break
                    else:
                        merged.append(curr_text)
                else:
                    merged.append(curr_text)
        
        return ''.join(merged)

    def _smart_join(self, texts: List[str]) -> str:
        if not texts:
            return ""
        
        result = []
        for i, text in enumerate(texts):
            if not text:
                continue
            
            if i > 0 and result:
                prev = result[-1]
                if (not re.search(r'[.！？.!?]\s*$', prev) and 
                    not re.search(r'^\s*[，,；;]', text)):
                    if any('\u4e00' <= c <= '\u9fff' for c in prev[-1] + text[0]):
                        pass
                    else:
                        text = "  " + text
            
            result.append(text)
        
        return ''.join(result)

    def _resample(self, audio, orig_sr, target_sr):
        if orig_sr == target_sr:
            return audio
        
        waveform = torch.from_numpy(audio).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        try:
            import torchaudio
            waveform = torchaudio.transforms.Resample(orig_sr, target_sr)(waveform)
        except:
            ratio = target_sr / orig_sr
            new_length = int(waveform.shape[-1] * ratio)
            waveform = F.interpolate(waveform.unsqueeze(0), size=new_length, mode='linear', align_corners=False).squeeze(0)
        
        return waveform.squeeze().numpy()

    def unload(self):
        if self.current_model_key:
            _MODEL_CACHE.remove(self.current_model_key)
            self.model = None
            self.current_model_key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def update_vad_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.vad_processor.config, key):
                setattr(self.vad_processor.config, key, value)
                logger.info(f"VAD config updated: {key} = {value}")