from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class VADConfig:
    threshold: float = 0.5
    min_silence_ms: int = 600  # end-of-turn silence: the dominant latency term
    min_speech_ms: int = 250
    speech_pad_ms: int = 150


@dataclass
class STTConfig:
    backend: str = "faster-whisper"
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 1
    # hallucination guard: the scanner mic makes Whisper emit "Thank you for watching." on noise
    no_speech_threshold: float = 0.6
    min_avg_logprob: float = -1.0


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434/v1"  # Ollama; llama.cpp / vLLM expose the same API
    model: str = "qwen2.5:7b"
    temperature: float = 0.7
    max_tokens: int = 160
    seed: int = 0
    system_prompt: str = ""


@dataclass
class TTSConfig:
    backend: str = "kokoro"
    voice: str = "af_heart"  # kokoro: preset name; xtts: path to a reference wav
    lang_code: str = "a"  # kokoro: a=en-US, f=fr (needs espeak-ng); xtts: fr, en, …
    speed: float = 1.0
    stream_chunk_size: int = 20  # xtts: GPT tokens per streamed chunk (smaller = earlier first audio)
    device: str = "cuda"


@dataclass
class AudioConfig:
    in_rate: int = 16000
    out_rate: int = 24000
    block: int = 512  # silero VAD takes exactly 512 samples at 16 kHz (32 ms)
    input_device: str | int | None = None
    output_device: str | int | None = None


@dataclass
class Config:
    language: str = "en"
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    runs_dir: str = "runs"
    opening: str = ""  # what the bot says first, before the participant speaks ("" = wait for them)
    offline: bool = True  # HF_HUB_OFFLINE=1: a missing model fails at startup, never downloads mid-session

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        sub = {"vad": VADConfig, "stt": STTConfig, "llm": LLMConfig, "tts": TTSConfig, "audio": AudioConfig}
        return cls(**{k: (sub[k](**v) if k in sub else v) for k, v in raw.items()})
