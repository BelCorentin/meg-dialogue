"""TTS backends. Each exposes `rate`, `stream(text)` (yields float32 chunks as they are
generated, so playback can start before the sentence is finished) and `synth(text)`."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import numpy as np

from .config import TTSConfig


class _Base:
    rate: int

    def stream(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def synth(self, text: str) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        parts = list(self.stream(text))
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio, time.perf_counter() - t


class KokoroTTS(_Base):
    """Kokoro-82M: fast, preset voices, no cloning, robotic in French (rated 1/5).
    French (lang_code='f') needs espeak-ng."""

    rate = 24000

    def __init__(self, cfg: TTSConfig):
        from kokoro import KPipeline

        self.cfg = cfg
        self.pipe = KPipeline(lang_code=cfg.lang_code, device=cfg.device, repo_id="hexgrad/Kokoro-82M")
        self.synth("Hello.")  # warm-up: first call pays model + G2P init

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for r in self.pipe(text, voice=self.cfg.voice, speed=self.cfg.speed):
            if r.audio is not None:
                yield r.audio.numpy().astype(np.float32)


class XTTS(_Base):
    """XTTS-v2 (Coqui; CPML licence = non-commercial / research use). Zero-shot cloning of
    `cfg.voice` (a reference wav), streamed: first audio ~0.2 s on the 8 GB laptop.
    Won the 2026-10-02 French bake-off on quality x latency."""

    rate = 24000

    def __init__(self, cfg: TTSConfig):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS

        self.cfg = cfg
        self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(cfg.device).synthesizer.tts_model
        self.latent, self.speaker = self.model.get_conditioning_latents(audio_path=[cfg.voice])
        self.synth("Bonjour." if cfg.lang_code == "fr" else "Hello.")  # warm-up

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for c in self.model.inference_stream(text, self.cfg.lang_code, self.latent, self.speaker,
                                             stream_chunk_size=self.cfg.stream_chunk_size,
                                             speed=self.cfg.speed, enable_text_splitting=False):
            yield c.float().cpu().numpy()


def make_tts(cfg: TTSConfig):
    if cfg.backend == "kokoro":
        return KokoroTTS(cfg)
    if cfg.backend == "xtts":
        return XTTS(cfg)
    raise ValueError(f"unknown TTS backend {cfg.backend!r}")
