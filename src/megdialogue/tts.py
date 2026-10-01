from __future__ import annotations

import time

import numpy as np

from .config import TTSConfig


class KokoroTTS:
    """Kokoro-82M: fast, preset voices, no cloning. French (lang_code='f') needs espeak-ng."""

    rate = 24000

    def __init__(self, cfg: TTSConfig):
        from kokoro import KPipeline

        self.cfg = cfg
        self.pipe = KPipeline(lang_code=cfg.lang_code, device=cfg.device, repo_id="hexgrad/Kokoro-82M")
        self.synth("Hello.")  # warm-up: first call pays model + G2P init

    def synth(self, text: str) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        parts = [r.audio.numpy() for r in self.pipe(text, voice=self.cfg.voice, speed=self.cfg.speed)
                 if r.audio is not None]
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return audio.astype(np.float32), time.perf_counter() - t


def make_tts(cfg: TTSConfig):
    if cfg.backend == "kokoro":
        return KokoroTTS(cfg)
    raise ValueError(f"unknown TTS backend {cfg.backend!r}")
