from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .config import STTConfig

# Whisper's stock outputs on silence/noise (seen on the MEG mic: "Thank you for watching.", "you")
HALLUCINATIONS = {"you", "thank you", "thank you for watching", "thanks for watching",
                  "merci", "sous-titres réalisés par la communauté d'amara.org", "merci d'avoir regardé"}


@dataclass
class Transcript:
    text: str
    language: str
    avg_logprob: float
    no_speech_prob: float
    seconds: float  # wall time spent transcribing
    rejected: str | None = None  # reason if the guard dropped it


class WhisperSTT:
    """faster-whisper, with a guard against the noise hallucinations seen on the scanner mic."""

    def __init__(self, cfg: STTConfig, language: str | None):
        import torch  # noqa: F401  loads the pip CUDA libs (libcublas.so.12) that ctranslate2 needs
        from faster_whisper import WhisperModel

        self.cfg = cfg
        self.language = language
        self.model = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type)
        self.transcribe(np.zeros(16000, np.float32))  # warm-up

    def transcribe(self, audio16k: np.ndarray) -> Transcript:
        t = time.perf_counter()
        segs, info = self.model.transcribe(
            audio16k.astype(np.float32),
            language=self.language,
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            vad_filter=False,  # our own VAD already segmented the turn
            without_timestamps=True,
        )
        segs = list(segs)
        dt = time.perf_counter() - t
        text = " ".join(s.text.strip() for s in segs).strip()
        lp = float(np.mean([s.avg_logprob for s in segs])) if segs else float("-inf")
        nsp = float(np.max([s.no_speech_prob for s in segs])) if segs else 1.0
        tr = Transcript(text, info.language, lp, nsp, dt)
        if not text:
            tr.rejected = "empty"
        elif text.lower().strip(" .!?") in HALLUCINATIONS:
            tr.rejected = "stock_hallucination"
        elif nsp > self.cfg.no_speech_threshold and lp < self.cfg.min_avg_logprob:
            tr.rejected = "no_speech"
        return tr


def make_stt(cfg: STTConfig, language: str | None):
    if cfg.backend == "faster-whisper":
        return WhisperSTT(cfg, language)
    raise ValueError(f"unknown STT backend {cfg.backend!r}")
