"""Streaming end-of-turn detection with Silero VAD on 512-sample blocks at 16 kHz."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import VADConfig

RATE = 16000
BLOCK = 512


@dataclass
class Turn:
    start_sample: int  # into the continuous mic recording (16 kHz), padding included
    speech_end_sample: int  # last voiced block end: zero point for response latency
    end_sample: int  # speech_end + pad
    detected_sample: int  # sample at which end-of-turn was declared (end + min_silence)
    audio: np.ndarray


class TurnDetector:
    def __init__(self, cfg: VADConfig):
        from silero_vad import load_silero_vad

        self.cfg = cfg
        self.model = load_silero_vad(onnx=True)
        self.pad = int(cfg.speech_pad_ms * RATE / 1000)
        self.min_sil = int(cfg.min_silence_ms * RATE / 1000)
        self.min_speech = int(cfg.min_speech_ms * RATE / 1000)
        self.reset()

    def reset(self) -> None:
        self.model.reset_states()
        self.n = 0  # samples consumed
        self.ring: list[np.ndarray] = []  # recent blocks for pre-speech padding
        self.buf: list[np.ndarray] = []
        self.in_speech = False
        self.start = self.last_voiced = 0

    def feed(self, block: np.ndarray) -> Turn | None:
        """Feed exactly BLOCK float32 samples; returns a Turn when end-of-turn is declared."""
        import torch

        p = float(self.model(torch.from_numpy(block), RATE))
        self.n += len(block)
        voiced = p >= self.cfg.threshold
        if not self.in_speech:
            self.ring = (self.ring + [block])[-(self.pad // BLOCK + 1):]
            if voiced:
                self.in_speech = True
                self.buf = list(self.ring)
                self.start = max(0, self.n - sum(len(b) for b in self.buf))
                self.last_voiced = self.n
            return None
        self.buf.append(block)
        if voiced:
            self.last_voiced = self.n
            return None
        if self.n - self.last_voiced < self.min_sil:
            return None
        # end of turn
        end = min(self.last_voiced + self.pad, self.n)
        audio = np.concatenate(self.buf)[: end - self.start]
        speech_len = self.last_voiced - self.start
        self.in_speech, self.buf, self.ring = False, [], []
        if speech_len < self.min_speech:
            return None
        return Turn(self.start, self.last_voiced, end, self.n, audio)
