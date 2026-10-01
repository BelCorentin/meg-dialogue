"""One duplex stream = one clock for mic and playback.

A single PortAudio callback reads the mic and writes the output at the same rate (48 kHz),
so frame k of the input and frame k of the output are the same instant on the device
clock. Everything is written to session.wav (ch0 = mic, ch1 = exactly what was played,
silence included): any event logged as a `rec_sample` is a row of that file.

In the MEG the authoritative timing is the analog copy of both signals on MISC/ADC
channels; session.wav is what you cross-correlate against them.

The mic is also decimated to 16 kHz (stateful FIR, no block-edge artefacts) for VAD/STT;
16 kHz sample j corresponds to rec_sample 3*j.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import firwin, lfilter, resample_poly

RATE = 48000
DECIM = 3  # 48 kHz → 16 kHz


class _Decimator:
    def __init__(self):
        self.b = firwin(63, 7000, fs=RATE).astype(np.float32)
        self.zi = np.zeros(len(self.b) - 1, np.float32)
        self.phase = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y, self.zi = lfilter(self.b, 1.0, x, zi=self.zi)
        out = y[(-self.phase) % DECIM::DECIM]
        self.phase = (self.phase + len(x)) % DECIM
        return out.astype(np.float32)


class DuplexAudio:
    def __init__(self, path: Path, block16k: int = 512, device=None, inject: np.ndarray | None = None,
                 mute_mic: bool = False):
        """inject: 48 kHz mono signal mixed into the mic channel from frame 0 (repeatable tests).
        mute_mic: zero the real mic first, so only the injected signal is heard."""
        self.inject = inject
        self.mute_mic = mute_mic
        self.block = block16k * DECIM  # 1536 frames = 32 ms
        self.file = sf.SoundFile(path, "w", RATE, 2, subtype="FLOAT")
        self.mic_q: queue.Queue[np.ndarray] = queue.Queue()
        self.rec_q: queue.Queue[np.ndarray] = queue.Queue()
        self.events: queue.Queue[tuple] = queue.Queue()  # (kind, payload) from the callback
        self.chunks: deque[tuple[np.ndarray, dict]] = deque()
        self.cur: np.ndarray | None = None
        self.cur_meta: dict = {}
        self.pos = 0
        self.n = 0  # frames processed = rows of session.wav
        self.lock = threading.Lock()
        self.idle = threading.Event()
        self.idle.set()
        self.decim = _Decimator()
        self._acc = np.zeros(0, np.float32)
        self.read16 = 0  # 16 kHz samples handed to the consumer; rec_sample = 3 * index
        self.stream = sd.Stream(samplerate=RATE, blocksize=self.block, channels=(1, 1), dtype="float32",
                                device=(device, device), callback=self._cb, latency="low")

    # ---- playback -------------------------------------------------------
    def play(self, audio: np.ndarray, rate: int, meta: dict) -> None:
        if rate != RATE:
            g = np.gcd(RATE, rate)
            audio = resample_poly(audio, RATE // g, rate // g)
        with self.lock:
            self.chunks.append((audio.astype(np.float32), meta))
            self.idle.clear()

    def busy(self) -> bool:
        return not self.idle.is_set()

    # ---- callback (real-time thread: no I/O, no logging) ------------------
    def _cb(self, indata, outdata, frames, time_info, status):
        if status:
            self.events.put(("xrun", (str(status), self.n)))
        out = np.zeros(frames, np.float32)
        i = 0
        with self.lock:
            while i < frames:
                if self.cur is None:
                    if not self.chunks:
                        break
                    self.cur, self.cur_meta = self.chunks.popleft()
                    self.pos = 0
                    self.events.put(("onset", (self.cur_meta, self.n + i)))
                take = min(frames - i, len(self.cur) - self.pos)
                out[i:i + take] = self.cur[self.pos:self.pos + take]
                i += take
                self.pos += take
                if self.pos >= len(self.cur):
                    self.events.put(("offset", (self.cur_meta, self.n + i)))
                    self.cur = None
            if self.cur is None and not self.chunks:
                self.idle.set()
        outdata[:, 0] = out
        mic = indata[:, 0].copy()
        if self.mute_mic:
            mic[:] = 0
        if self.inject is not None and self.n < len(self.inject):
            seg = self.inject[self.n:self.n + frames]
            mic[: len(seg)] += seg
        self.rec_q.put(np.stack([mic, out], 1))
        self.mic_q.put(mic)
        self.n += frames

    # ---- consumer side --------------------------------------------------
    def read16k(self, timeout: float = 1.0) -> np.ndarray | None:
        """Next 512-sample 16 kHz mic block (also flushes the recording to disk)."""
        while len(self._acc) < self.block // DECIM:
            try:
                x = self.mic_q.get(timeout=timeout)
            except queue.Empty:
                return None
            self._acc = np.concatenate([self._acc, self.decim(x)])
            self.flush()
        blk, self._acc = self._acc[: self.block // DECIM], self._acc[self.block // DECIM:]
        self.read16 += len(blk)
        return blk

    def flush(self) -> None:
        while not self.rec_q.empty():
            self.file.write(self.rec_q.get_nowait())

    @property
    def latency(self) -> tuple[float, float]:
        return self.stream.latency

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
        self.flush()
        self.file.close()
