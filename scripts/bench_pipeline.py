"""Offline end-to-end latency, no audio devices: user wav → STT → LLM (streamed) → TTS of
the first sentence. Reports the compute part of "end of user speech → first bot sample".

The live budget adds: VAD end-of-turn silence (cfg.vad.min_silence_ms) + audio output
buffer + transport (tubes, ~10-20 ms; Bluetooth headsets 150-250 ms — never use BT).

    uv run scripts/bench_pipeline.py --config configs/en_laptop.yaml --n 8
"""

from __future__ import annotations

import argparse
import statistics as st
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from megdialogue.config import Config
from megdialogue.llm import ChatLLM
from megdialogue.stt import make_stt
from megdialogue.tts import make_tts


def load16k(path: Path) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    g = np.gcd(16000, sr)
    return resample_poly(x.mean(1), 16000 // g, sr // g).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en_laptop.yaml")
    ap.add_argument("--wavs", default="data/scanner_mic/sub-00*/*.wav")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--save", default="runs/bench_pipeline")
    a = ap.parse_args()
    cfg = Config.load(a.config)

    stt, tts = make_stt(cfg.stt, cfg.language), make_tts(cfg.tts)
    wavs = sorted(Path().glob(a.wavs))[:: max(1, len(sorted(Path().glob(a.wavs))) // a.n)][: a.n]
    Path(a.save).mkdir(parents=True, exist_ok=True)
    ChatLLM(cfg.llm).stream_sentences("Hi.").__next__()  # warm the LLM (load weights into VRAM)

    rows = []
    for i, w in enumerate(wavs):
        llm = ChatLLM(cfg.llm)  # fresh context each time: first-turn latency
        t0 = time.perf_counter()
        tr = stt.transcribe(load16k(w))
        t_stt = time.perf_counter()
        ttft = []
        gen = llm.stream_sentences(tr.text, on_first_token=lambda: ttft.append(time.perf_counter()))
        first = next(gen, "")
        t_sent = time.perf_counter()
        audio, _ = tts.synth(first)
        t_audio = time.perf_counter()
        rest = list(gen)
        sf.write(f"{a.save}/{i:02d}.wav", np.concatenate([audio] + [tts.synth(s)[0] for s in rest]), tts.rate)
        r = dict(stt=t_stt - t0, ttft=ttft[0] - t_stt if ttft else float("nan"),
                 sent=t_sent - t_stt, tts=t_audio - t_sent, total=t_audio - t0)
        rows.append(r)
        print(f"[{i}] user: {tr.text!r}\n    bot:  {' '.join([first] + rest)!r}\n    "
              + " | ".join(f"{k} {v * 1000:.0f} ms" for k, v in r.items()))
    print("\nmedian  " + " | ".join(f"{k} {st.median(r[k] for r in rows) * 1000:.0f} ms" for k in rows[0]))
    print(f"+ VAD end-of-turn silence {cfg.vad.min_silence_ms} ms + output buffer → live mouth-to-ear")
    print(f"bot audio → {a.save}/")


if __name__ == "__main__":
    main()
