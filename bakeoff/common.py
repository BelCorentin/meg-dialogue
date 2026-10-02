"""Shared harness for the TTS bake-off. Each engine script runs in its own venv and calls
`run(engine, load, synth)`; only numpy + soundfile + stdlib are used here.

synth(text, voice) -> (audio float32 mono, sample_rate, ttfa_s or None)
  ttfa_s = time to first audio chunk for streaming engines (None if not streaming).

Writes bakeoff/out/<engine>/<voice>/NN.wav and bakeoff/out/<engine>/timing.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).parent
VOICES = ["fr_female", "fr_male"]


def ref(voice: str) -> tuple[str, str]:
    return str(HERE / "refs" / f"{voice}.wav"), (HERE / "refs" / f"{voice}.txt").read_text().strip()


def run(engine: str, load, synth) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", default=str(HERE / "sentences_fr.txt"))
    ap.add_argument("--voices", nargs="+", default=VOICES)
    a = ap.parse_args()
    sents = [s.strip() for s in open(a.sentences) if s.strip()]

    t = time.perf_counter()
    load()
    load_s = time.perf_counter() - t
    import torch

    rows = []
    for v in a.voices:
        out = HERE / "out" / engine / v
        out.mkdir(parents=True, exist_ok=True)
        synth(sents[0], v)  # warm-up per voice (conditioning caches, kernels)
        for i, s in enumerate(sents):
            torch.cuda.synchronize()
            t = time.perf_counter()
            audio, sr, ttfa = synth(s, v)
            dt = time.perf_counter() - t
            audio = np.asarray(audio, np.float32).squeeze()
            peak = float(np.abs(audio).max()) or 1.0
            sf.write(out / f"{i:02d}.wav", audio / max(peak, 1.0), sr)
            rows.append(dict(voice=v, idx=i, text=s, synth_s=dt, audio_s=len(audio) / sr, ttfa_s=ttfa))
            print(f"[{engine}/{v}] {i:02d} synth {dt * 1000:5.0f} ms"
                  + (f" ttfa {ttfa * 1000:4.0f} ms" if ttfa is not None else "")
                  + f" audio {len(audio) / sr:4.1f} s  {s[:50]!r}", flush=True)
    summ = dict(engine=engine, load_s=load_s, vram_peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                synth_median_s=st.median(r["synth_s"] for r in rows),
                rtf_median=st.median(r["synth_s"] / r["audio_s"] for r in rows),
                ttfa_median_s=st.median(r["ttfa_s"] for r in rows) if rows[0]["ttfa_s"] is not None else None)
    json.dump(dict(summary=summ, rows=rows), open(HERE / "out" / engine / "timing.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(summ, indent=1))
