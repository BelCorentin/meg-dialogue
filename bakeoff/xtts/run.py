"""XTTS-v2 (Coqui, CPML non-commercial licence), zero-shot cloning, streaming inference."""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import ref, run  # noqa: E402

os.environ["COQUI_TOS_AGREED"] = "1"  # CPML: research / non-commercial use
M = {}


def load():
    from TTS.api import TTS

    M["model"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda").synthesizer.tts_model
    M["cond"] = {}


def synth(text, voice):
    m = M["model"]
    if voice not in M["cond"]:
        M["cond"][voice] = m.get_conditioning_latents(audio_path=[ref(voice)[0]])
    lat, spk = M["cond"][voice]
    t0, first, chunks = time.perf_counter(), None, []
    for c in m.inference_stream(text, "fr", lat, spk, enable_text_splitting=False):
        if first is None:
            first = time.perf_counter() - t0
        chunks.append(c.cpu().numpy())
    return np.concatenate(chunks), 24000, first


if __name__ == "__main__":
    run("xtts", load, synth)
