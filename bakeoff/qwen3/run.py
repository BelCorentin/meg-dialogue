"""Qwen3-TTS Base (0.6B or 1.7B), zero-shot voice cloning from reference audio + transcript.

    QWEN_SIZE=1.7B python run.py   (default 0.6B)
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import ref, run  # noqa: E402

SIZE = os.environ.get("QWEN_SIZE", "0.6B")
M = {}


def load():
    from qwen_tts import Qwen3TTSModel

    M["m"] = Qwen3TTSModel.from_pretrained(f"Qwen/Qwen3-TTS-12Hz-{SIZE}-Base", device_map="cuda:0",
                                           dtype=torch.bfloat16, attn_implementation="sdpa")
    M["prompt"] = {}


def synth(text, voice):
    m = M["m"]
    if voice not in M["prompt"]:
        wav, txt = ref(voice)
        M["prompt"][voice] = m.create_voice_clone_prompt(ref_audio=wav, ref_text=txt)
    # default max_new_tokens=2048 (~170 s at 12 Hz): "Hmm, c'est une bonne question…" never hit EOS
    # and ran >10 min. Cap at ~2.5x the expected duration (French ~15 chars/s).
    cap = int(12 * (2 + 2.5 * len(text) / 15))
    wavs, sr = m.generate_voice_clone(text=text, language="French", voice_clone_prompt=M["prompt"][voice],
                                      max_new_tokens=cap)
    if len(wavs[0]) / sr >= cap / 12 - 0.5:
        print(f"  !! hit the {cap}-token cap (no end-of-speech): {text[:40]!r}", flush=True)
    return wavs[0], sr, None


if __name__ == "__main__":
    run(f"qwen3-{SIZE}", load, synth)
