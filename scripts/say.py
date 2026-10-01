"""Synthesize sentences with a config's TTS to wav files (listening tests, --inject material).

    uv run scripts/say.py --config configs/fr_laptop.yaml --out runs/say "Bonjour, comment ça va ?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from megdialogue.config import Config
from megdialogue.tts import make_tts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="runs/say")
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    tts = make_tts(Config.load(a.config).tts)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(a.text):
        wav, dt = tts.synth(t)
        p = Path(a.out) / f"{i:02d}.wav"
        sf.write(p, wav, tts.rate)
        print(f"{p}  synth {dt * 1000:.0f} ms  audio {len(wav) / tts.rate:.2f} s  {t!r}")


if __name__ == "__main__":
    main()
