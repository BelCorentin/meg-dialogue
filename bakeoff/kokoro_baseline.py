"""Kokoro ff_siwis (the current pipeline voice) on the same sentences: the baseline anchor.
Runs in the main project venv:  uv run bakeoff/kokoro_baseline.py"""

import json
import statistics as st
from pathlib import Path

import soundfile as sf

from megdialogue.config import Config
from megdialogue.tts import make_tts

HERE = Path(__file__).parent
tts = make_tts(Config.load(HERE.parent / "configs/fr_laptop.yaml").tts)
out = HERE / "out/kokoro/fr_female"
out.mkdir(parents=True, exist_ok=True)
rows = []
for i, s in enumerate(l.strip() for l in open(HERE / "sentences_fr.txt") if l.strip()):
    a, dt = tts.synth(s)
    sf.write(out / f"{i:02d}.wav", a, tts.rate)
    rows.append(dict(voice="fr_female", idx=i, text=s, synth_s=dt, audio_s=len(a) / tts.rate, ttfa_s=None))
summ = dict(engine="kokoro", load_s=None, vram_peak_gb=None, ttfa_median_s=None,
            synth_median_s=st.median(r["synth_s"] for r in rows),
            rtf_median=st.median(r["synth_s"] / r["audio_s"] for r in rows))
json.dump(dict(summary=summ, rows=rows), open(HERE / "out/kokoro/timing.json", "w"), indent=1, ensure_ascii=False)
print(summ)
