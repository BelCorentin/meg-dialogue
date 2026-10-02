"""Fetch the CC0 French reference voices (Kyutai Unmute's own recordings) used for cloning."""

import shutil
from pathlib import Path

import soundfile as sf
from huggingface_hub import hf_hub_download

REFS = {
    "fr_female": "unmute-prod-website/developpeuse-3.wav",
    "fr_male": "unmute-prod-website/fabieng-enhanced-v2.wav",
}

out = Path(__file__).parent / "refs"
out.mkdir(exist_ok=True)
for name, path in REFS.items():
    p = hf_hub_download("kyutai/tts-voices", path)
    shutil.copy(p, out / f"{name}.wav")
    i = sf.info(p)
    print(f"{name}: {path} {i.duration:.1f} s {i.samplerate} Hz")
