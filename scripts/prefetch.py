"""Download everything the given configs need, then prove it loads with the network off.

    uv run scripts/prefetch.py configs/en_laptop.yaml configs/fr_laptop.yaml

Fills: HF cache (faster-whisper, Kokoro weights + voices), Ollama models. Silero VAD and
the spaCy English model ship inside the venv. To move to an offline machine, see
README "Offline machine".
"""

from __future__ import annotations

import os
import subprocess
import sys

from megdialogue.config import Config


def main() -> None:
    cfgs = [Config.load(p) for p in sys.argv[1:]]
    from huggingface_hub import hf_hub_download, snapshot_download

    for c in cfgs:
        from faster_whisper.utils import download_model

        print(f"STT  {c.stt.model}: {download_model(c.stt.model)}")
        if c.tts.backend == "kokoro":
            snapshot_download("hexgrad/Kokoro-82M", allow_patterns=["*.json", "*.pth"])
            hf_hub_download("hexgrad/Kokoro-82M", f"voices/{c.tts.voice}.pt")
            print(f"TTS  kokoro voice {c.tts.voice}")
        if c.tts.backend == "xtts":
            os.environ.setdefault("COQUI_TOS_AGREED", "1")  # CPML: research / non-commercial use
            from TTS.utils.manage import ModelManager

            path = ModelManager().download_model("tts_models/multilingual/multi-dataset/xtts_v2")[0]
            print(f"TTS  xtts-v2 → {path} (Coqui's own folder, not the HF cache)")
        if "11434" in c.llm.base_url:
            subprocess.run(["ollama", "pull", c.llm.model], check=True)
            print(f"LLM  ollama {c.llm.model}")

    print("\nverifying offline load…")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    for p in sys.argv[1:]:
        code = ("from megdialogue.config import Config; from megdialogue.stt import make_stt; "
                "from megdialogue.tts import make_tts; from megdialogue.vad import TurnDetector; "
                f"c=Config.load({p!r}); make_stt(c.stt, c.language); make_tts(c.tts); TurnDetector(c.vad); print('  ok', {p!r})")
        subprocess.run([sys.executable, "-c", code], env=env, check=True)


if __name__ == "__main__":
    main()
