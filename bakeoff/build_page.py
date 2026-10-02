"""Build the blind listening page's assets from bakeoff/out/.

- every clip → mp3, loudness-normalised to the same level (louder clips otherwise win)
- random clip ids; the id → engine map ships base64-encoded so it isn't readable at a glance
- per-engine timing summaries (revealed with the results)

    python bakeoff/build_page.py   →  bakeoff/page/{clips/*.mp3, clips.json}
"""

from __future__ import annotations

import base64
import json
import random
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
OUT, PAGE = HERE / "out", HERE / "page"
LABELS = {
    "kokoro": "Kokoro 82M (current)",
    "qwen3-0.6B": "Qwen3-TTS 0.6B",
    "qwen3-1.7B": "Qwen3-TTS 1.7B",
    "kyutai": "Kyutai TTS 1.6B",
    "chatterbox": "Chatterbox Multilingual",
    "xtts": "XTTS-v2",
}


def main() -> None:
    (PAGE / "clips").mkdir(parents=True, exist_ok=True)
    for f in (PAGE / "clips").glob("*.mp3"):
        f.unlink()
    rng = random.Random(7)
    sentences = [l.strip() for l in open(HERE / "sentences_fr.txt") if l.strip()]
    clips, engines = [], {}
    for edir in sorted(p for p in OUT.iterdir() if p.is_dir()):
        timing = edir / "timing.json"
        if not timing.exists():
            print(f"skip {edir.name}: no timing.json (incomplete run)")
            continue
        engines[edir.name] = dict(label=LABELS.get(edir.name, edir.name), **json.load(open(timing))["summary"])
        for wav in sorted(edir.glob("*/*.wav")):
            cid = f"{rng.getrandbits(40):010x}"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(wav),
                            "-af", "loudnorm=I=-20:TP=-2:LRA=11", "-ar", "24000", "-ac", "1",
                            "-b:a", "64k", str(PAGE / "clips" / f"{cid}.mp3")], check=True)
            clips.append(dict(id=cid, voice=wav.parent.name, idx=int(wav.stem), engine=edir.name))
    secret = base64.b64encode(json.dumps(
        dict(map={c["id"]: c["engine"] for c in clips}, engines=engines)).encode()).decode()
    for c in clips:
        del c["engine"]
    json.dump(dict(sentences=sentences, clips=clips, k=secret), open(PAGE / "clips.json", "w"), ensure_ascii=False)
    print(f"{len(clips)} clips from {len(engines)} engines → {PAGE}")


if __name__ == "__main__":
    main()
