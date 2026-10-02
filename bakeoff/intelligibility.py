"""Objective check on every bake-off clip: Whisper transcript vs intended text (WER), plus
speech duration vs clip duration (trailing silence or babble shows up as a low ratio).

    PYTHONPATH=src .venv/bin/python bakeoff/intelligibility.py   (main venv)

Adds `wer_median`, `bad_clips` (WER > 0.2) and `speech_ratio_median` to each timing.json summary.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch  # noqa: F401  (loads CUDA libs for ctranslate2)
from faster_whisper import WhisperModel
from scipy.signal import resample_poly
from silero_vad import get_speech_timestamps, load_silero_vad

HERE = Path(__file__).parent
NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])


def load16k(p: Path) -> np.ndarray:
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    g = np.gcd(16000, sr)
    return resample_poly(x.mean(1), 16000 // g, sr // g).astype(np.float32)


def main() -> None:
    stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    vad = load_silero_vad()
    sents = [l.strip() for l in open(HERE / "sentences_fr.txt") if l.strip()]
    for tj in sorted((HERE / "out").glob("*/timing.json")):
        d = json.load(open(tj))
        wers, ratios, bad = [], [], []
        for r in d["rows"]:
            x = load16k(tj.parent / r["voice"] / f"{r['idx']:02d}.wav")
            segs, _ = stt.transcribe(x, language="fr", beam_size=5, condition_on_previous_text=False)
            hyp = " ".join(s.text.strip() for s in segs)
            w = jiwer.wer(NORM(sents[r["idx"]]), NORM(hyp) or "<empty>")
            ts = get_speech_timestamps(torch.from_numpy(x), vad, sampling_rate=16000)
            speech = (ts[-1]["end"] - ts[0]["start"]) / 16000 if ts else 0.0
            r.update(asr=hyp, wer=w, speech_s=speech, speech_ratio=speech / (len(x) / 16000))
            wers.append(w)
            ratios.append(r["speech_ratio"])
            if w > 0.2:
                bad.append(f"{r['voice']}/{r['idx']:02d}: {hyp!r}")
        d["summary"].update(wer_median=st.median(wers), wer_mean=float(np.mean(wers)),
                            bad_clips=len(bad), speech_ratio_median=st.median(ratios))
        json.dump(d, open(tj, "w"), indent=1, ensure_ascii=False)
        print(f"{tj.parent.name:12s} WER mean {np.mean(wers):.3f} | clips WER>0.2: {len(bad)}/{len(wers)} | "
              f"speech/clip {st.median(ratios):.2f}")
        for b in bad:
            print("    " + b)


if __name__ == "__main__":
    main()
