"""STT bake-off on real MEG-scanner mic recordings (MindSentences overt answers).

Reference = derivatives/extension_stt/transcripts_review.tsv, which is itself a Whisper
output (partly reviewed), so WER here is *agreement with that reference*, not ground truth.
Also reported: latency per utterance (warm GPU), real-time factor, guard rejections,
and how often our streaming VAD finds exactly one turn in the answer.

    uv run scripts/bench_stt.py --models small medium large-v3-turbo
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from megdialogue.config import STTConfig, VADConfig
from megdialogue.stt import WhisperSTT
from megdialogue.vad import BLOCK, TurnDetector

HALLU = ("thank you for watching", "thanks for watching", "subscribe", "sous-titr", "amara")
NORM = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()])


def load16k(path: Path) -> np.ndarray:
    # faster_whisper.audio.decode_audio breaks with the installed PyAV (metadata_errors kwarg)
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    g = np.gcd(16000, sr)
    return resample_poly(x.mean(1), 16000 // g, sr // g).astype(np.float32)


def vad_turns(audio: np.ndarray, cfg: VADConfig) -> int:
    det = TurnDetector(cfg)
    n = 0
    padded = np.concatenate([audio, np.zeros(16000, np.float32)])  # trailing silence to close the turn
    for i in range(0, len(padded) - BLOCK + 1, BLOCK):
        n += det.feed(padded[i:i + BLOCK]) is not None
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/scanner_mic")
    ap.add_argument("--models", nargs="+", default=["small", "medium", "large-v3-turbo"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--out", default="runs/bench_stt.jsonl")
    a = ap.parse_args()

    data = Path(a.data)
    ref = {r["wav_file"]: r for r in csv.DictReader(open(data / "transcripts_review.tsv"), delimiter="\t")}
    wavs = sorted(p for p in data.glob("sub-*/*.wav") if p.name in ref)
    audios = {p.name: load16k(p) for p in wavs}
    print(f"{len(wavs)} wavs with reference, {sum(len(x) for x in audios.values()) / 16000:.0f} s audio")

    vcfg = VADConfig()
    nturns = [vad_turns(x, vcfg) for x in audios.values()]
    print(f"VAD turns per answer: {dict(sorted({k: nturns.count(k) for k in set(nturns)}.items()))}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out = open(a.out, "w")
    for m in a.models:
        stt = WhisperSTT(STTConfig(model=m), a.language)
        hyps, refs, lat, rtf, rej, hallu = [], [], [], [], 0, 0
        for name, x in audios.items():
            tr = stt.transcribe(x)
            r = ref[name]["transcript"]
            rej += tr.rejected is not None
            hallu += any(h in tr.text.lower() for h in HALLU) and tr.rejected is None
            hyps.append(NORM(tr.text) or "<empty>")
            refs.append(NORM(r) or "<empty>")
            lat.append(tr.seconds)
            rtf.append(tr.seconds / (len(x) / 16000))
            out.write(json.dumps(dict(model=m, wav=name, ref=r, hyp=tr.text, s=tr.seconds,
                                      lp=tr.avg_logprob, nsp=tr.no_speech_prob, rejected=tr.rejected)) + "\n")
        wer = jiwer.wer(refs, hyps)
        print(f"{m:16s} WER-vs-ref {wer:.3f} | latency median {st.median(lat) * 1000:.0f} ms "
              f"p90 {np.percentile(lat, 90) * 1000:.0f} ms | RTF {st.median(rtf):.3f} | "
              f"guard-rejected {rej} | unguarded hallucinations {hallu}")
        del stt
    print(f"per-utterance rows → {a.out}")


if __name__ == "__main__":
    main()
