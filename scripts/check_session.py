"""Verify a session's logged timings against its own recording, and decompose latency.

For each bot onset: is ch1 (played) silent just before rec_sample and non-silent from it?
For each user turn: where does ch0 (mic) energy actually stop vs the logged speech end?
Then the per-turn latency budget from events.jsonl.

    uv run scripts/check_session.py runs/<session>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 48000


def main() -> None:
    run = Path(sys.argv[1])
    ev = [json.loads(l) for l in open(run / "events.jsonl")]
    x, sr = sf.read(run / "session.wav", dtype="float32")
    assert sr == RATE
    mic, out = x[:, 0], x[:, 1]
    print(f"{run.name}: {len(x) / RATE:.1f} s recorded, "
          f"xruns {sum(e['kind'] == 'xrun' for e in ev)}")

    on = [e for e in ev if e["kind"] == "play_onset"]
    errs = []
    for e in on:
        n = e["rec_sample"]
        nz = np.flatnonzero(np.abs(out[n:n + RATE]) > 0)
        before = np.abs(out[max(0, n - 480):n]).max(initial=0)
        errs.append(nz[0] if len(nz) else -1)
        if before > 0 and e["idx"] == 0:
            print(f"  turn {e['turn']}: output not silent before first onset ({before:.3g})")
    print(f"play onsets: {len(on)}, first non-zero sample offset from logged onset "
          f"(TTS leading silence included): median {np.median(errs) / RATE * 1000:.1f} ms, "
          f"max {max(errs) / RATE * 1000:.1f} ms")

    turns = {e["turn"]: e for e in ev if e["kind"] == "user_turn"}
    by = lambda k: {e["turn"]: e for e in ev if e["kind"] == k}  # noqa: E731
    stt, first_tok, first_sent, tts0 = by("stt"), by("llm_first_token"), {}, {}
    for e in ev:
        if e["kind"] == "llm_sentence" and e["idx"] == 0:
            first_sent[e["turn"]] = e
        if e["kind"] == "tts_done" and e["idx"] == 0:
            tts0[e["turn"]] = e
    onset0 = {e["turn"]: e for e in on if e["idx"] == 0}
    env = np.sqrt(np.convolve(mic ** 2, np.ones(480) / 480, "same"))  # 10 ms RMS
    print("\nturn | mic energy end vs logged speech end | VAD wait | STT | LLM→1st sentence | TTS | →out | total gap")
    for t, u in turns.items():
        if t not in onset0:
            continue
        se = u["rec_speech_end_sample"]
        win = env[u["rec_start_sample"]:se + RATE]
        thr = 0.1 * np.percentile(win, 95)
        act = np.flatnonzero(win > thr)
        true_end = u["rec_start_sample"] + act[-1] if len(act) else se
        det_t = turns[t]["t"]
        vad_wait = (u["rec_detected_sample"] - se) / RATE
        stt_s = stt[t]["t"] - det_t
        llm_s = first_sent[t]["t"] - stt[t]["t"]
        tts_s = tts0[t]["t"] - first_sent[t]["t"]
        gap = onset0[t]["gap_s"]
        rest = gap - vad_wait - stt_s - llm_s - tts_s
        print(f"{t:4d} | {(se - true_end) / RATE * 1000:+6.0f} ms | {vad_wait * 1000:4.0f} | {stt_s * 1000:4.0f} | "
              f"{llm_s * 1000:4.0f} | {tts_s * 1000:4.0f} | {rest * 1000:4.0f} | {onset0[t]['acoustic_gap_s'] * 1000:.0f} ms")


if __name__ == "__main__":
    main()
