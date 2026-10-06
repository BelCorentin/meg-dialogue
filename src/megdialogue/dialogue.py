"""Live loop: mic → VAD end-of-turn → STT → LLM (streamed) → TTS per sentence → speaker.

    uv run megdialogue --config configs/en_laptop.yaml

Each session writes runs/<timestamp>/{session.wav, events.jsonl, config.yaml}.
session.wav is 48 kHz stereo: ch0 = mic, ch1 = what was played. Every audio event is
logged as `rec_sample`, a row of that file (one device clock for both channels).

Half-duplex: the VAD ignores the mic while the system speaks (no barge-in), which keeps
turns clean for epoching and avoids self-triggering on open speakers.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import threading
import time
from pathlib import Path

import yaml

from .audio import DECIM, RATE, DuplexAudio
from .config import Config
from .events import EventLog
from .llm import ChatLLM
from .stt import make_stt
from .tts import make_tts
from .vad import TurnDetector


def respond(text: str, llm: ChatLLM, tts, audio: DuplexAudio, log: EventLog, turn_id: int) -> None:
    log.log("llm_request", turn=turn_id)
    speak(llm.stream_sentences(text, on_first_token=lambda: log.log("llm_first_token", turn=turn_id)),
          tts, audio, log, turn_id)


def speak(sentences, tts, audio: DuplexAudio, log: EventLog, turn_id: int) -> None:
    """Synthesize sentences one by one, queueing each audio chunk for playback as it is generated."""
    n = 0
    for k, sent in enumerate(sentences):
        log.log("llm_sentence", turn=turn_id, idx=k, text=sent)
        print(f"bot: {sent}")
        t0, dur = time.perf_counter(), 0
        for c, chunk in enumerate(tts.stream(sent)):  # play each chunk as soon as it exists
            if c == 0:
                log.log("tts_first_chunk", turn=turn_id, idx=k, synth_s=round(time.perf_counter() - t0, 4))
            audio.play(chunk, tts.rate, {"turn": turn_id, "idx": k, "chunk": c, "text": sent})
            dur += len(chunk)
        log.log("tts_done", turn=turn_id, idx=k, synth_s=round(time.perf_counter() - t0, 4),
                dur_s=round(dur / tts.rate, 3), chunks=c + 1 if dur else 0)
        n += 1
    if not n:
        log.log("llm_empty", turn=turn_id)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en_laptop.yaml")
    ap.add_argument("--inject", nargs="*", default=[],
                    help="wavs mixed into the mic, one every --inject-every s (repeatable test without a speaker)")
    ap.add_argument("--inject-every", type=float, default=15.0)
    ap.add_argument("--inject-start", type=float, default=2.0, help="seconds before the first injected wav")
    ap.add_argument("--mute-mic", action="store_true", help="ignore the real mic (use with --inject)")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    if cfg.offline:
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    inject = None
    if a.inject:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly

        inject = np.zeros(int(RATE * (a.inject_start + a.inject_every * len(a.inject))), np.float32)
        for i, p in enumerate(a.inject):
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            g = np.gcd(RATE, sr)
            x = resample_poly(x.mean(1), RATE // g, sr // g).astype(np.float32)
            s = int(RATE * (a.inject_start + a.inject_every * i))
            inject[s:s + len(x)] += x[: len(inject) - s]

    run = Path(cfg.runs_dir) / time.strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(dataclasses.asdict(cfg), allow_unicode=True))
    log = EventLog(run / "events.jsonl")

    print("loading models…")
    stt = make_stt(cfg.stt, cfg.language)
    tts = make_tts(cfg.tts)
    llm = ChatLLM(cfg.llm)
    llm.warm()  # load + pin in VRAM, and prime the system-prompt cache
    vad = TurnDetector(cfg.vad)
    log.log("models_loaded", llm_gpu_fraction=getattr(llm, "gpu_fraction", None))

    worker: threading.Thread | None = None
    speech_end: dict[int, int] = {}  # turn → rec_sample of end of user speech
    turn_id = 0
    audio = DuplexAudio(run / "session.wav", cfg.audio.block, cfg.audio.input_device, inject=inject,
                        mute_mic=a.mute_mic)
    if a.inject:
        log.log("inject", files=a.inject, every_s=a.inject_every, first_rec_sample=int(a.inject_start * RATE))
    print(f"listening — speak (Ctrl+C to stop). Logging to {run}")
    try:
        with audio:
            lat_in, lat_out = audio.latency
            log.log("stream_started", rate=RATE, latency_in=lat_in, latency_out=lat_out)
            if cfg.opening:  # the bot greets first (turn 0), and remembers that it did
                llm.history.append({"role": "assistant", "content": cfg.opening})
                log.log("opening", turn=0, text=cfg.opening)
                sents = [s for s in re.split(r"(?<=[.!?…])\s+", cfg.opening.strip()) if s]
                worker = threading.Thread(target=speak, args=(sents, tts, audio, log, 0), daemon=True)
                worker.start()
            while True:
                block = audio.read16k()
                while not audio.events.empty():
                    kind, payload = audio.events.get_nowait()
                    if kind == "xrun":
                        log.log("xrun", status=payload[0], rec_sample=payload[1])
                        continue
                    meta, n = payload
                    ev = dict(turn=meta["turn"], idx=meta["idx"], chunk=meta.get("chunk", 0), rec_sample=n)
                    if kind == "onset" and meta["idx"] == 0 and meta["turn"] in speech_end:
                        gap = (n - speech_end.pop(meta["turn"])) / RATE
                        ev["gap_s"] = round(gap, 4)  # end of speech → first output sample, same clock
                        ev["acoustic_gap_s"] = round(gap + lat_in + lat_out, 4)  # + device buffers
                        print(f"  ↳ response gap {ev['acoustic_gap_s'] * 1000:.0f} ms (end of speech → sound out)")
                    log.log(f"play_{kind}", **ev)
                if block is None:
                    continue
                if audio.busy() or (worker is not None and worker.is_alive()):
                    vad.reset()
                    vad.n = audio.read16  # keep VAD indices on the session clock
                    continue
                turn = vad.feed(block)
                if turn is None:
                    continue
                turn_id += 1
                rec = {k: v * DECIM for k, v in dict(start_sample=turn.start_sample, speech_end_sample=turn.speech_end_sample,
                                                    end_sample=turn.end_sample, detected_sample=turn.detected_sample).items()}
                log.log("user_turn", turn=turn_id, **{f"rec_{k}": v for k, v in rec.items()})
                tr = stt.transcribe(turn.audio)
                log.log("stt", turn=turn_id, text=tr.text, stt_s=round(tr.seconds, 4),
                        avg_logprob=round(tr.avg_logprob, 3), no_speech_prob=round(tr.no_speech_prob, 3),
                        rejected=tr.rejected)
                print(f"you: {tr.text}" + (f"   [rejected: {tr.rejected}]" if tr.rejected else ""))
                if tr.rejected:
                    continue
                speech_end[turn_id] = rec["speech_end_sample"]
                worker = threading.Thread(target=respond, args=(tr.text, llm, tts, audio, log, turn_id), daemon=True)
                worker.start()
    except KeyboardInterrupt:
        pass
    finally:
        log.log("session_end", rec_samples=audio.n)
        log.close()
        print(f"\nsaved {run}")


if __name__ == "__main__":
    main()
