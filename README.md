# meg-dialogue

Offline, low-latency spoken conversation with a participant in the MEG scanner:
**mic → VAD end-of-turn → STT → LLM (streamed) → TTS per sentence → earphones**, on one
local GPU machine with no internet. Built so both sides of the exchange can be analysed
with precise timing.

## Design choices

| Choice | Why |
|---|---|
| Cascade, not speech-to-speech | Text + timestamps at every stage; best voice and best LLM can be picked independently; French support |
| VAD turn-taking, no button | Least motor contamination (trained participants) |
| Half-duplex, no barge-in | Clean turns for epoching; no self-triggering |
| **One duplex audio stream** | Mic and playback share one device clock. `session.wav` = ch0 mic, ch1 exactly what was played (silence included); every audio event is logged as `rec_sample`, a row of that file |
| LLM behind an OpenAI-compatible URL | Ollama on the laptop, llama.cpp / vLLM on the desktop, no code change |

In the MEG, the authoritative timing is the analog copy of mic + played audio on
MISC/ADC channels. `session.wav` is what you cross-correlate against them to place every
logged event on the MEG clock.

## Layout

```
src/megdialogue/
  dialogue.py   live loop (CLI: megdialogue)
  audio.py      duplex stream, session.wav, 48→16 kHz stateful decimation, --inject
  vad.py        Silero streaming end-of-turn
  stt.py        faster-whisper + hallucination guard
  llm.py        streaming chat, sentence chunking
  tts.py        Kokoro (first backend)
  events.py     JSONL event log
configs/        en_laptop.yaml
scripts/
  bench_stt.py       STT bake-off on real scanner-mic recordings
  bench_pipeline.py  offline STT→LLM→TTS latency, no audio device
  check_audio.py     is a duplex stream real-time on this device?
  check_session.py   verify a session's logged timings against its recording; latency budget
```

`data/` (participant voice) and `runs/` (sessions) are gitignored and must stay so.

## Run

```bash
uv sync
ollama pull qwen2.5:3b          # 8 GB laptop; bigger model on the desktop
uv run megdialogue --config configs/en_laptop.yaml
# repeatable test without talking: inject recorded answers, ignore the real mic
uv run megdialogue --config configs/en_laptop.yaml --mute-mic --inject a.wav b.wav --inject-every 25
uv run scripts/check_session.py runs/<session>
```

Use wired output. Bluetooth adds 150–250 ms and stalled the stream in testing.

## Results so far (2026-10-01, laptop RTX PRO 1000 8 GB)

**STT on scanner-mic recordings** (77 MindSentences overt answers, sub-001..003, English).
The reference is itself Whisper output (partly reviewed, sometimes garbled), so WER is
*agreement*, not accuracy:

| model | WER vs ref | median latency | p90 |
|---|---|---|---|
| small | 0.257 | 115 ms | 150 ms |
| medium | 0.170 | 300 ms | 366 ms |
| large-v3-turbo | 0.154 | 369 ms | 390 ms |

Disagreements cluster at `avg_logprob < −0.5`, which is a usable "please repeat" cue.
Whisper's stock noise outputs ("Thank you for watching.", "you") appear on this mic;
`stt.py` blocklists them. Our VAD finds exactly one turn in 76/77 answers.

**End-to-end, offline bench** (8 answers, first turn, warm): STT 337 + LLM to first
sentence 244 + TTS 126 = **~700 ms compute** with qwen2.5:3b fully on GPU. With
qwen2.5:7b, Ollama put 17 % of the model on CPU (VRAM taken by Whisper + Kokoro) and the
first sentence took 3.3 s median. On 8 GB the stack must fit on the GPU, or latency collapses.

**Live loop, injected answers, laptop speakers** (3 turns): response gap **1.2–1.5 s**
(end of speech → first sample out, device clock, incl. 8+8 ms buffers). Budget:
VAD silence 608 · STT ~310 · LLM→1st sentence 130–340 · TTS 95–190 ms. Logged onsets
match the first non-zero output sample exactly (internal consistency only, not acoustic
truth). 1 xrun in 82 s.

Not yet measured: acoustic end of speech vs VAD (needs forced alignment, since the injected
recordings carry noise to file end); French; anything in the scanner.

## Roadmap

1. **French**: Kokoro FR needs `sudo apt install espeak-ng` and has one voice. Bake off
   Qwen3-TTS 0.6B/1.7B (cached; FR, voice cloning), Chatterbox multilingual, Kyutai TTS,
   by blind listening, *through the MEG tube earphones*.
2. **Latency**: tune `vad.min_silence_ms` on trained speakers (600 → 300–400 is the biggest
   lever); a streaming STT; synth TTS while the LLM is still generating.
3. **Conversation quality**: the 3B is bland and ignores length limits. The desktop
   runs 24–32B (Mistral Small FR / Qwen3); persona prompt and topic scenarios.
4. **Desktop (~€3–4K)**: 24–32 GB GPU; pre-fetch every model (Kokoro also pulls a
   spaCy model at first run, which fails offline).
5. **MEG integration**: trigger per turn/sentence, mic + output into MISC/ADC,
   cross-correlation alignment script, forced alignment for word onsets on both sides.
6. **STT reference**: hand-correct ~50 scanner answers to rank models on accuracy, not agreement.
