# meg-dialogue: design notes, benchmarks, bake-off

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

French: `--config configs/fr_laptop.yaml` = XTTS-v2 cloning `voices/fr_female.wav` (CC0),
streamed chunk by chunk. Kokoro (English config) needs `sudo apt install espeak-ng` for French.
XTTS-v2 is under the Coqui CPML licence: research / non-commercial use only.

**French + XTTS live loop** (2026-10-04, laptop, 3 injected turns): response gap
**1.60–1.76 s** = VAD 608 + STT ~300 (int8) + LLM to first sentence 250–410 + XTTS first
chunk ~400 ms; 0 stalls in 49 within-sentence chunk joins. On 8 GB, Whisper must run int8
or gemma spills 45 % to CPU; even then the LLM and XTTS compete for the GPU (XTTS first
chunk 200 ms alone, ~400 ms here).

## Offline machine

Sessions run with `offline: true` (default): `HF_HUB_OFFLINE=1`, so a missing model
fails at startup instead of downloading mid-experiment.

On a connected machine:

```bash
uv run scripts/prefetch.py configs/*.yaml   # fills caches, then re-loads everything offline
```

Copy to the experiment machine (USB / internal network), same OS + Python 3.12:

| what | from |
|---|---|
| this repo | `git bundle` or the clone |
| Python wheels | `~/.cache/uv` → then `uv sync --offline` |
| STT + TTS weights | `~/.cache/huggingface/hub` (`models--*whisper*`, `models--hexgrad--Kokoro-82M`) |
| LLM | `ollama` binary + its model dir (`/usr/share/ollama/.ollama/models` for the systemd service, else `~/.ollama/models`) |
| system | `espeak-ng` .deb (`apt-get download espeak-ng espeak-ng-data libespeak-ng1`) |

Then `uv run scripts/prefetch.py` is not needed; check with a session using `--mute-mic --inject`.

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

**French live loop** (gemma3:4b, Kokoro `ff_siwis`, 3 synthesized utterances injected):
all transcribed correctly; response gap **1.38–1.47 s**; replies carry an opinion or
anecdote plus one question. qwen2.5:3b in French: blander, and 2.6 s on the first turn
before the LLM was pinned at startup (`llm.warm()`). On clean audio the VAD speech end
lands +72–87 ms after the true energy end.

Not yet measured: acoustic end of speech vs VAD (needs forced alignment, since the injected
recordings carry noise to file end); French; anything in the scanner.

## French TTS bake-off (2026-10-02)

`bakeoff/`: one venv per engine (`install.sh`), same 8 conversational sentences, same two
CC0 French reference voices (Kyutai Unmute recordings `developpeuse-3`, `fabieng-enhanced-v2`),
`run_all.sh` → `out/`, `intelligibility.py` (Whisper WER + speech/clip ratio),
`build_page.py` → blind rating page (`page/`, published as an artifact with shared ratings).

Laptop RTX PRO 1000 8 GB, one sentence at a time:

| engine | synth / sentence | first audio | RTF | VRAM | Whisper WER | speech / clip |
|---|---|---|---|---|---|---|
| Kokoro 82M (current) | 0.11 s | – | 0.03 | small | 8.3 % | 0.81 |
| XTTS-v2 | 0.97 s | **0.20 s** (streamed) | 0.25 | 2.3 GB | 3.2 % | 0.99 |
| Qwen3-TTS 0.6B | 2.46 s | – | 0.77 | 2.8 GB | 1.4 % | 0.94 |
| Qwen3-TTS 1.7B | 2.52 s | – | 0.84 | 4.8 GB | 2.1 % | 0.92 |
| Chatterbox Multilingual | 2.73 s | – | 0.61 | 3.9 GB | 1.8 % | 0.79 |
| Kyutai TTS 1.6B | 3.33 s | **0.52 s** (streamed) | 0.68 | 4.4 GB | 1.5 % | 0.90 |

Only XTTS and Kyutai stream; the others must finish a sentence before it can play.
Qwen3 needs a `max_new_tokens` cap (one sentence never ended, >10 min). Chatterbox
watermarks its output (Perth) and pads ~20 % silence. Naturalness ratings: see the page.

Install traps: coqui-tts needs `transformers<5`, `[codec]`, and `torchcodec<0.9` for
torch 2.9; Chatterbox needs `setuptools<81` (pkg_resources); XTTS downloads to its own
folder (`~/.local/share/tts`), not the HF cache.

## Roadmap

1. **Voice**: XTTS chosen (2026-10-04). Still to do: listen *through the MEG tube
   earphones*; pick female/male reference; optionally record a consenting real speaker.
   **Listen while speaking**: half-duplex currently drops mic input while the bot talks, so a
   participant who starts before the bot finishes loses their first words. With tube
   earphones the mic hears no bot audio, so in the MEG the VAD can keep listening and
   only defer acting until the bot is done.
2. **Latency**: tune `vad.min_silence_ms` on trained speakers (600 → 300–400 is the biggest
   lever); a streaming STT; synth TTS while the LLM is still generating.
3. **Conversation quality**: the 3B is bland and ignores length limits. The desktop
   runs 24–32B (Mistral Small FR / Qwen3); persona prompt and topic scenarios.
4. **Desktop (~€3–4K)**: 24–32 GB GPU; pre-fetch every model (Kokoro also pulls a
   spaCy model at first run, which fails offline).
5. **MEG integration**: trigger per turn/sentence, mic + output into MISC/ADC,
   cross-correlation alignment script, forced alignment for word onsets on both sides.
6. **STT reference**: hand-correct ~50 scanner answers to rank models on accuracy, not agreement.
