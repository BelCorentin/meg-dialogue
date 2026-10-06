# meg-dialogue

A spoken conversation partner for MEG experiments that runs on one local GPU machine, with no
internet. The participant talks; the system listens, transcribes, answers with a local
language model, and speaks back in a natural French (or English) voice, usually within
1.5–2 s. Every sound in and out is logged on one clock so both sides of the conversation
can be analysed against the MEG.

```
mic → voice activity detection → Whisper (speech-to-text) → local LLM → XTTS (text-to-speech) → earphones
```

## Requirements

- Linux with an NVIDIA GPU, 8 GB of VRAM or more (tested on an 8 GB laptop RTX PRO 1000;
  24–32 GB recommended for the experiment machine), recent NVIDIA driver
- Python 3.12 and [uv](https://docs.astral.sh/uv/) (or plain `pip`)
- [Ollama](https://ollama.com) to serve the language model
- A microphone and **wired** headphones (Bluetooth adds delay and stalled the audio in tests)
- English config only: `sudo apt install espeak-ng`

Python packages are pinned in `uv.lock` (and exported to `requirements.txt`).

## Install

```bash
git clone git@github.com:BelCorentin/meg-dialogue.git
cd meg-dialogue
uv sync                      # or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

ollama pull gemma3:4b        # French conversation model
ollama pull qwen2.5:3b       # English conversation model

# download speech-to-text and voice models, then check that everything loads offline
uv run scripts/prefetch.py configs/fr_laptop.yaml configs/en_laptop.yaml
```

After this, nothing needs the internet. To move to an offline machine, see
[docs/NOTES.md](docs/NOTES.md#offline-machine).

## Run

```bash
ollama stop gemma3:4b        # make sure the model reloads fully onto the GPU
uv run megdialogue --config configs/fr_laptop.yaml
```

Camille greets you first; answer when she has finished speaking. She waits for about
0.6 s of silence before replying, and does not listen while she is talking. Press
Ctrl+C to stop.

If you see `WARNING: gemma3:4b is only N% on the GPU`, another program (often a second
`megdialogue` session) is holding GPU memory: close it, run `ollama stop gemma3:4b`, and restart.

English: `uv run megdialogue --config configs/en_laptop.yaml`.

**Test without speaking**: feed recorded sentences in place of the mic.

```bash
uv run megdialogue --config configs/fr_laptop.yaml --mute-mic \
    --inject-start 14 --inject-every 25 --inject sentence1.wav sentence2.wav
```

## What a session saves

Each run writes `runs/<date_time>/`:

| file | contents |
|---|---|
| `session.wav` | 48 kHz stereo: left = microphone, right = exactly what was played |
| `events.jsonl` | every event (speech detected, transcript, each reply sentence, each sound onset) with its sample position in `session.wav` |
| `config.yaml` | the settings used |

`uv run scripts/check_session.py runs/<date_time>` verifies the timings and prints the
response delay of each turn, split into listening, transcription, language model and voice.

`runs/` can contain participants' voices and is never committed.

## Change the conversation

Everything is in `configs/fr_laptop.yaml`:

| setting | what it changes |
|---|---|
| `opening` | what Camille says first (empty = she waits for the participant) |
| `llm.system_prompt` | who she is, how she talks, example exchanges |
| `tts.voice` | the voice to imitate: `voices/fr_female.wav` or `voices/fr_male.wav`, or any clean 10 s recording |
| `vad.min_silence_ms` | how long a pause ends the participant's turn (lower = faster, but cuts people off) |
| `llm.model` | any model pulled in Ollama (bigger = better conversation, needs more VRAM) |

`uv run scripts/try_openers.py --config configs/fr_laptop.yaml` prints her replies to a set
of typical first sentences, text only, to tune the prompt quickly.

## Licences

The code is ours. The XTTS-v2 voice model is under the Coqui Public Model Licence:
**research and non-commercial use only**. The reference voices in `voices/` are CC0
(Kyutai Unmute recordings).

## More

[docs/NOTES.md](docs/NOTES.md): design choices, latency measurements, the French voice
comparison, the offline-machine checklist and the roadmap.
