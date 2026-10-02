"""Chatterbox Multilingual (Resemble AI), zero-shot cloning from reference audio."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import ref, run  # noqa: E402

M = {}


def load():
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    M["m"] = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
    M["voice"] = None


def synth(text, voice):
    m = M["m"]
    if M["voice"] != voice:
        m.prepare_conditionals(ref(voice)[0], exaggeration=0.5)
        M["voice"] = voice
    wav = m.generate(text, language_id="fr")
    return wav.squeeze().cpu().numpy(), m.sr, None


if __name__ == "__main__":
    run("chatterbox", load, synth)
