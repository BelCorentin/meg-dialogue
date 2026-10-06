"""Print the LLM's reply to typical first turns, each in a fresh conversation (text only, no audio).
Use it to tune the persona prompt and the opening line.

    uv run scripts/try_openers.py --config configs/fr_laptop.yaml
"""

from __future__ import annotations

import argparse

from megdialogue.config import Config
from megdialogue.llm import ChatLLM

OPENERS = {
    "fr": ["Bonjour.", "Salut, ça va ?", "Bonjour, moi c'est Paul.", "Euh… je sais pas trop quoi dire.",
           "Oui.", "Il fait beau aujourd'hui."],
    "en": ["Hello.", "Hi, how are you?", "Hi, I'm Paul.", "Um… I don't really know what to say.",
           "Yes.", "It's sunny today."],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fr_laptop.yaml")
    a = ap.parse_args()
    cfg = Config.load(a.config)
    for u in OPENERS.get(cfg.language, OPENERS["en"]):
        llm = ChatLLM(cfg.llm)
        if cfg.opening:
            llm.history.append({"role": "assistant", "content": cfg.opening})
        print(f"\nuser: {u}\nbot:  {' '.join(llm.stream_sentences(u))}")


if __name__ == "__main__":
    main()
