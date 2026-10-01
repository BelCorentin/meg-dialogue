from __future__ import annotations

import re
from collections.abc import Callable, Iterator

from openai import OpenAI

from .config import LLMConfig

# flush to TTS at sentence ends, or at a clause break once the buffer is long
_SENT_END = re.compile(r"[.!?…]+[\"»)]?\s")
_CLAUSE = re.compile(r"[,;:]\s")


class ChatLLM:
    """Any OpenAI-compatible server (Ollama, llama.cpp, vLLM). Streams sentence-sized chunks."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.base_url, api_key="local")
        self.history: list[dict] = []
        if cfg.system_prompt:
            self.history.append({"role": "system", "content": cfg.system_prompt})

    def warm(self) -> None:
        """Load the model and keep it resident: a cold Ollama model cost ~1.5 s on the first turn,
        and Ollama unloads idle models after 5 min by default."""
        if "11434" in self.cfg.base_url:  # Ollama native API: no prompt = load only
            import json
            import urllib.request

            url = self.cfg.base_url.removesuffix("/v1").rstrip("/") + "/api/generate"
            body = json.dumps({"model": self.cfg.model, "keep_alive": -1}).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=300).read()
        self.client.chat.completions.create(model=self.cfg.model, max_tokens=1,
                                            messages=self.history + [{"role": "user", "content": "Bonjour."}])

    def stream_sentences(self, user_text: str, on_first_token: Callable[[], None] | None = None) -> Iterator[str]:
        self.history.append({"role": "user", "content": user_text})
        stream = self.client.chat.completions.create(
            model=self.cfg.model,
            messages=self.history,
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_tokens,
            seed=self.cfg.seed,
            stream=True,
        )
        buf, full, n_out = "", "", 0
        for chunk in stream:
            delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if not delta:
                continue
            if not full and on_first_token:
                on_first_token()
            buf += delta
            full += delta
            # clause breaks only for the first chunk (latency); later chunks are whole sentences,
            # since each chunk is synthesized separately and intonation resets at chunk edges
            while m := (_SENT_END.search(buf) or (not n_out and len(buf) > 80 and _CLAUSE.search(buf, 40))):
                if s := _clean(buf[: m.end()]):
                    n_out += 1
                    yield s
                buf = buf[m.end():]
        if s := _clean(buf):
            yield s
        self.history.append({"role": "assistant", "content": full})


def _clean(s: str) -> str:
    # strip markdown the TTS would read aloud
    return re.sub(r"[*_#`]", "", s).strip()
