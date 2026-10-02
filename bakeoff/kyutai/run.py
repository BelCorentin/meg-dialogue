"""Kyutai TTS 1.6B (en/fr), streaming. Voices are Kyutai's precomputed embeddings of the same
CC0 Unmute recordings used as references by the cloning engines."""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import run  # noqa: E402

VOICE = {"fr_female": "unmute-prod-website/developpeuse-3.wav",
         "fr_male": "unmute-prod-website/fabieng-enhanced-v2.wav"}
M = {}


def load():
    from moshi.models.loaders import CheckpointInfo
    from moshi.models.tts import DEFAULT_DSM_TTS_REPO, TTSModel

    info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
    M["tts"] = TTSModel.from_checkpoint_info(info, n_q=32, temp=0.6, device=torch.device("cuda"))
    M["cond"] = {}


def synth(text, voice):
    tts = M["tts"]
    if voice not in M["cond"]:
        M["cond"][voice] = tts.make_condition_attributes([tts.get_voice_path(VOICE[voice])], cfg_coef=2.0)
    entries = tts.prepare_script([text], padding_between=1)
    pcms, first = [], []
    t0 = time.perf_counter()

    def on_frame(frame):
        if (frame != -1).all():
            pcm = tts.mimi.decode(frame[:, 1:, :]).cpu().numpy()[0, 0]
            if not first and np.abs(pcm).max() > 0.01:
                first.append(time.perf_counter() - t0)
            pcms.append(np.clip(pcm, -1, 1))

    with tts.mimi.streaming(1), torch.no_grad():
        tts.generate([entries], [M["cond"][voice]], on_frame=on_frame)
    return np.concatenate(pcms), tts.mimi.sample_rate, first[0] if first else None


if __name__ == "__main__":
    run("kyutai", load, synth)
