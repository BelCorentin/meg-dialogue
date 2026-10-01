"""Check that a duplex stream runs in real time on a device: plays a quiet 440 Hz tone
for a few seconds and reports frames processed vs wall time, xruns, and reported latency.

    uv run scripts/check_audio.py --device pipewire --seconds 5
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", nargs="+", default=["pipewire", "default"])
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--seconds", type=float, default=5)
    a = ap.parse_args()
    for dev in a.device:
        dev = int(dev) if dev.isdigit() else dev
        st = {"frames": 0, "status": set()}
        tone = (0.05 * np.sin(2 * np.pi * 440 * np.arange(int(a.rate * a.seconds)) / a.rate)).astype(np.float32)

        def cb(indata, outdata, frames, ti, status):
            if status:
                st["status"].add(str(status))
            o = tone[st["frames"]:st["frames"] + frames]
            outdata[:] = 0
            outdata[: len(o), 0] = o
            st["frames"] += frames

        try:
            with sd.Stream(device=(dev, dev), samplerate=a.rate, channels=(1, 1), dtype="float32",
                           callback=cb, latency="low") as s:
                t = time.time()
                time.sleep(a.seconds)
                el = time.time() - t
            print(f"{dev}: wall {el:.2f}s, processed {st['frames'] / a.rate:.2f}s, "
                  f"status {st['status'] or 'clean'}, latency in/out {s.latency}")
        except Exception as e:  # noqa: BLE001
            print(f"{dev}: ERROR {e}")


if __name__ == "__main__":
    main()
