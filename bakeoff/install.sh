#!/usr/bin/env bash
# One venv per engine (their pins conflict). torch forced to 2.9.1 (cu128): Blackwell needs >=2.7.
# usage: bakeoff/install.sh kyutai qwen3 chatterbox xtts
set -u
cd "$(dirname "$0")"
declare -A PKGS=(
  [kyutai]="moshi soundfile"
  [qwen3]="qwen-tts soundfile"
  [chatterbox]="chatterbox-tts soundfile"
  [xtts]="coqui-tts[codec] transformers<5 torch torchaudio soundfile"
)
for e in "$@"; do
  (
    uv venv -q --python 3.12 "$e/.venv" &&
    VIRTUAL_ENV="$e/.venv" uv pip install -q --override overrides.txt ${PKGS[$e]} &&
    echo "[$e] OK" || echo "[$e] FAILED"
  ) > "$e/install.log" 2>&1 &
done
wait
for e in "$@"; do tail -1 "$e/install.log"; done
