#!/bin/bash
# Fetches the models the app ships with into ios/Model (not in git), at pinned Hugging Face revisions:
#   asr/  Qwen3-ASR 1.7B 5-bit MLX (model.safetensors, config.json, tokenizer files), about 2.4 GB
#   vad/  Silero VAD v6.2.1 CoreML
# CI runs it before xcodegen; run it once before building locally. Needs the `hf` CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"
ASR_REV=1645ca9ce8767dac9a4a047e9de835950c42596e        # aufklarer/Qwen3-ASR-1.7B-MLX-5bit
VAD_REV=523876545a57961474fee9df913e833e130560b8        # aufklarer/Silero-VAD-v6.2.1-CoreML
rm -rf Model/asr Model/tokenizer
hf download aufklarer/Qwen3-ASR-1.7B-MLX-5bit config.json model.safetensors vocab.json merges.txt tokenizer_config.json \
  --revision $ASR_REV --local-dir Model/asr > /dev/null
# One --include per pattern: after the first, more words are read as file names.
hf download aufklarer/Silero-VAD-v6.2.1-CoreML --revision $VAD_REV --local-dir Model/vad --include config.json \
  --include "silero_vad.mlmodelc/*" > /dev/null
rm -rf Model/*/.cache
du -sh Model/*
