#!/bin/bash
# Fetches the model the app ships with into ios/Model (not in git), at a pinned Hugging Face revision:
#   asr/  Qwen3-ASR 1.7B 5-bit MLX (model.safetensors, config.json, tokenizer files), about 2.4 GB
# CI runs it before xcodegen; run it once before building locally. Needs the `hf` CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"
ASR_REV=1645ca9ce8767dac9a4a047e9de835950c42596e        # aufklarer/Qwen3-ASR-1.7B-MLX-5bit
rm -rf Model/asr Model/vad Model/tokenizer
hf download aufklarer/Qwen3-ASR-1.7B-MLX-5bit config.json model.safetensors vocab.json merges.txt tokenizer_config.json \
  --revision $ASR_REV --local-dir Model/asr > /dev/null
rm -rf Model/*/.cache
du -sh Model/*
