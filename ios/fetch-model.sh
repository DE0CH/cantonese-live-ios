#!/bin/bash
# Fetches the models the app ships with into ios/Model (not in git), at pinned Hugging Face revisions:
#   asr/      Qwen3-ASR 1.7B 5-bit MLX (model.safetensors, config.json, tokenizer files), about 2.4 GB
#   aligner/  Qwen3-ForcedAligner 0.6B 5-bit MLX (word times for the sliding window)
# CI runs it before xcodegen; run it once before building locally. Needs the `hf` CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"
ASR_REV=1645ca9ce8767dac9a4a047e9de835950c42596e        # aufklarer/Qwen3-ASR-1.7B-MLX-5bit
ALIGNER_REV=ade7f8066f66e5b207dbb0deb45e61af780a0d1f    # aufklarer/Qwen3-ForcedAligner-0.6B-5bit
rm -rf Model/asr Model/aligner Model/vad Model/tokenizer
hf download aufklarer/Qwen3-ASR-1.7B-MLX-5bit config.json model.safetensors vocab.json merges.txt tokenizer_config.json \
  --revision $ASR_REV --local-dir Model/asr > /dev/null
hf download aufklarer/Qwen3-ForcedAligner-0.6B-5bit config.json model.safetensors vocab.json merges.txt \
  tokenizer_config.json quantize_config.json --revision $ALIGNER_REV --local-dir Model/aligner > /dev/null
rm -rf Model/*/.cache
du -sh Model/*
