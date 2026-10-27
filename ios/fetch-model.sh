#!/bin/bash
# Fetches the models the app ships with into ios/Model (not in git), at pinned Hugging Face revisions:
#   asr/        Qwen3-ASR 0.6B int8 CoreML (encoder, embedding, two decoder halves, config.json)
#   tokenizer/  the official Qwen3-ASR vocab (the CoreML decoder answers with token ids)
#   vad/        Silero VAD v6.2.1 CoreML
# CI runs it before xcodegen; run it once before building locally. Needs the `hf` CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"
ASR_REV=8c6bc87b87856930b435550e94ce47de710ce4ed        # aufklarer/Qwen3-ASR-CoreML
TOK_REV=5eb144179a02acc5e5ba31e748d22b0cf3e303b0        # Qwen/Qwen3-ASR-0.6B
VAD_REV=523876545a57961474fee9df913e833e130560b8        # aufklarer/Silero-VAD-v6.2.1-CoreML
hf download aufklarer/Qwen3-ASR-CoreML --revision $ASR_REV --local-dir Model/asr --include config.json \
  "encoder.mlmodelc/*" "embedding.mlmodelc/*" "decoder_part1.mlmodelc/*" "decoder_part2.mlmodelc/*" > /dev/null
hf download Qwen/Qwen3-ASR-0.6B --revision $TOK_REV --local-dir Model/tokenizer --include vocab.json tokenizer_config.json > /dev/null
hf download aufklarer/Silero-VAD-v6.2.1-CoreML --revision $VAD_REV --local-dir Model/vad --include config.json "silero_vad.mlmodelc/*" > /dev/null
rm -rf Model/*/.cache
du -sh Model/*
