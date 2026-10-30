#!/bin/bash
# Fetches the models the app ships with into ios/Model (not in git), pinned by revision or hash:
#   asr/              Qwen3-ASR 1.7B 5-bit MLX (model.safetensors, config.json, tokenizer files), about 2.4 GB: black text
#   sensevoice/       SenseVoice-Small int8 for sherpa-onnx (model.int8.onnx, tokens.txt), 237 MB: grey text
#   silero_vad.onnx   Silero speech detection for sherpa-onnx, 0.6 MB
# CI runs it before xcodegen; run it once before building locally. Needs the `hf` CLI (pip install huggingface_hub).
set -euo pipefail
cd "$(dirname "$0")"
ASR_REV=1645ca9ce8767dac9a4a047e9de835950c42596e        # aufklarer/Qwen3-ASR-1.7B-MLX-5bit
SHERPA=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
SV=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09
SV_SHA=7305f7905bfcf77fa0b39388a313f3da35c68d971661a65475b56fb2162c8e63
VAD_SHA=9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6
rm -rf Model/asr Model/aligner Model/sensevoice Model/silero_vad.onnx
mkdir -p Model/sensevoice
hf download aufklarer/Qwen3-ASR-1.7B-MLX-5bit config.json model.safetensors vocab.json merges.txt tokenizer_config.json \
  --revision $ASR_REV --local-dir Model/asr > /dev/null
rm -rf Model/*/.cache
curl -sSfL "$SHERPA/silero_vad.onnx" -o Model/silero_vad.onnx
echo "$VAD_SHA  Model/silero_vad.onnx" | shasum -a 256 -c - > /dev/null
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
curl -sSfL "$SHERPA/$SV.tar.bz2" -o "$TMP/sv.tar.bz2"
echo "$SV_SHA  $TMP/sv.tar.bz2" | shasum -a 256 -c - > /dev/null
tar xjf "$TMP/sv.tar.bz2" -C "$TMP"
mv "$TMP/$SV/model.int8.onnx" "$TMP/$SV/tokens.txt" Model/sensevoice/
du -sh Model/*
