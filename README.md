# Cantonese live

Live transcription of Cantonese (mixed with Mandarin and English), as spoken, in Hong Kong Traditional characters.
Nothing is stored. The iOS app runs the model on the phone (Qwen3-ASR 1.7B on CoreML, Neural Engine); the web page
uses Soniox (`stt-rt-v5`) through the relay.

On the web page network drops are invisible: it keeps the audio Soniox hasn't finalized, reconnects at once and replays
it, and splits a long backlog onto its own connection so live subtitles never lag (see `relay/src/index.js`). On desktop,
「彈出視窗」 moves the subtitles into a narrow always-on-top window (Document Picture-in-Picture in Chrome/Edge, a plain
popup elsewhere).

- `relay/`: a Cloudflare Worker at `cantonese.deyaochen.com` (behind Cloudflare Access) that serves the web page,
  relays 16 kHz PCM from the clients to Soniox over WebSockets (holding the keys), converts the text to Hong Kong
  Traditional with OpenCC.
- `ios/`: the native SwiftUI app (TestFlight, `dev.de0ch.cantonese`). It transcribes on the phone with no network:
  Silero VAD cuts the mic audio at pauses, each segment is transcribed by Qwen3-ASR 1.7B through speech-swift's
  CoreML runner (Neural Engine and CPU only, so it keeps going with the screen locked) and converted to HK Traditional
  with OpenCC. The first start downloads the model (about 2.5 GB) from the Hugging Face repo
  `DE0CH/Qwen3-ASR-1.7B-CoreML` and compiles it on the phone.
- `coreml/convert.py`: converts Qwen3-ASR to that CoreML layout (encoder, token embedding, decoder in two halves with
  their KV caches as CoreML states, int8 weights). `--check clip.wav` first runs the exported graphs through the
  runner's own transcription loop next to the official `qwen_asr` output (`--fp16` in half precision, as the Neural
  Engine computes).
- `scripts/relay-test.mjs`: streams `testdata/cantonese-16k.wav` through the live relay.
- `scripts/cloudflare-access.sh`: the Access app (Deyao's email only).

CI (`.github/workflows/`): `relay` deploys the Worker with the keys from GitHub secrets and runs the live test. `ios`
runs every flow on an iPhone 17 simulator with the real model (downloaded, run on the simulator's CPU), then
cloud-signs and uploads to TestFlight.

Secrets (GitHub Actions only, never in git): `CLOUDFLARE_API_TOKEN`, `SONIOX_API_KEY`, `ASC_KEY_ID`, `ASC_ISSUER_ID`,
`ASC_PRIVATE_KEY`.
