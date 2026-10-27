# Cantonese live

Live transcription of Cantonese (mixed with Mandarin and English), as spoken, in Hong Kong Traditional characters.
Nothing is stored. The iOS app runs the model on the phone (Qwen3-ASR 1.7B on the GPU through MLX); the web page uses
Soniox (`stt-rt-v5`) through the relay.

On the web page network drops are invisible: it keeps the audio Soniox hasn't finalized, reconnects at once and replays
it, and splits a long backlog onto its own connection so live subtitles never lag (see `relay/src/index.js`). On desktop,
「彈出視窗」 moves the subtitles into a narrow always-on-top window (Document Picture-in-Picture in Chrome/Edge, a plain
popup elsewhere).

- `relay/`: a Cloudflare Worker at `cantonese.deyaochen.com` (behind Cloudflare Access) that serves the web page,
  relays 16 kHz PCM from the clients to Soniox over WebSockets (holding the keys), converts the text to Hong Kong
  Traditional with OpenCC.
- `ios/`: the native SwiftUI app (TestFlight, `dev.de0ch.cantonese`). It transcribes on the phone with no network:
  Silero VAD cuts the mic audio at pauses, each segment is transcribed by Qwen3-ASR 1.7B (5-bit,
  `aufklarer/Qwen3-ASR-1.7B-MLX-5bit`, downloaded from Hugging Face on the first start, about 2.3 GB) through
  speech-swift on the GPU, and converted to HK Traditional with OpenCC. iOS allows no GPU work in the background, so
  it works with the app on screen: the screen stays awake while listening and leaving the app stops it.
- `scripts/relay-test.mjs`: streams `testdata/cantonese-16k.wav` through the live relay.
- `scripts/cloudflare-access.sh`: the Access app (Deyao's email only).

CI (`.github/workflows/`): `relay` deploys the Worker with the keys from GitHub secrets and runs the live test. `ios`
runs every flow on an iPhone 17 simulator (a stand-in answers for the model there: the simulator has no Metal for
MLX), then cloud-signs and uploads to TestFlight. `mlx-check` transcribes the test clip with the real model on a Mac.

Secrets (GitHub Actions only, never in git): `CLOUDFLARE_API_TOKEN`, `SONIOX_API_KEY`, `ASC_KEY_ID`, `ASC_ISSUER_ID`,
`ASC_PRIVATE_KEY`.
