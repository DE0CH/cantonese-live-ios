# Cantonese live

Live Cantonese (粵語) transcription in spoken-form Hong Kong Traditional characters, with a switch between two engines:
Soniox (`stt-rt-v5`) and Alibaba Model Studio Fun-ASR realtime. Nothing is stored.

- `relay/`: a Cloudflare Worker at `cantonese.deyaochen.com` (behind Cloudflare Access) that serves the web page and
  relays 16 kHz PCM from the clients to the chosen engine over WebSockets, holding the provider keys. Output is converted
  to Hong Kong Traditional with OpenCC.
- `ios/`: the native SwiftUI app (TestFlight, `dev.de0ch.cantonese`). It streams the mic, keeps going with the
  screen locked, and passes Access with a service token.
- `scripts/relay-test.mjs`: streams `testdata/cantonese-16k.wav` through the live relay for each engine.
- `scripts/cloudflare-access.sh`: the Access app and the iOS service token (one-time; `ROTATE=1` rotates the token).

CI (`.github/workflows/`): `relay` deploys the Worker with the provider keys from GitHub secrets and runs the live test.
`ios` runs every flow on an iPhone 17 simulator against the live relay, then cloud-signs and uploads to TestFlight.

Secrets (GitHub Actions only, never in git): `CLOUDFLARE_API_TOKEN`, `SONIOX_API_KEY`, `DASHSCOPE_API_KEY`,
`CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`, `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_PRIVATE_KEY`. Variable:
`FUNASR_WORKSPACE` (the Model Studio workspace id).
