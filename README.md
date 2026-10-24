# Cantonese live

Live transcription of Cantonese (mixed with Mandarin and English) by Soniox (`stt-rt-v5`), in Hong Kong Traditional
characters, with a switch between 原文 (as spoken) and 書面語 (each finished line rewritten into written Chinese by
`qwen/qwen3.7-plus` through OpenRouter; English stays English). Nothing is stored.

- `relay/`: a Cloudflare Worker at `cantonese.deyaochen.com` (behind Cloudflare Access) that serves the web page,
  relays 16 kHz PCM from the clients to Soniox over WebSockets (holding the keys), converts the text to Hong Kong
  Traditional with OpenCC and, in 書面語, sends a rewrite of each finished line.
- `ios/`: the native SwiftUI app (TestFlight, `dev.de0ch.cantonese`). It streams the mic, keeps going with the
  screen locked, and passes Access with a service token.
- `scripts/relay-test.mjs`: streams `testdata/cantonese-16k.wav` through the live relay in both styles.
- `scripts/cloudflare-access.sh`: the Access app and the iOS service token (one-time; `ROTATE=1` rotates the token).

CI (`.github/workflows/`): `relay` deploys the Worker with the keys from GitHub secrets and runs the live test. `ios`
runs every flow on an iPhone 17 simulator against the live relay, then cloud-signs and uploads to TestFlight.

Secrets (GitHub Actions only, never in git): `CLOUDFLARE_API_TOKEN`, `SONIOX_API_KEY`, `OPENROUTER_API_KEY`,
`CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`, `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_PRIVATE_KEY`.
