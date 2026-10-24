# Cantonese live

Live transcription of Cantonese (mixed with Mandarin and English) by Soniox (`stt-rt-v5`), as spoken, in Hong Kong
Traditional characters. Nothing is stored.

- `relay/`: a Cloudflare Worker at `cantonese.deyaochen.com` (behind Cloudflare Access) that serves the web page,
  relays 16 kHz PCM from the clients to Soniox over WebSockets (holding the keys), converts the text to Hong Kong
  Traditional with OpenCC.
- `ios/`: the native SwiftUI app (TestFlight, `dev.de0ch.cantonese`). It streams the mic, keeps going with the
  screen locked, and passes Access with a service token.
- `scripts/relay-test.mjs`: streams `testdata/cantonese-16k.wav` through the live relay.
- `scripts/cloudflare-access.sh`: the Access app and the iOS service token (one-time; `ROTATE=1` rotates the token).

CI (`.github/workflows/`): `relay` deploys the Worker with the keys from GitHub secrets and runs the live test. `ios`
runs every flow on an iPhone 17 simulator against the live relay, then cloud-signs and uploads to TestFlight.

Secrets (GitHub Actions only, never in git): `CLOUDFLARE_API_TOKEN`, `SONIOX_API_KEY`,
`CF_ACCESS_CLIENT_ID`, `CF_ACCESS_CLIENT_SECRET`, `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_PRIVATE_KEY`.
