# App Store submission (loaded by the `appstore` workflow; Deyao submits)

App record 6826497735, bundle `dev.de0ch.cantonese`, version 1.0, the newest TestFlight build.

## App information

| Field | Value |
| --- | --- |
| Name (30) | 廣東話字幕 Cantonese Live Captions |
| Subtitle (30), zh-Hant | 即時廣東話字幕，全程離線 |
| Subtitle (30), en | Cantonese captions, offline |
| Primary language | Chinese (Traditional) |
| Localizations | Chinese (Traditional), English (U.K.), English (U.S.) |
| Category | Utilities; secondary Productivity |
| Price | Free, no in-app purchases |
| Availability | Every territory except China mainland (an app sold there needs an MIIT app filing) |
| Age rating | 4+ (every questionnaire answer "None"/"No") |
| Copyright | 2026 Deyao Chen |
| Privacy policy URL | https://github.com/DE0CH/cantonese-live/blob/main/PRIVACY.md |
| Support URL | https://github.com/DE0CH/cantonese-live/issues |
| App privacy | Data Not Collected |
| Encryption | None (`ITSAppUsesNonExemptEncryption` = false) |
| Content rights | Contains third-party content: yes (Qwen3-ASR, Apache 2.0; Silero VAD, MIT) |
| Sign-in for review | Not needed |

## Listing text

Description, keywords, promotional text, subtitle and name per localization live in `ios/fastlane/metadata/` (what the
`appstore` workflow loads): a plain description and the features as bullet points, nothing about the developer.

## Review notes

The app transcribes speech on the device with Qwen3-ASR 1.7B (MLX, on the GPU) and makes no network requests. Tap the
microphone button and speak Cantonese (or play a Cantonese video near the phone); captions appear after the speaker
pauses, the line being spoken refreshes about every 1.5 s. Loading the model takes a few seconds on the first tap.
iOS does not allow GPU work in the background, so leaving the app stops listening. No account is needed.

## Devices

Only iPhones with 8 GB of memory: `UIRequiredDeviceCapabilities` `iphone-performance-gaming-tier` (iPhone 15 Pro and
later, every iPhone 16 and 17). The model needs about 2.4 GB, and an app that may run out of memory on older phones is
not offered there at all (Deyao, 2026-10-28). Apple does not let a later update drop a required capability's users the
other way round, so this floor can only be lowered, not raised, after release.

## Screenshots

6.9-inch iPhone (1320 × 2868), the one size required, from the `ios` workflow run by hand with `device` = iPhone 17
Pro Max (light and dark, `upload` off): the UI test's shots while streaming the bundled clip (finished captions in light
and dark, the welcome sheet, 關於).

## Licences

The 關於 sheet (ⓘ, top left) lists every open-source part shipped (Qwen3-ASR and speech-swift, swift-transformers,
OpenCC: Apache 2.0; Silero VAD, MLX, MLX Swift, SwiftyOpenCC: MIT) with its licence text (`ios/App/Licences/`).
