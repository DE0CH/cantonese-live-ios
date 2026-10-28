# App Store submission (draft, not submitted)

App record 6826497735, bundle `dev.de0ch.cantonese`, version 1.0, the newest TestFlight build.

## App information

| Field | Value |
| --- | --- |
| Name (30) | 粵語字幕 Cantonese Captions |
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

## Description (zh-Hant)

講廣東話，即刻出字幕。

粵語字幕將身邊嘅廣東話即時轉做香港繁體字幕，講到邊出到邊。廣東話夾英文、夾普通話都得，照講出嚟嘅樣寫，唔會改做書面語。

全部喺部 iPhone 度轉，唔使上網：聲音同文字都唔會離開部手機，冇帳戶、冇廣告、冇追蹤。

• 即時字幕：講緊嗰句先用灰色顯示，講完變做黑色定稿
• 廣東話、英文、普通話混埋講都認得
• 香港繁體字
• 完全離線，飛機上、冇網都用得
• 可以揀選、複製字幕
• 用 iPhone 自己個咪，藍牙耳筒照聽歌

適合聽唔清、聽唔明廣東話嘅人，開會、上堂、睇醫生、同長輩傾偈都用得。

用嗰陣 app 要開住喺畫面，螢幕會保持著住；轉去第二個 app 或者鎖機就會停。

## Description (en)

Live Cantonese captions, right on your iPhone.

粵語字幕 turns the Cantonese around you into Hong Kong Traditional Chinese captions as it is spoken. Cantonese mixed
with English or Mandarin works too, written the way it was said.

Everything runs on the phone with no internet: audio and text never leave it. No account, no ads, no tracking.

• Live captions: the sentence being spoken shows in grey, then settles in black
• Cantonese, English and Mandarin, even mixed in one sentence
• Hong Kong Traditional characters
• Fully offline, works on a plane
• Select and copy the captions
• Uses the iPhone's own microphone, so Bluetooth headphones keep playing music

For anyone who finds Cantonese hard to hear or follow: meetings, classes, doctor's visits, family dinners.

Keep the app on screen while it listens; the screen stays on. Switching apps or locking the phone stops it.

## Keywords (100)

- zh-Hant: `廣東話,粵語,字幕,聽寫,語音轉文字,即時字幕,香港,繁體,離線,聽障,轉錄,Cantonese`
- en: `cantonese,captions,subtitles,transcribe,speech to text,hong kong,offline,hearing,dictation,live`

## Promotional text (170)

- zh-Hant: 講廣東話即刻出字幕，全部喺部手機度轉，唔使上網，乜都唔儲存。
- en: Live Cantonese captions that run entirely on your iPhone. No internet, nothing stored.

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
Pro Max (light and dark, `upload` off): the UI test's shots while streaming the bundled clip (empty screen, listening,
finished captions, 關於).

## Licences

The 關於 sheet (ⓘ, top left) lists every open-source part shipped (Qwen3-ASR and speech-swift, swift-transformers,
OpenCC: Apache 2.0; Silero VAD, MLX, MLX Swift, SwiftyOpenCC: MIT) with its licence text (`ios/App/Licences/`).
