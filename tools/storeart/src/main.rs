//! The App Store art, made from the phone replica's output in CI (the appstore workflow):
//!
//! storeart stills  --light DIR --dark DIR --fonts DIR --out ios/fastlane/screenshots
//!     Screenshots (1320 x 2868 RGB) per locale: light grey, a black bold headline and a grey line on top, the
//!     simulator's screenshot in a black rounded iPhone-like frame with a soft shadow below.
//! storeart preview --recording demo.mov --marks demo-marks.txt --audio demo-16k.wav --fonts DIR --out ios/fastlane/previews
//!     The App Preview (886 x 1920, 30 fps, H.264, stereo AAC) per locale: the simulator's screen recording of the demo
//!     test, cut by its marks, on the same grey with a headline band that changes as the demo goes on, and the demo
//!     conversation's audio from the moment the app starts listening. ffmpeg does the video.
//!
//! The demo test prints "DEMO welcome|listening|end <unix time>"; the replica writes "RECORDING <unix time>" when the
//! screen recording starts.

use ab_glyph::{FontVec, PxScale};
use image::{imageops, Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_text_mut, text_size};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const BG: [u8; 3] = [242, 242, 247]; // iOS systemGroupedBackground
const HEAD: [u8; 3] = [0, 0, 0];
const SUB: [u8; 3] = [99, 99, 102];
const FRAME: [u8; 3] = [18, 18, 20];

/// Headline and sub-line of each screenshot, per App Store locale.
struct Copy {
    head: &'static str,
    sub: &'static str,
}

const LOCALES: [&str; 3] = ["zh-Hant", "en-GB", "en-US"];

/// 0: captions, 1: three languages, 2: on the phone, 3: the live (grey) line.
fn copy(locale: &str) -> [Copy; 4] {
    let recognised = if locale == "en-US" { "recognized" } else { "recognised" };
    match locale {
        "zh-Hant" => [
            Copy { head: "講廣東話，即刻出字幕", sub: "身邊嘅說話，即時變成香港繁體字幕" },
            Copy { head: "中英普，混埋講都得", sub: "廣東話、英文、普通話一齊認得" },
            Copy { head: "全部喺手機度運行", sub: "唔使上網，唔收集任何資料" },
            Copy { head: "一路講，一路出", sub: "未講完嗰句用灰色，講完即刻定稿" },
        ],
        _ => [
            Copy { head: "Speak Cantonese, see captions", sub: "What people around you say, as live Hong Kong Chinese captions" },
            Copy {
                head: "Cantonese, English, Mandarin",
                sub: if recognised == "recognized" {
                    "All three recognized, even mixed in one sentence"
                } else {
                    "All three recognised, even mixed in one sentence"
                },
            },
            Copy { head: "Runs entirely on your iPhone", sub: "Works offline. No data collected." },
            Copy { head: "Captions as people talk", sub: "The line still being spoken shows in grey" },
        ],
    }
}

struct Fonts {
    bold: FontVec,
    regular: FontVec,
}

fn load_fonts(dir: &Path) -> Fonts {
    let load = |n: &str| {
        let p = dir.join(n);
        FontVec::try_from_vec(std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())))
            .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
    };
    Fonts { bold: load("NotoSansHK-Bold.otf"), regular: load("NotoSansHK-Regular.otf") }
}

/// Splits text into lines no wider than `width`: at spaces for Latin text, anywhere (but not before CJK punctuation)
/// for Chinese.
fn wrap(font: &FontVec, size: f32, text: &str, width: u32) -> Vec<String> {
    let w = |s: &str| text_size(px(font, size), font, s).0;
    if w(text) <= width {
        return vec![text.to_string()];
    }
    // Two balanced lines of Latin text when they fit.
    if text.contains(' ') {
        let words: Vec<&str> = text.split(' ').collect();
        let best = (1..words.len())
            .map(|k| (words[..k].join(" "), words[k..].join(" ")))
            .min_by_key(|(a, b)| w(a).max(w(b)));
        if let Some((a, b)) = best {
            if w(&a).max(w(&b)) <= width {
                return vec![a, b];
            }
        }
    }
    let words: Vec<String> = if text.contains(' ') {
        text.split(' ').map(str::to_string).collect()
    } else {
        text.chars().map(|c| c.to_string()).collect()
    };
    let sep = if text.contains(' ') { " " } else { "" };
    let mut lines: Vec<String> = vec![];
    let mut cur = String::new();
    for word in words {
        let next = if cur.is_empty() { word.clone() } else { format!("{cur}{sep}{word}") };
        let punct = "，。、？！：；」』）".contains(word.as_str());
        if w(&next) > width && !cur.is_empty() && !punct {
            lines.push(std::mem::take(&mut cur));
            cur = word;
        } else {
            cur = next;
        }
    }
    lines.push(cur);
    lines
}

/// The em size `size` (as CSS or PIL take it) as ab_glyph's scale, which is the ascent-to-descent height.
fn px(font: &FontVec, size: f32) -> PxScale {
    use ab_glyph::Font;
    let em = font.units_per_em().unwrap_or(1000.0);
    PxScale::from(size * font.height_unscaled() / em)
}

/// Draws a centred block of text from `y` (the top of the first line's em box); returns the y below it.
fn draw_block(img: &mut RgbaImage, font: &FontVec, size: f32, text: &str, y: i32, width: u32, color: [u8; 3]) -> i32 {
    let lines = wrap(font, size, text, width);
    let line_h = (size * 1.3) as i32;
    let mut y = y;
    for l in &lines {
        let (tw, _) = text_size(px(font, size), font, l);
        let x = (img.width() as i32 - tw as i32) / 2;
        draw_text_mut(img, Rgba([color[0], color[1], color[2], 255]), x, y, px(font, size), font, l);
        y += line_h;
    }
    y
}

/// Coverage (0..1) of pixel (x, y) by a rounded rectangle, antialiased over one pixel.
fn rounded_cov(x: f32, y: f32, w: f32, h: f32, r: f32) -> f32 {
    let (px, py) = (x + 0.5, y + 0.5);
    let qx = (px - w / 2.0).abs() - (w / 2.0 - r);
    let qy = (py - h / 2.0).abs() - (h / 2.0 - r);
    let outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - r;
    (0.5 - outside).clamp(0.0, 1.0)
}

fn blend(dst: &mut Rgba<u8>, src: [u8; 3], a: f32) {
    for i in 0..3 {
        dst.0[i] = (dst.0[i] as f32 * (1.0 - a) + src[i] as f32 * a).round() as u8;
    }
}

/// A soft shadow of a rounded rectangle at (x, y, w, h): blurred at a quarter of the size, then scaled up.
fn shadow(img: &mut RgbaImage, x: i32, y: i32, w: u32, h: u32, r: f32, blur: f32, offset: i32, alpha: f32) {
    let q = 4u32;
    let (cw, ch) = (img.width() / q, img.height() / q);
    let mut m = image::GrayImage::new(cw, ch);
    for py in 0..ch {
        for px in 0..cw {
            let c = rounded_cov(
                (px * q) as f32 - x as f32,
                (py * q) as f32 - (y + offset) as f32,
                w as f32,
                h as f32,
                r,
            );
            m.put_pixel(px, py, image::Luma([(c * 255.0) as u8]));
        }
    }
    let m = imageproc::filter::gaussian_blur_f32(&m, blur / q as f32);
    let m = imageops::resize(&m, img.width(), img.height(), imageops::FilterType::Triangle);
    for (px, py, p) in img.enumerate_pixels_mut() {
        let a = m.get_pixel(px, py).0[0] as f32 / 255.0 * alpha;
        if a > 0.0 {
            blend(p, [0, 0, 0], a);
        }
    }
}

/// A screenshot inside a black iPhone-like frame, `width` px wide, with the shot's corners rounded.
fn phone(shot: &RgbImage, width: u32) -> RgbaImage {
    let bez = (width as f32 * 0.035).round() as u32;
    let sw = width - 2 * bez;
    let sh = (shot.height() as f32 * sw as f32 / shot.width() as f32).round() as u32;
    let shot = imageops::resize(shot, sw, sh, imageops::FilterType::Lanczos3);
    let (fw, fh) = (width, sh + 2 * bez);
    let (r_out, r_in) = (width as f32 * 0.155, width as f32 * 0.125);
    let mut out = RgbaImage::new(fw, fh);
    for (x, y, p) in out.enumerate_pixels_mut() {
        let a = rounded_cov(x as f32, y as f32, fw as f32, fh as f32, r_out);
        if a == 0.0 {
            continue;
        }
        let mut c = FRAME;
        if x >= bez && y >= bez && x < bez + sw && y < bez + sh {
            let ai = rounded_cov((x - bez) as f32, (y - bez) as f32, sw as f32, sh as f32, r_in);
            let s = shot.get_pixel(x - bez, y - bez).0;
            for i in 0..3 {
                c[i] = (FRAME[i] as f32 * (1.0 - ai) + s[i] as f32 * ai).round() as u8;
            }
        }
        *p = Rgba([c[0], c[1], c[2], (a * 255.0).round() as u8]);
    }
    out
}

fn over(dst: &mut RgbaImage, src: &RgbaImage, x: i32, y: i32) {
    for (sx, sy, p) in src.enumerate_pixels() {
        let (dx, dy) = (x + sx as i32, y + sy as i32);
        if dx < 0 || dy < 0 || dx >= dst.width() as i32 || dy >= dst.height() as i32 || p.0[3] == 0 {
            continue;
        }
        blend(dst.get_pixel_mut(dx as u32, dy as u32), [p.0[0], p.0[1], p.0[2]], p.0[3] as f32 / 255.0);
    }
}

fn flatten(img: &RgbaImage) -> RgbImage {
    RgbImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        Rgb([p[0], p[1], p[2]])
    })
}

/// The grey canvas with the headline band; returns it and the y where the band ends.
fn band(fonts: &Fonts, w: u32, h: u32, c: &Copy, scale: f32) -> (RgbaImage, i32) {
    let mut img = RgbaImage::from_pixel(w, h, Rgba([BG[0], BG[1], BG[2], 255]));
    let width = w - (110.0 * scale) as u32;
    // Latin headlines a little smaller than Chinese ones, which are 10 full-width characters at most.
    let head = if c.head.is_ascii() { 100.0 } else { 118.0 };
    let y = draw_block(&mut img, &fonts.bold, head * scale, c.head, (230.0 * scale) as i32, width, HEAD);
    let y = draw_block(&mut img, &fonts.regular, 58.0 * scale, c.sub, y + (36.0 * scale) as i32, width, SUB);
    (img, y)
}

fn still(fonts: &Fonts, c: &Copy, shot: &RgbImage) -> RgbImage {
    let (w, h) = (1320, 2868);
    let (mut img, text_end) = band(fonts, w, h, c, 1.0);
    // The phone is 1060 px wide (as in the approved mockup) unless the text needs more room; its bottom stays 52 px
    // above the edge.
    let top = (text_end + 110).max(600);
    let avail = (h as i32 - 52 - top) as f32;
    let ratio = shot.height() as f32 / shot.width() as f32;
    // The frame is width * (0.93 * ratio + 0.07) tall (3.5% bezels).
    let width = 1060f32.min(avail / (0.93 * ratio + 0.07)).floor() as u32;
    let fr = phone(shot, width);
    let x = (w as i32 - fr.width() as i32) / 2;
    shadow(&mut img, x, top, fr.width(), fr.height(), width as f32 * 0.155, 40.0, 30, 0.24);
    over(&mut img, &fr, x, top);
    flatten(&img)
}

fn find_shot(dir: &Path, suffix: &str) -> RgbImage {
    fn walk(d: &Path, suffix: &str) -> Option<PathBuf> {
        for e in std::fs::read_dir(d).ok()?.flatten() {
            let p = e.path();
            if p.is_dir() {
                if let Some(f) = walk(&p, suffix) {
                    return Some(f);
                }
            } else if p.file_name()?.to_str()?.ends_with(suffix) {
                return Some(p);
            }
        }
        None
    }
    let p = walk(dir, suffix).unwrap_or_else(|| panic!("no *{suffix} under {}", dir.display()));
    eprintln!("{suffix}: {}", p.display());
    image::open(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).to_rgb8()
}

fn stills(args: &HashMap<String, String>) {
    let fonts = load_fonts(Path::new(&args["fonts"]));
    let (light, dark) = (Path::new(&args["light"]), Path::new(&args["dark"]));
    let shots = [
        ("1-captions", find_shot(light, "-demo-captions.png")),
        ("2-languages", find_shot(dark, "-demo-captions.png")),
        ("3-on-device", find_shot(light, "-demo-welcome.png")),
        ("4-live", find_shot(light, "-demo-middle.png")),
    ];
    for locale in LOCALES {
        let dir = Path::new(&args["out"]).join(locale);
        std::fs::create_dir_all(&dir).unwrap();
        for (i, c) in copy(locale).iter().enumerate() {
            let p = dir.join(format!("{}.png", shots[i].0));
            still(&fonts, c, &shots[i].1).save(&p).unwrap();
            println!("{}", p.display());
        }
    }
}

/// Marks: RECORDING and DEMO welcome/listening/end, as unix times.
fn marks(path: &Path) -> HashMap<String, f64> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut m = HashMap::new();
    for l in text.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        match f.as_slice() {
            ["RECORDING", t] => {
                m.insert("recording".into(), t.parse().unwrap());
            }
            ["DEMO", what, t] => {
                if let Ok(t) = t.parse() {
                    m.insert(what.to_string(), t);
                }
            }
            _ => {}
        }
    }
    for k in ["recording", "welcome", "listening"] {
        assert!(m.contains_key(k), "no {k} mark in {}", path.display());
    }
    m
}

fn ffmpeg(args: &[String]) {
    eprintln!("ffmpeg {}", args.join(" "));
    let ok = Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-y"]).args(args).status().expect("ffmpeg");
    assert!(ok.success(), "ffmpeg failed");
}

fn preview(args: &HashMap<String, String>) {
    let fonts = load_fonts(Path::new(&args["fonts"]));
    let m = marks(Path::new(&args["marks"]));
    let (rec, welcome, listening) = (m["recording"], m["welcome"], m["listening"]);
    // The demo: the welcome sheet (half a second after it is up), the tap, then the 19.2 s conversation and its last
    // line being finished. 15–30 s in all (Apple's limits), kept under 29.5.
    let end = listening + 21.5;
    let start = (welcome - 0.5).max(end - 29.5);
    let dur = end - start;
    assert!(dur >= 15.0, "preview only {dur:.2} s");
    let at = start - rec; // into the recording
    assert!(at >= 0.0, "the recording started {:.2} s after the welcome sheet", -at);
    // The conversation's audio starts when streaming does (≈ the listening mark; the test sees 聽緊 within 0.2 s).
    let lead: f64 = args.get("lead").map(|s| s.parse().unwrap()).unwrap_or(0.2);
    let audio_at = (listening - start - lead).max(0.0);
    // Headline band: on the phone while the welcome sheet shows, captions once it listens, the three languages once
    // the second line (Central, proposal) is up.
    let cuts = [0.0, listening - start, listening - start + 7.5, dur];
    eprintln!("recording +{at:.2} s, {dur:.2} s long, audio at {audio_at:.2} s, cuts {cuts:?}");

    let (w, h) = (886u32, 1920u32);
    let scale = w as f32 / 1320.0;
    let tmp = std::env::temp_dir().join(format!("storeart-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    for locale in LOCALES {
        let c = copy(locale);
        let order = [&c[2], &c[0], &c[1]];
        let bands: Vec<(RgbaImage, i32)> = order.iter().map(|c| band(&fonts, w, h, c, scale)).collect();
        // The recording, scaled to fit under the tallest band, with rounded corners and a soft shadow (no device
        // frame: Apple doesn't allow one in previews).
        let top = bands.iter().map(|b| b.1).max().unwrap() + 60;
        let vh = ((h as i32 - 56 - top) / 2 * 2) as u32;
        let vw = ((vh as f32 * 1320.0 / 2868.0 / 2.0).round() * 2.0) as u32;
        let vx = ((w - vw) / 2) as i32;
        let r = vw as f32 * 0.12;
        let mut layers = vec![];
        for (i, (img, _)) in bands.into_iter().enumerate() {
            let mut img = img;
            shadow(&mut img, vx, top, vw, vh, r, 28.0, 20, 0.24);
            let p = tmp.join(format!("{locale}-band{i}.png"));
            flatten(&img).save(&p).unwrap();
            layers.push((p, img));
        }
        // The corners over the video: the background where the rounded rectangle doesn't cover.
        let bg = &layers[0].1;
        let corners = RgbaImage::from_fn(vw, vh, |x, y| {
            let a = 1.0 - rounded_cov(x as f32, y as f32, vw as f32, vh as f32, r);
            let p = bg.get_pixel(vx as u32 + x, top as u32 + y).0;
            Rgba([p[0], p[1], p[2], (a * 255.0).round() as u8])
        });
        let corners_p = tmp.join(format!("{locale}-corners.png"));
        corners.save(&corners_p).unwrap();

        let dir = Path::new(&args["out"]).join(locale);
        std::fs::create_dir_all(&dir).unwrap();
        let out = dir.join("1-demo-APP_IPHONE_67.mp4");
        let mut a: Vec<String> = vec!["-ss".into(), format!("{at:.3}"), "-t".into(), format!("{dur:.3}"), "-i".into(), args["recording"].clone()];
        for (i, (p, _)) in layers.iter().enumerate() {
            let d = cuts[i + 1] - cuts[i];
            a.extend(["-loop", "1", "-framerate", "30", "-t", &format!("{d:.3}"), "-i"].map(String::from));
            a.push(p.display().to_string());
        }
        a.extend(["-loop", "1", "-framerate", "30", "-i"].map(String::from));
        a.push(corners_p.display().to_string());
        a.extend(["-i".to_string(), args["audio"].clone()]);
        let ms = (audio_at * 1000.0).round() as i64;
        let filter = format!(
            "[1:v][2:v][3:v]concat=n=3:v=1:a=0,fps=30,format=rgb24[bg];\
             [0:v]setpts=PTS-STARTPTS,fps=30,scale={vw}:{vh}:flags=lanczos[rec];\
             [bg][rec]overlay={vx}:{top}:eof_action=repeat[v1];\
             [v1][4:v]overlay={vx}:{top}:shortest=1,trim=duration={dur:.3},format=yuv420p[v];\
             [5:a]aformat=channel_layouts=stereo,aresample=44100,adelay={ms}:all=1,apad,atrim=duration={dur:.3}[a]"
        );
        a.extend(["-filter_complex", &filter, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-profile:v", "high", "-level", "4.0", "-preset", "slow", "-b:v", "10M", "-maxrate", "12M",
            "-bufsize", "20M", "-r", "30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", "44100", "-ac", "2", "-movflags", "+faststart"].map(String::from));
        a.push(out.display().to_string());
        ffmpeg(&a);
        println!("{}", out.display());
    }
    std::fs::remove_dir_all(&tmp).ok();
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cmd = argv.first().cloned().unwrap_or_default();
    let mut args = HashMap::new();
    for kv in argv[1.min(argv.len())..].chunks(2) {
        if let [k, v] = kv {
            args.insert(k.trim_start_matches("--").to_string(), v.clone());
        }
    }
    let need = |keys: &[&str]| {
        for k in keys {
            assert!(args.contains_key(*k), "missing --{k}");
        }
    };
    match cmd.as_str() {
        "stills" => {
            need(&["light", "dark", "fonts", "out"]);
            stills(&args)
        }
        "preview" => {
            need(&["recording", "marks", "audio", "fonts", "out"]);
            preview(&args)
        }
        _ => {
            eprintln!("usage: storeart stills --light DIR --dark DIR --fonts DIR --out DIR\n       storeart preview --recording MOV --marks TXT --audio WAV --fonts DIR --out DIR [--lead S]");
            std::process::exit(2);
        }
    }
}
