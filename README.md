# FaceSwap Studio

A **Windows desktop application** for face swapping in video clips using
**classical computer vision** (no AI / deep-learning / generative face-swap
models), with an optional voice-change feature using classical audio
signal processing.

## What "no AI" means here

| Feature | Technique used | AI/ML model involved? |
|---|---|---|
| Face detection | dlib HOG + Linear SVM detector (fallback: OpenCV Haar Cascade) | Small classical detector, not generative AI |
| Facial landmarks | dlib 68-point Ensemble-of-Regression-Trees predictor (Kazemi & Sullivan, 2014) | Shape-regression algorithm, not a face-generation model |
| Character identification (who is who, across multiple people/angles) | OpenCV LBPH (Local Binary Patterns Histograms) recognizer | Classical texture-histogram method, not a neural embedding |
| Face swap | Delaunay triangulation + affine warp per triangle + `cv2.seamlessClone` (Poisson blending) | Pure geometric warping + classical blending -- **no GAN/diffusion/autoencoder** |
| Voice change | Phase-vocoder pitch shift + resample-based formant shift (librosa) | Classical DSP -- **no AI voice cloning/conversion** |

This means results will look/sound different from tools like DeepFaceLab,
roop, or RVC -- those use deep generative models for higher realism. This
app trades some realism for being fully offline, lightweight, fast, and
explainable, using only 20-30-year-old, well-understood CV/DSP algorithms.

## Download the ready-to-run .exe (no setup required)

Every push to `main` is automatically built into a single, self-contained
`FaceSwapStudio.exe` by GitHub Actions (Windows runner) — it bundles all
Python dependencies, the facial-landmark model, and ffmpeg.exe. **No
Python, no `pip install`, no separate ffmpeg install needed.**

1. Go to the **[Releases page](../../releases/tag/latest-build)** of this
   repository (or check the **Actions** tab -> latest successful
   "Build Windows EXE" run -> Artifacts, if the release isn't visible yet).
2. Download `FaceSwapStudio.exe`.
3. Double-click it to run. Windows SmartScreen may show an "Unknown
   publisher" warning the first time (this app isn't code-signed) --
   click "More info" -> "Run anyway".

That's it -- the app window should open with 3 tabs (Characters, Import
Video & Face Swap, Voice Change).

> **First-run size/startup note:** because everything is bundled into one
> file, the .exe is large (150-250 MB) and takes a few seconds to unpack
> to a temp folder on first launch each time you run it -- this is normal
> for PyInstaller `--onefile` builds.

## Features

- Import a video (short or long clip).
- Create multiple "characters," each with multiple reference images from
  different angles (front, left, right, up, down, profile, etc.).
- Automatically detect faces in the video, identify which known character
  each belongs to (supports multiple different people in one video, each
  mapped to their own reference), and swap in the geometrically closest
  matching reference angle for the character.
- Optional voice change:
  - Manual pitch (semitones) + formant ratio adjustment.
  - "Match target voice" mode: analyze a short sample of a target voice
    and shift the source audio's average pitch/tone toward it
    (approximation only -- not true voice cloning).
- Export the final video (face-swapped, with either original or
  voice-changed audio).

## Running from source (only needed if you want to modify the code)

If you just want to use the app, download the `.exe` above and skip this
section entirely.

### Requirements

- Windows 10/11 (developed for Windows; runs on Linux/macOS too with the
  same dependencies).
- Python 3.10 or 3.11 recommended.
- [ffmpeg](https://ffmpeg.org/download.html) available on PATH (needed for
  audio extraction/muxing). Download the Windows build and add its `bin`
  folder to PATH, or place `ffmpeg.exe` next to the app.

### Setup

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the classical facial-landmark model (one-time)
python scripts/download_landmark_model.py

# 4. Run the app
python main.py
```

> **Note on dlib:** `requirements.txt` uses `dlib-bin`, a prebuilt-wheel
> distribution of dlib (Windows/Linux/macOS) -- no C++ compiler needed, and
> `import dlib` works identically to the regular `dlib` package. If you
> want to build classic `dlib` from source instead and hit a CMake error
> like `Compatibility with CMake < 3.5 has been removed` (happens with
> newer CMake, 3.20+), set this before installing:
> ```
> set CMAKE_ARGS=-DCMAKE_POLICY_VERSION_MINIMUM=3.5
> pip install dlib==19.24.2
> ```

## Using the app

1. **Characters & References tab**
   - Click "+ Add Character" and name them (e.g. "Hero", "Villain").
   - Select a character, choose an angle from the dropdown (front/left/
     right/up/down/etc.), and click "+ Add Reference Image" to add one or
     more images for that angle. Add several angles per character for best
     results when the video's face turns/tilts.

2. **Import Video & Face Swap tab**
   - Browse and select your video clip.
   - Check the characters you want swapped into the video (each detected
     face in the video is automatically matched to the closest-known
     character via the LBPH recognizer trained on your reference images).
   - Click "Start Face Swap Processing" and wait for the progress bar to
     complete.
   - Click "Save Output Video As..." to export.

3. **Voice Change tab (optional)**
   - Choose a source video/audio (usually the same clip you're processing).
   - Pick "Manual pitch/formant shift" and set values, OR "Match target
     voice" and select a short sample of the target voice.
   - Click "Apply Voice Change". The result is automatically wired in as
     the audio track for the next export from the Process tab.

## Project structure

```
FaceSwapStudio/
├── main.py                        # entry point
├── requirements.txt
├── models/                        # place shape_predictor_68_face_landmarks.dat here
├── data/
│   ├── characters/                # saved characters + reference images (auto-created)
│   └── temp/                      # intermediate processing files (auto-created)
├── scripts/
│   ├── download_landmark_model.py
│   └── build_windows.bat          # PyInstaller packaging script
└── app/
    ├── core/
    │   ├── face_detector.py       # detection + 68-point landmarks
    │   ├── character_matcher.py   # LBPH-based identity matching
    │   ├── reference_manager.py   # character/reference image storage
    │   ├── face_swapper.py        # Delaunay warp + seamless blend
    │   └── video_processor.py     # frame loop + ffmpeg mux
    ├── audio/
    │   └── voice_changer.py       # pitch/formant shift (librosa)
    ├── gui/
    │   ├── main_window.py
    │   ├── workers.py             # QThread background workers
    │   └── widgets/
    │       ├── character_panel.py
    │       ├── process_panel.py
    │       └── voice_panel.py
    └── utils/
        └── config.py              # paths & landmark index constants
```

## Packaging as a standalone .exe

**Automatic (recommended):** every push to `main` triggers
`.github/workflows/build-windows-exe.yml`, which runs on a real Windows
GitHub Actions runner and produces a fully self-contained
`FaceSwapStudio.exe` (dependencies + landmark model + ffmpeg.exe all
bundled), published to the [latest-build release](../../releases/tag/latest-build).
You can also trigger it manually from the **Actions** tab -> "Build
Windows EXE" -> "Run workflow".

**Manual (on your own Windows machine):**
```bash
scripts\build_windows.bat
```
This downloads the landmark model if missing, and bundles `ffmpeg.exe`
(place it in the project root first -- see the script's own instructions)
plus all Python dependencies into `dist\FaceSwapStudio.exe`.

## Known limitations (inherent to the classical, non-AI approach)

- Extreme head turns/occlusions (e.g. profile > ~60°, hand over face) may
  fail to detect landmarks reliably -- add more angle references to help.
- Blending quality depends on similarity of lighting/skin tone between the
  reference image and the video; very different lighting can produce a
  visible seam despite Poisson blending.
- "Match target voice" only approximates average pitch/tone, not full
  voice identity/timbre -- true voice cloning requires AI models, which are
  intentionally excluded from this project.
- Processing speed is CPU-bound (no GPU/deep-learning acceleration needed),
  so very long/high-resolution videos will take proportionally longer.
