"""
Central configuration and path constants for FaceSwap Studio.
"""
import os
import sys


def get_base_dir() -> str:
    """Return the directory of the running executable/script.

    Used for writable, user-visible data (characters, temp files) so it
    lives next to the .exe -- NOT for bundled read-only resources (see
    get_bundle_dir below), since PyInstaller --onefile extracts those to a
    separate temporary directory, not next to the .exe itself."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # app/utils/config.py -> app/utils -> app -> project root
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_bundle_dir() -> str:
    """Return the directory containing bundled read-only resources (the
    landmark model, ffmpeg.exe). In a PyInstaller --onefile build these are
    unpacked at startup into sys._MEIPASS; in a --onedir build or when
    running from source, they sit next to the executable/script."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return meipass
    return get_base_dir()


BASE_DIR = get_base_dir()
BUNDLE_DIR = get_bundle_dir()
MODELS_DIR = os.path.join(BUNDLE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
CHARACTERS_DIR = os.path.join(DATA_DIR, "characters")
PROJECTS_DIR = os.path.join(DATA_DIR, "projects")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Classical, non-deep-generative landmark predictor (68-point ERT model,
# Kazemi & Sullivan 2014 -- this is a regression-tree ensemble, not a
# generative/deep neural face-swap model).
LANDMARK_MODEL_FILENAME = "shape_predictor_68_face_landmarks.dat"
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, LANDMARK_MODEL_FILENAME)

# 68-point landmark index groups (standard dlib/iBUG 300-W layout)
JAW_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_POINTS = list(range(48, 68))

# Points used to build the convex-hull / triangulation region that gets
# warped and blended onto the target face (everything except the outer jaw
# gives the most stable blend; jaw is included for full-face coverage).
FACE_SWAP_POINTS = JAW_POINTS + list(range(17, 68))

for _d in (DATA_DIR, CHARACTERS_DIR, PROJECTS_DIR, TEMP_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)


def get_ffmpeg_path() -> str:
    """Locate a usable ffmpeg executable, preferring a copy bundled
    alongside the packaged application (so the end user never has to
    install ffmpeg separately) and falling back to whatever is on PATH.

    Search order:
      1. The bundle directory (sys._MEIPASS for a PyInstaller --onefile
         build where --add-binary resources are unpacked at runtime; the
         executable's own directory for --onedir builds).
      2. Next to this source file's project root (useful when running from
         source with ffmpeg.exe dropped into the repo root).
      3. Plain "ffmpeg" / "ffmpeg.exe", relying on the system PATH.
    """
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"

    candidates = [
        os.path.join(BUNDLE_DIR, exe_name),   # PyInstaller --onefile/--onedir bundle
        os.path.join(BASE_DIR, exe_name),     # next to the .exe, or project root when run from source
    ]

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return exe_name  # fall back to relying on PATH
