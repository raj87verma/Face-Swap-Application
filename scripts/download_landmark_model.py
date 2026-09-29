"""
Downloads and extracts the classical 68-point facial landmark model
(shape_predictor_68_face_landmarks.dat, Kazemi & Sullivan 2014
Ensemble-of-Regression-Trees predictor -- NOT a deep-learning model) into
the models/ directory.

Source: dlib.net official model files.
Run:  python scripts/download_landmark_model.py
"""
import bz2
import os
import sys
import urllib.request

URL = "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2"

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEST_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.isfile(DEST_PATH):
        print(f"Model already present at {DEST_PATH}")
        return

    print(f"Downloading {URL} ...")
    compressed_path = DEST_PATH + ".bz2"
    try:
        urllib.request.urlretrieve(URL, compressed_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Download failed: {exc}")
        print("Please manually download the file from the dlib-models GitHub "
              "repository and extract it into the models/ folder.")
        sys.exit(1)

    print("Extracting...")
    with bz2.BZ2File(compressed_path, "rb") as src, open(DEST_PATH, "wb") as dst:
        dst.write(src.read())
    os.remove(compressed_path)
    print(f"Model ready at {DEST_PATH}")


if __name__ == "__main__":
    main()
