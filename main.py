"""
FaceSwap Studio - entry point.

A Windows desktop application for classical-CV face swapping (landmark
detection + Delaunay-triangulation warping + seamless blending -- no
deep-learning / generative AI face-swap models) with an optional voice
change feature using classical DSP (pitch/formant shifting).
"""
import os
import sys

# Ensure the project root is on sys.path so `app.*` imports work whether run
# via `python main.py` from any working directory, or frozen with PyInstaller.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import QApplication, QMessageBox

from app.utils.config import LANDMARK_MODEL_PATH
from app.gui.main_window import MainWindow


def _check_prerequisites() -> bool:
    """Warn (but don't necessarily block) if the required landmark model
    file is missing -- face detection/swap won't work without it."""
    if not os.path.isfile(LANDMARK_MODEL_PATH):
        QMessageBox.warning(
            None, "Missing Landmark Model",
            "The face landmark model file was not found:\n\n"
            f"{LANDMARK_MODEL_PATH}\n\n"
            "Download 'shape_predictor_68_face_landmarks.dat' (see README.md) "
            "and place it in the 'models' folder before processing videos. "
            "The app will still open, but face swap will fail until this is added."
        )
    return True


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FaceSwap Studio")

    _check_prerequisites()

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
