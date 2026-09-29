"""
Classical face detection + 68-point landmark extraction.

Uses:
  - dlib's HOG + Linear SVM frontal face detector (classical CV, not a
    generative model), OR OpenCV's Haar Cascade as a lightweight fallback.
  - dlib's 68-point landmark predictor, an Ensemble-of-Regression-Trees
    (Kazemi & Sullivan, 2014). This is a shape-regression algorithm, NOT a
    deep generative / face-swap AI model -- it only locates facial points.

No deep-learning face generation/synthesis is used anywhere in this module.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from app.utils.config import LANDMARK_MODEL_PATH

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _DLIB_AVAILABLE = False


@dataclass
class DetectedFace:
    """A single detected face within a frame."""
    rect: tuple            # (x, y, w, h) bounding box in the frame
    landmarks: np.ndarray  # (68, 2) int array of facial landmark points
    descriptor: Optional[np.ndarray] = field(default=None)  # optional identity feature


class FaceDetector:
    """Detects faces and extracts 68-point landmarks using classical CV.

    Falls back to OpenCV's Haar Cascade detector if dlib is unavailable,
    but landmark extraction requires dlib's shape predictor model file.
    """

    def __init__(self, landmark_model_path: str = LANDMARK_MODEL_PATH,
                 upsample: int = 1):
        self.upsample = upsample
        self._dlib_available = _DLIB_AVAILABLE
        self._predictor = None
        self._detector = None
        self._haar = None

        if self._dlib_available:
            self._detector = dlib.get_frontal_face_detector()
            try:
                self._predictor = dlib.shape_predictor(landmark_model_path)
            except RuntimeError as exc:
                raise FileNotFoundError(
                    f"Landmark model not found at '{landmark_model_path}'. "
                    "Download 'shape_predictor_68_face_landmarks.dat' and "
                    "place it in the models/ directory (see README)."
                ) from exc
        else:  # pragma: no cover - fallback path
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)

    @property
    def has_landmark_support(self) -> bool:
        return self._predictor is not None

    def detect(self, frame_bgr: np.ndarray, max_faces: Optional[int] = None) -> List[DetectedFace]:
        """Detect all faces in a BGR frame and return landmarks for each."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._dlib_available:
            rects = self._detector(gray, self.upsample)
            faces = []
            for rect in rects:
                x, y = max(rect.left(), 0), max(rect.top(), 0)
                w, h = rect.width(), rect.height()
                shape = self._predictor(gray, rect)
                pts = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)
                faces.append(DetectedFace(rect=(x, y, w, h), landmarks=pts))
                if max_faces and len(faces) >= max_faces:
                    break
            return faces

        # Fallback: Haar cascade detection only, no landmarks available.
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        faces = []
        for (x, y, w, h) in boxes:
            faces.append(DetectedFace(rect=(int(x), int(y), int(w), int(h)), landmarks=None))
            if max_faces and len(faces) >= max_faces:
                break
        return faces

    def detect_largest(self, frame_bgr: np.ndarray) -> Optional[DetectedFace]:
        """Convenience: return only the largest face in the frame (by area)."""
        faces = self.detect(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda f: f.rect[2] * f.rect[3])
