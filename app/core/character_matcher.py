"""
Identifies WHICH known character a detected face in a video frame belongs
to, so that in a video with multiple people, each detected face gets swapped
with the correct reference character (Person A -> Reference X, Person B ->
Reference Y).

Classical approach (no deep-learning face embeddings):
  - OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer.
    This is a classical texture-histogram algorithm, not a neural network.
  - Trained on-the-fly from each character's multiple reference angle
    images, so the recognizer learns to associate several poses of the
    same person with one label.

If `cv2.face` (opencv-contrib) is unavailable, a simple histogram-correlation
fallback is used instead so the app still functions, at reduced accuracy.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.face_detector import FaceDetector, DetectedFace
from app.core.reference_manager import Character

_LBPH_SIZE = (200, 200)


@dataclass
class MatchResult:
    character_id: Optional[str]
    confidence: float  # lower distance = better match for LBPH; normalized score for fallback


class CharacterMatcher:
    """Trains a classical recognizer over all characters' reference images
    and matches new faces (from video frames) to the closest character."""

    def __init__(self, face_detector: Optional[FaceDetector] = None,
                 confidence_threshold: float = 85.0):
        self.face_detector = face_detector or FaceDetector()
        self.confidence_threshold = confidence_threshold
        self._label_to_id: Dict[int, str] = {}
        self._recognizer = None
        self._fallback_templates: Dict[str, List[np.ndarray]] = {}
        self._has_lbph = hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")

    @staticmethod
    def _crop_aligned(image_bgr: np.ndarray, face: DetectedFace) -> Optional[np.ndarray]:
        x, y, w, h = face.rect
        if w <= 0 or h <= 0:
            return None
        crop = image_bgr[max(y, 0):y + h, max(x, 0):x + w]
        if crop.size == 0:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, _LBPH_SIZE)
        gray = cv2.equalizeHist(gray)
        return gray

    def train(self, characters: List[Character]) -> None:
        """(Re)train the recognizer from all reference images of all
        provided characters. Call this whenever characters/references
        change, before running matching on a video."""
        samples: List[np.ndarray] = []
        labels: List[int] = []
        self._label_to_id.clear()
        self._fallback_templates.clear()

        next_label = 0
        for character in characters:
            label = next_label
            next_label += 1
            self._label_to_id[label] = character.character_id
            self._fallback_templates.setdefault(character.character_id, [])

            for img in character.load_images():
                faces = self.face_detector.detect(img, max_faces=1)
                if faces:
                    face_img = self._crop_aligned(img, faces[0])
                else:
                    # No face detected in reference (unusual); use whole image.
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_img = cv2.resize(gray, _LBPH_SIZE)
                if face_img is None:
                    continue
                samples.append(face_img)
                labels.append(label)
                self._fallback_templates[character.character_id].append(face_img)

        if not samples:
            self._recognizer = None
            return

        if self._has_lbph:
            self._recognizer = cv2.face.LBPHFaceRecognizer_create()
            self._recognizer.train(samples, np.array(labels))
        else:  # pragma: no cover - fallback when opencv-contrib not present
            self._recognizer = None

    def match(self, frame_bgr: np.ndarray, face: DetectedFace) -> MatchResult:
        """Return the best-matching character_id for a face detected in a
        video frame, using the trained classical recognizer."""
        face_img = self._crop_aligned(frame_bgr, face)
        if face_img is None:
            return MatchResult(character_id=None, confidence=0.0)

        if self._has_lbph and self._recognizer is not None:
            label, distance = self._recognizer.predict(face_img)
            char_id = self._label_to_id.get(label)
            if distance > self.confidence_threshold:
                return MatchResult(character_id=None, confidence=distance)
            return MatchResult(character_id=char_id, confidence=distance)

        # Fallback: normalized correlation against stored templates.
        best_id, best_score = None, -1.0
        for char_id, templates in self._fallback_templates.items():
            for tmpl in templates:
                score = cv2.matchTemplate(face_img, tmpl, cv2.TM_CCOEFF_NORMED)[0][0]
                if score > best_score:
                    best_score = score
                    best_id = char_id
        if best_score < 0.35:  # weak match threshold
            return MatchResult(character_id=None, confidence=best_score)
        return MatchResult(character_id=best_id, confidence=best_score)

    def match_all(self, frame_bgr: np.ndarray,
                   faces: List[DetectedFace]) -> List[Tuple[DetectedFace, MatchResult]]:
        return [(face, self.match(frame_bgr, face)) for face in faces]
