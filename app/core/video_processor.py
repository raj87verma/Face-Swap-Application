"""
Ties together face detection, character matching, multi-angle reference
selection, and the face-swap engine to process an entire video frame by
frame, then re-muxes the processed video with the original (or
voice-changed) audio using ffmpeg.
"""
import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.face_detector import FaceDetector
from app.core.character_matcher import CharacterMatcher
from app.core.face_swapper import FaceSwapper
from app.core.reference_manager import Character
from app.utils.config import TEMP_DIR, get_ffmpeg_path

ProgressCallback = Optional[Callable[[int, int], None]]  # (current_frame, total_frames)


@dataclass
class CharacterAssignment:
    """Maps a Character to be used as the swap source; if
    `enabled` is False the matched faces for this character are left
    untouched in the output video."""
    character: Character
    enabled: bool = True


class VideoProcessor:
    def __init__(self, ffmpeg_path: Optional[str] = None):
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
        self.face_detector = FaceDetector()
        self.matcher = CharacterMatcher(face_detector=self.face_detector)
        self.swapper = FaceSwapper()

    # ------------------------------------------------------------------ #
    def _prepare_reference_cache(self, characters: List[Character]) -> Dict[str, dict]:
        """Pre-compute landmarks for every reference image of every
        character once, so per-frame processing only does the (cheap) pose
        matching + warp, not repeated face detection on references."""
        cache: Dict[str, dict] = {}
        for character in characters:
            images, landmarks = [], []
            for img in character.load_images():
                faces = self.face_detector.detect(img, max_faces=1)
                if faces and faces[0].landmarks is not None:
                    images.append(img)
                    landmarks.append(faces[0].landmarks)
            cache[character.character_id] = {"images": images, "landmarks": landmarks}
        return cache

    def process_video(self, input_video_path: str, output_video_path: str,
                       assignments: List[CharacterAssignment],
                       replacement_audio_path: Optional[str] = None,
                       progress_cb: ProgressCallback = None) -> str:
        """Process `input_video_path`, swapping faces per `assignments`, and
        write the result (with audio re-attached) to `output_video_path`.

        `replacement_audio_path`, if given, is used instead of the original
        video's audio track (e.g. output of the voice changer).
        """
        enabled_characters = [a.character for a in assignments if a.enabled]
        if not enabled_characters:
            raise ValueError("No enabled characters to swap in.")

        self.matcher.train(enabled_characters)
        ref_cache = self._prepare_reference_cache(enabled_characters)

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {input_video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        os.makedirs(TEMP_DIR, exist_ok=True)
        silent_video_path = os.path.join(TEMP_DIR, "_swapped_silent.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(silent_video_path, fourcc, fps, (width, height))

        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                faces = self.face_detector.detect(frame)
                if faces:
                    matches = self.matcher.match_all(frame, faces)
                    for face, result in matches:
                        if result.character_id is None or face.landmarks is None:
                            continue
                        cache_entry = ref_cache.get(result.character_id)
                        if not cache_entry or not cache_entry["landmarks"]:
                            continue
                        best_idx = self.swapper.pick_best_reference(
                            face.landmarks, cache_entry["landmarks"])
                        src_img = cache_entry["images"][best_idx]
                        src_landmarks = cache_entry["landmarks"][best_idx]
                        frame = self.swapper.swap(frame, face.landmarks, src_img, src_landmarks)

                writer.write(frame)
                if progress_cb:
                    progress_cb(frame_idx, total_frames)
        finally:
            cap.release()
            writer.release()

        self._mux_audio(silent_video_path, input_video_path, output_video_path,
                         replacement_audio_path)
        return output_video_path

    # ------------------------------------------------------------------ #
    def _mux_audio(self, silent_video_path: str, original_video_path: str,
                    output_video_path: str, replacement_audio_path: Optional[str]) -> None:
        """Combine the (silent) processed video frames with an audio track
        via ffmpeg: either the original video's audio, or a replacement
        (e.g. voice-changed) audio file."""
        if shutil.which(self.ffmpeg_path) is None:
            # No ffmpeg available -- ship the silent video as-is so the
            # pipeline still produces output the user can inspect.
            shutil.copy2(silent_video_path, output_video_path)
            return

        audio_source = replacement_audio_path or original_video_path
        cmd = [
            self.ffmpeg_path, "-y",
            "-i", silent_video_path,
            "-i", audio_source,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-shortest",
            output_video_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @staticmethod
    def extract_audio(video_path: str, output_audio_path: str,
                       ffmpeg_path: Optional[str] = None) -> Optional[str]:
        ffmpeg_path = ffmpeg_path or get_ffmpeg_path()
        """Extract the audio track of a video to a WAV file for the voice
        changer to process. Returns None if there is no audio or ffmpeg is
        unavailable."""
        if shutil.which(ffmpeg_path) is None:
            return None
        cmd = [ffmpeg_path, "-y", "-i", video_path, "-vn",
               "-acodec", "pcm_s16le", "-ar", "44100", output_audio_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0 or not os.path.isfile(output_audio_path):
            return None
        return output_audio_path
