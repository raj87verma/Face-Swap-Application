"""
Ties together face detection, multi-angle reference selection, and the
face-swap engine to process an entire video frame by frame, then re-muxes
the processed video with the original (or voice-changed) audio using
ffmpeg.

Face-to-character assignment strategy
--------------------------------------
A reference image is the face you want to swap IN -- it does not depict
the person currently in the video, so it cannot be used to "recognize"
who in the video is who (that would require matching the video person's
own face, which we don't have a photo of). Because of that:

  - If exactly one character is enabled, every detected face in every
    frame is swapped with that character's reference (no identity
    decision needed -- there is only one possible target).
  - If multiple characters are enabled, faces are assigned to characters
    by their left-to-right horizontal position in the frame, matched
    against the character list's order. This is a simple, deterministic,
    classical-CV-friendly heuristic; it assumes people stay in roughly
    the same left-right order throughout the clip. It will misassign
    faces if people cross paths or if the number of detected faces
    doesn't match the number of enabled characters in a given frame.
"""
import os
import subprocess
import shutil
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from app.core.face_detector import FaceDetector
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

    @staticmethod
    def _assign_faces_to_characters(faces, characters: List[Character]):
        """Decide which detected face (in the current frame) gets swapped
        with which character's reference face.

        - Exactly one character enabled: swap every detected face with it
          (no identity decision to make).
        - Multiple characters enabled: sort faces left-to-right and pair
          them with characters in the order the user listed/checked them,
          pairing at most min(len(faces), len(characters)) of each.
        """
        if len(characters) == 1:
            return [(face, characters[0]) for face in faces]

        faces_sorted = sorted(faces, key=lambda f: f.rect[0])  # left-to-right by x
        return list(zip(faces_sorted, characters))

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

        ref_cache = self._prepare_reference_cache(enabled_characters)
        # Drop any enabled character whose reference images yielded no
        # usable face+landmarks (e.g. a blurry/occluded selfie) so it
        # doesn't silently block assignment; warn the caller via a plain
        # print for now (surfaced to the GUI status log by the caller).
        usable_characters = [
            c for c in enabled_characters
            if ref_cache.get(c.character_id, {}).get("landmarks")
        ]
        if not usable_characters:
            raise ValueError(
                "None of the enabled characters have a usable reference "
                "image (a face could not be detected in any of them). "
                "Use a clear, front-facing, well-lit photo.")

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

                faces = [f for f in self.face_detector.detect(frame) if f.landmarks is not None]
                if faces:
                    face_to_character = self._assign_faces_to_characters(faces, usable_characters)
                    for face, character in face_to_character:
                        cache_entry = ref_cache[character.character_id]
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
