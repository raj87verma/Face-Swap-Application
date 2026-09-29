"""
QThread-based background workers so long-running video/audio processing
never blocks the GUI event loop.
"""
from typing import List, Optional

from PyQt5.QtCore import QThread, pyqtSignal

from app.core.video_processor import VideoProcessor, CharacterAssignment
from app.audio.voice_changer import VoiceChanger
from app.utils.config import get_ffmpeg_path


class VideoProcessingWorker(QThread):
    progress = pyqtSignal(int, int)   # current_frame, total_frames
    finished_ok = pyqtSignal(str)     # output_video_path
    failed = pyqtSignal(str)          # error message

    def __init__(self, input_video_path: str, output_video_path: str,
                 assignments: List[CharacterAssignment],
                 replacement_audio_path: Optional[str] = None,
                 ffmpeg_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.assignments = assignments
        self.replacement_audio_path = replacement_audio_path
        self.ffmpeg_path = ffmpeg_path or get_ffmpeg_path()

    def run(self):
        try:
            processor = VideoProcessor(ffmpeg_path=self.ffmpeg_path)
            processor.process_video(
                self.input_video_path,
                self.output_video_path,
                self.assignments,
                replacement_audio_path=self.replacement_audio_path,
                progress_cb=lambda cur, total: self.progress.emit(cur, total),
            )
            self.finished_ok.emit(self.output_video_path)
        except Exception as exc:  # noqa: BLE001 - surface any error to the UI
            self.failed.emit(str(exc))


class VoiceChangeWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, input_audio_path: str, output_audio_path: str,
                 mode: str, semitones: float = 0.0, formant_ratio: float = 1.0,
                 target_reference_audio_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.input_audio_path = input_audio_path
        self.output_audio_path = output_audio_path
        self.mode = mode  # "manual" or "match_target"
        self.semitones = semitones
        self.formant_ratio = formant_ratio
        self.target_reference_audio_path = target_reference_audio_path

    def run(self):
        try:
            changer = VoiceChanger()
            if self.mode == "match_target" and self.target_reference_audio_path:
                changer.match_target_voice(
                    self.input_audio_path, self.target_reference_audio_path,
                    self.output_audio_path, formant_ratio=self.formant_ratio)
            else:
                changer.apply_pitch_and_formant(
                    self.input_audio_path, self.output_audio_path,
                    self.semitones, self.formant_ratio)
            self.finished_ok.emit(self.output_audio_path)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
