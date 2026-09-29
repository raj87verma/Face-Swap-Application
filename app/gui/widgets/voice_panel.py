"""
Widget for the (optional) voice-change step: extract the audio track of the
currently processed video, apply classical pitch/formant shifting (manual
or "match target voice"), and use the result as the audio for the final
export instead of the original audio.
"""
import os

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget
)

from app.core.video_processor import VideoProcessor
from app.gui.workers import VoiceChangeWorker
from app.utils.config import TEMP_DIR


class VoicePanel(QWidget):
    voice_change_ready = pyqtSignal(str)  # emitted with output_audio_path when done

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source_video_path = None
        self.extracted_audio_path = None
        self.target_reference_audio_path = None
        self.output_audio_path = None
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        info = QLabel(
            "Voice change uses classical pitch/formant shifting (signal "
            "processing) -- NOT AI voice cloning. 'Match target voice' "
            "approximates the target's average pitch/tone; it will not "
            "reproduce their exact voice identity."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        extract_box = QGroupBox("1. Source Audio")
        extract_layout = QHBoxLayout(extract_box)
        self.video_label = QLabel("No video set")
        extract_btn = QPushButton("Use Video's Audio...")
        extract_btn.clicked.connect(self._on_pick_source_video)
        extract_layout.addWidget(self.video_label, 1)
        extract_layout.addWidget(extract_btn)
        root.addWidget(extract_box)

        mode_box = QGroupBox("2. Voice Change Mode")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Manual pitch/formant shift", "Match target voice (approx.)"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)

        self.mode_stack = QStackedWidget()

        # Manual mode controls
        manual_widget = QWidget()
        manual_layout = QHBoxLayout(manual_widget)
        manual_layout.addWidget(QLabel("Pitch shift (semitones):"))
        self.pitch_spin = QDoubleSpinBox()
        self.pitch_spin.setRange(-12.0, 12.0)
        self.pitch_spin.setValue(0.0)
        manual_layout.addWidget(self.pitch_spin)
        manual_layout.addWidget(QLabel("Formant ratio:"))
        self.formant_spin = QDoubleSpinBox()
        self.formant_spin.setRange(0.5, 2.0)
        self.formant_spin.setSingleStep(0.05)
        self.formant_spin.setValue(1.0)
        manual_layout.addWidget(self.formant_spin)
        self.mode_stack.addWidget(manual_widget)

        # Target-match mode controls
        target_widget = QWidget()
        target_layout = QHBoxLayout(target_widget)
        self.target_audio_label = QLabel("No target reference audio")
        target_btn = QPushButton("Select Target Voice Sample...")
        target_btn.clicked.connect(self._on_pick_target_audio)
        target_layout.addWidget(self.target_audio_label, 1)
        target_layout.addWidget(target_btn)
        self.mode_stack.addWidget(target_widget)

        mode_layout.addWidget(self.mode_stack)
        root.addWidget(mode_box)

        run_box = QGroupBox("3. Apply")
        run_layout = QVBoxLayout(run_box)
        self.status_label = QLabel("Idle")
        self.apply_btn = QPushButton("Apply Voice Change")
        self.apply_btn.clicked.connect(self._on_apply)
        run_layout.addWidget(self.apply_btn)
        run_layout.addWidget(self.status_label)
        root.addWidget(run_box)

        root.addStretch(1)

    def _on_mode_changed(self, index):
        self.mode_stack.setCurrentIndex(index)

    def _on_pick_source_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video (audio will be extracted)", "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm);;Audio files (*.wav *.mp3)")
        if not path:
            return
        self.source_video_path = path
        self.video_label.setText(os.path.basename(path))

        os.makedirs(TEMP_DIR, exist_ok=True)
        audio_out = os.path.join(TEMP_DIR, "_extracted_audio.wav")
        if path.lower().endswith((".wav", ".mp3")):
            self.extracted_audio_path = path
        else:
            extracted = VideoProcessor.extract_audio(path, audio_out)
            if extracted is None:
                QMessageBox.warning(
                    self, "Audio Extraction Failed",
                    "Could not extract audio (ffmpeg missing or video has no audio track).")
                self.extracted_audio_path = None
            else:
                self.extracted_audio_path = extracted

    def _on_pick_target_audio(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select target voice sample", "", "Audio files (*.wav *.mp3 *.flac)")
        if path:
            self.target_reference_audio_path = path
            self.target_audio_label.setText(os.path.basename(path))

    def _on_apply(self):
        if not self.extracted_audio_path:
            QMessageBox.warning(self, "No Audio", "Select a source video/audio first.")
            return

        os.makedirs(TEMP_DIR, exist_ok=True)
        self.output_audio_path = os.path.join(TEMP_DIR, "_voice_changed.wav")
        mode_idx = self.mode_combo.currentIndex()

        if mode_idx == 0:
            mode = "manual"
            semitones = self.pitch_spin.value()
            formant_ratio = self.formant_spin.value()
            target_ref = None
        else:
            mode = "match_target"
            semitones = 0.0
            formant_ratio = self.formant_spin.value()
            target_ref = self.target_reference_audio_path
            if not target_ref:
                QMessageBox.warning(self, "No Target Audio",
                                     "Select a target voice sample first.")
                return

        self.apply_btn.setEnabled(False)
        self.status_label.setText("Processing audio...")

        self.worker = VoiceChangeWorker(
            self.extracted_audio_path, self.output_audio_path, mode,
            semitones=semitones, formant_ratio=formant_ratio,
            target_reference_audio_path=target_ref)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_finished(self, output_path):
        self.status_label.setText(f"Done: {output_path}")
        self.apply_btn.setEnabled(True)
        self.voice_change_ready.emit(output_path)
        QMessageBox.information(
            self, "Voice Change Complete",
            "Voice change applied. This audio will now be used for the "
            "final video export in the Process tab (replacing the "
            "original audio).")

    def _on_failed(self, error_message):
        self.status_label.setText("Failed.")
        self.apply_btn.setEnabled(True)
        QMessageBox.critical(self, "Voice Change Failed", error_message)
