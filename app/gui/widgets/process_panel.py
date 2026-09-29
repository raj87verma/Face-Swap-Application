"""
Widget for the main processing workflow:
  1. Import a video.
  2. Enable/disable which characters should be swapped in (Person A ->
     Reference X, Person B -> Reference Y is achieved automatically by the
     CharacterMatcher during processing; here the user just chooses which
     known characters participate).
  3. Run processing with progress feedback.
  4. Export the resulting video.
"""
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QVBoxLayout, QWidget
)

from app.core.reference_manager import ReferenceManager
from app.core.video_processor import CharacterAssignment
from app.gui.workers import VideoProcessingWorker
from app.utils.config import TEMP_DIR


class ProcessPanel(QWidget):
    def __init__(self, reference_manager: ReferenceManager, parent=None):
        super().__init__(parent)
        self.ref_manager = reference_manager
        self.input_video_path = None
        self.output_video_path = None
        self.replacement_audio_path = None
        self.worker = None
        self._build_ui()

    def set_replacement_audio(self, audio_path):
        """Called by MainWindow when the Voice panel finishes producing a
        voice-changed audio file, so Process panel can use it in place of
        the original video's audio for the final export."""
        self.replacement_audio_path = audio_path
        if audio_path:
            self.audio_status_label.setText(f"Using voice-changed audio: {os.path.basename(audio_path)}")
        else:
            self.audio_status_label.setText("Using original video audio")

    def _build_ui(self):
        root = QVBoxLayout(self)

        # -- Video import --
        import_box = QGroupBox("1. Import Video")
        import_layout = QHBoxLayout(import_box)
        self.video_path_label = QLabel("No video selected")
        import_btn = QPushButton("Browse...")
        import_btn.clicked.connect(self._on_browse_video)
        import_layout.addWidget(self.video_path_label, 1)
        import_layout.addWidget(import_btn)
        root.addWidget(import_box)

        # -- Character assignment --
        assign_box = QGroupBox("2. Choose Characters to Swap In")
        assign_layout = QVBoxLayout(assign_box)
        self.character_check_list = QListWidget()
        refresh_btn = QPushButton("Refresh Character List")
        refresh_btn.clicked.connect(self.refresh_characters)
        assign_layout.addWidget(self.character_check_list)
        assign_layout.addWidget(refresh_btn)
        root.addWidget(assign_box)

        # -- Processing --
        process_box = QGroupBox("3. Process")
        process_layout = QVBoxLayout(process_box)
        self.audio_status_label = QLabel("Using original video audio")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.status_label = QLabel("Idle")
        self.process_btn = QPushButton("Start Face Swap Processing")
        self.process_btn.clicked.connect(self._on_start_processing)
        process_layout.addWidget(self.audio_status_label)
        process_layout.addWidget(self.process_btn)
        process_layout.addWidget(self.progress_bar)
        process_layout.addWidget(self.status_label)
        root.addWidget(process_box)

        # -- Export --
        export_box = QGroupBox("4. Export")
        export_layout = QHBoxLayout(export_box)
        self.export_btn = QPushButton("Save Output Video As...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._on_export)
        export_layout.addWidget(self.export_btn)
        root.addWidget(export_box)

        root.addStretch(1)
        self.refresh_characters()

    # ------------------------------------------------------------------ #
    def refresh_characters(self):
        self.character_check_list.clear()
        for character in self.ref_manager.list_characters():
            item = QListWidgetItem(f"{character.name} ({len(character.references)} refs)")
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if character.references else Qt.Unchecked)
            item.setData(Qt.UserRole, character.character_id)
            self.character_check_list.addItem(item)

    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select video", "",
            "Video files (*.mp4 *.mov *.avi *.mkv *.webm)")
        if path:
            self.input_video_path = path
            self.video_path_label.setText(os.path.basename(path))

    def _selected_assignments(self):
        assignments = []
        for i in range(self.character_check_list.count()):
            item = self.character_check_list.item(i)
            char_id = item.data(Qt.UserRole)
            character = self.ref_manager.get_character(char_id)
            if character:
                assignments.append(CharacterAssignment(
                    character=character, enabled=(item.checkState() == Qt.Checked)))
        return assignments

    def _on_start_processing(self):
        if not self.input_video_path:
            QMessageBox.warning(self, "No Video", "Please import a video first.")
            return
        assignments = self._selected_assignments()
        if not any(a.enabled for a in assignments):
            QMessageBox.warning(self, "No Characters",
                                 "Check at least one character with reference images.")
            return

        os.makedirs(TEMP_DIR, exist_ok=True)
        self.output_video_path = os.path.join(TEMP_DIR, "_processed_output.mp4")

        self.process_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing... this may take a while depending on video length.")

        self.worker = VideoProcessingWorker(
            self.input_video_path, self.output_video_path, assignments,
            replacement_audio_path=self.replacement_audio_path)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(int(current * 100 / total))
        self.status_label.setText(f"Processing frame {current}/{total or '?'}")

    def _on_finished(self, output_path):
        self.status_label.setText("Done. Preview/export the result below.")
        self.process_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        QMessageBox.information(self, "Processing Complete",
                                 "Face swap processing finished successfully.")

    def _on_failed(self, error_message):
        self.status_label.setText("Failed.")
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Processing Failed", error_message)

    def _on_export(self):
        if not self.output_video_path or not os.path.isfile(self.output_video_path):
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save output video as", "output.mp4", "MP4 video (*.mp4)")
        if dest:
            import shutil
            shutil.copy2(self.output_video_path, dest)
            QMessageBox.information(self, "Saved", f"Saved to {dest}")
