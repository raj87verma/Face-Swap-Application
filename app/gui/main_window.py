"""
Main application window: tabs for Character management, Video processing,
and Voice change, all wired to the shared ReferenceManager.
"""
from PyQt5.QtWidgets import QMainWindow, QTabWidget

from app.core.reference_manager import ReferenceManager
from app.gui.widgets.character_panel import CharacterPanel
from app.gui.widgets.process_panel import ProcessPanel
from app.gui.widgets.voice_panel import VoicePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FaceSwap Studio - Classical CV Face Swap & Voice Change")
        self.resize(1000, 700)

        self.reference_manager = ReferenceManager()

        self.tabs = QTabWidget()
        self.character_panel = CharacterPanel(self.reference_manager)
        self.process_panel = ProcessPanel(self.reference_manager)
        self.voice_panel = VoicePanel()

        self.tabs.addTab(self.character_panel, "1. Characters && References")
        self.tabs.addTab(self.process_panel, "2. Import Video && Face Swap")
        self.tabs.addTab(self.voice_panel, "3. Voice Change (optional)")

        # Refresh character checklist in Process tab whenever the user
        # switches to it, in case references were just added/edited.
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # When a voice change finishes, offer its output as the audio
        # source for the final video export.
        self.voice_panel.voice_change_ready.connect(self.process_panel.set_replacement_audio)

        self.setCentralWidget(self.tabs)

    def _on_tab_changed(self, index):
        if self.tabs.widget(index) is self.process_panel:
            self.process_panel.refresh_characters()
