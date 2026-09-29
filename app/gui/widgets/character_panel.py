"""
Widget for managing characters: create a character, add multiple reference
images from different angles, and view/remove existing references.
"""
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QInputDialog, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget
)

from app.core.reference_manager import ReferenceManager

ANGLE_CHOICES = ["front", "left", "right", "up", "down", "left_profile",
                  "right_profile", "unspecified"]


class CharacterPanel(QWidget):
    """Left side: list of characters + add/remove. Right side: reference
    images (multi-angle) of the selected character."""

    def __init__(self, reference_manager: Optional[ReferenceManager] = None, parent=None):
        super().__init__(parent)
        self.ref_manager = reference_manager or ReferenceManager()
        self._build_ui()
        self._refresh_character_list()

    # ------------------------------------------------------------------ #
    def _build_ui(self):
        root = QHBoxLayout(self)

        # -- Characters column --
        char_box = QGroupBox("Characters")
        char_layout = QVBoxLayout(char_box)
        self.character_list = QListWidget()
        self.character_list.currentItemChanged.connect(self._on_character_selected)
        char_layout.addWidget(self.character_list)

        char_btns = QHBoxLayout()
        add_char_btn = QPushButton("+ Add Character")
        add_char_btn.clicked.connect(self._on_add_character)
        del_char_btn = QPushButton("Delete")
        del_char_btn.clicked.connect(self._on_delete_character)
        char_btns.addWidget(add_char_btn)
        char_btns.addWidget(del_char_btn)
        char_layout.addLayout(char_btns)

        # -- Reference images column --
        ref_box = QGroupBox("Reference Images (multiple angles)")
        ref_layout = QVBoxLayout(ref_box)

        self.preview_label = QLabel("No image selected")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(220, 220)
        self.preview_label.setStyleSheet("border: 1px solid #888;")
        ref_layout.addWidget(self.preview_label)

        self.reference_list = QListWidget()
        self.reference_list.currentItemChanged.connect(self._on_reference_selected)
        ref_layout.addWidget(self.reference_list)

        add_ref_row = QHBoxLayout()
        self.angle_combo = QComboBox()
        self.angle_combo.addItems(ANGLE_CHOICES)
        add_ref_btn = QPushButton("+ Add Reference Image")
        add_ref_btn.clicked.connect(self._on_add_reference)
        remove_ref_btn = QPushButton("Remove")
        remove_ref_btn.clicked.connect(self._on_remove_reference)
        add_ref_row.addWidget(QLabel("Angle:"))
        add_ref_row.addWidget(self.angle_combo)
        add_ref_row.addWidget(add_ref_btn)
        add_ref_row.addWidget(remove_ref_btn)
        ref_layout.addLayout(add_ref_row)

        root.addWidget(char_box, 1)
        root.addWidget(ref_box, 2)

    # ------------------------------------------------------------------ #
    def _refresh_character_list(self):
        self.character_list.clear()
        for character in self.ref_manager.list_characters():
            item = QListWidgetItem(f"{character.name}  ({len(character.references)} refs)")
            item.setData(Qt.UserRole, character.character_id)
            self.character_list.addItem(item)

    def _current_character_id(self) -> Optional[str]:
        item = self.character_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def _refresh_reference_list(self):
        self.reference_list.clear()
        char_id = self._current_character_id()
        if not char_id:
            return
        character = self.ref_manager.get_character(char_id)
        if not character:
            return
        for ref in character.references:
            item = QListWidgetItem(f"[{ref.angle_label}] {ref.path.split('/')[-1]}")
            item.setData(Qt.UserRole, ref.path)
            self.reference_list.addItem(item)

    # ------------------------------------------------------------------ #
    def _on_add_character(self):
        name, ok = QInputDialog.getText(self, "New Character", "Character name:")
        if ok and name.strip():
            self.ref_manager.create_character(name.strip())
            self._refresh_character_list()

    def _on_delete_character(self):
        char_id = self._current_character_id()
        if not char_id:
            return
        confirm = QMessageBox.question(self, "Delete Character",
                                        "Delete this character and all its reference images?")
        if confirm == QMessageBox.Yes:
            self.ref_manager.delete_character(char_id)
            self._refresh_character_list()
            self.reference_list.clear()
            self.preview_label.setText("No image selected")
            self.preview_label.setPixmap(QPixmap())

    def _on_character_selected(self, *_args):
        self._refresh_reference_list()

    def _on_add_reference(self):
        char_id = self._current_character_id()
        if not char_id:
            QMessageBox.warning(self, "No Character", "Select or create a character first.")
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select reference image(s)", "",
            "Images (*.jpg *.jpeg *.png *.bmp)")
        if not paths:
            return
        angle = self.angle_combo.currentText()
        for path in paths:
            self.ref_manager.add_reference_image(char_id, path, angle_label=angle)
        self._refresh_character_list()
        self._refresh_reference_list()

    def _on_remove_reference(self):
        char_id = self._current_character_id()
        item = self.reference_list.currentItem()
        if not char_id or not item:
            return
        ref_path = item.data(Qt.UserRole)
        self.ref_manager.remove_reference_image(char_id, ref_path)
        self._refresh_character_list()
        self._refresh_reference_list()

    def _on_reference_selected(self, *_args):
        item = self.reference_list.currentItem()
        if not item:
            self.preview_label.setText("No image selected")
            self.preview_label.setPixmap(QPixmap())
            return
        path = item.data(Qt.UserRole)
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.preview_label.setText("Could not load image")
            return
        self.preview_label.setPixmap(
            pixmap.scaled(220, 220, Qt.KeepAspectRatio, Qt.SmoothTransformation))
