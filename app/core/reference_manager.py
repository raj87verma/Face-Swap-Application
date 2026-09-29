"""
Manages "characters" -- named identities each backed by multiple reference
images taken from different angles (front, left, right, up, down, etc.).

Each character's reference images are stored on disk under
data/characters/<character_name>/ and an index (angles.json) records which
image corresponds to which pose label (optional metadata, purely informative
for the user; matching itself does not require exact labels).
"""
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.utils.config import CHARACTERS_DIR


@dataclass
class ReferenceImage:
    path: str
    angle_label: str = "unspecified"  # e.g. "front", "left", "right", "up", "down"


@dataclass
class Character:
    character_id: str
    name: str
    directory: str
    references: List[ReferenceImage] = field(default_factory=list)

    def load_images(self) -> List[np.ndarray]:
        images = []
        for ref in self.references:
            img = cv2.imread(ref.path)
            if img is not None:
                images.append(img)
        return images


class ReferenceManager:
    """CRUD operations for characters and their multi-angle reference images."""

    INDEX_FILENAME = "angles.json"

    def __init__(self, base_dir: str = CHARACTERS_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self._characters: Dict[str, Character] = {}
        self._load_all()

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _load_all(self) -> None:
        self._characters.clear()
        if not os.path.isdir(self.base_dir):
            return
        for entry in sorted(os.listdir(self.base_dir)):
            char_dir = os.path.join(self.base_dir, entry)
            if not os.path.isdir(char_dir):
                continue
            index_path = os.path.join(char_dir, self.INDEX_FILENAME)
            name = entry
            char_id = entry
            refs: List[ReferenceImage] = []
            if os.path.isfile(index_path):
                try:
                    with open(index_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    name = data.get("name", entry)
                    char_id = data.get("character_id", entry)
                    for r in data.get("references", []):
                        refs.append(ReferenceImage(path=os.path.join(char_dir, r["file"]),
                                                    angle_label=r.get("angle", "unspecified")))
                except (json.JSONDecodeError, OSError):
                    pass
            else:
                # No index yet; pick up any image files already in the folder.
                for fname in sorted(os.listdir(char_dir)):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        refs.append(ReferenceImage(path=os.path.join(char_dir, fname)))
            self._characters[char_id] = Character(character_id=char_id, name=name,
                                                    directory=char_dir, references=refs)

    def _save_index(self, character: Character) -> None:
        index_path = os.path.join(character.directory, self.INDEX_FILENAME)
        data = {
            "character_id": character.character_id,
            "name": character.name,
            "references": [
                {"file": os.path.basename(r.path), "angle": r.angle_label}
                for r in character.references
            ],
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list_characters(self) -> List[Character]:
        return list(self._characters.values())

    def get_character(self, character_id: str) -> Optional[Character]:
        return self._characters.get(character_id)

    def create_character(self, name: str) -> Character:
        char_id = f"{name.strip().replace(' ', '_')}_{uuid.uuid4().hex[:6]}"
        char_dir = os.path.join(self.base_dir, char_id)
        os.makedirs(char_dir, exist_ok=True)
        character = Character(character_id=char_id, name=name, directory=char_dir)
        self._characters[char_id] = character
        self._save_index(character)
        return character

    def add_reference_image(self, character_id: str, image_path: str,
                             angle_label: str = "unspecified") -> ReferenceImage:
        character = self._characters.get(character_id)
        if character is None:
            raise KeyError(f"Unknown character_id: {character_id}")

        ext = os.path.splitext(image_path)[1] or ".jpg"
        dest_name = f"{angle_label}_{uuid.uuid4().hex[:8]}{ext}"
        dest_path = os.path.join(character.directory, dest_name)
        shutil.copy2(image_path, dest_path)

        ref = ReferenceImage(path=dest_path, angle_label=angle_label)
        character.references.append(ref)
        self._save_index(character)
        return ref

    def remove_reference_image(self, character_id: str, ref_path: str) -> None:
        character = self._characters.get(character_id)
        if character is None:
            return
        character.references = [r for r in character.references if r.path != ref_path]
        if os.path.isfile(ref_path):
            os.remove(ref_path)
        self._save_index(character)

    def delete_character(self, character_id: str) -> None:
        character = self._characters.pop(character_id, None)
        if character and os.path.isdir(character.directory):
            shutil.rmtree(character.directory)

    def rename_character(self, character_id: str, new_name: str) -> None:
        character = self._characters.get(character_id)
        if character is None:
            return
        character.name = new_name
        self._save_index(character)
