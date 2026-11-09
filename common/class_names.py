"""Explicit class order, matching the referenced ResNet-9 preparation script.

WbNet uses a DIFFERENT order. Never infer this mapping from directory iteration.
"""
from pathlib import PurePath
import re

CLASS_NAMES = (
    "Ae. aegypti", "Ae. albopictus", "An. gambiae", "An. arabiensis",
    "C. pipiens", "C. quinquefasciatus",
)
CLASS_SLUGS = ("aegypti", "albopictus", "gambiae", "arabiensis", "pipiens", "quinquefasciatus")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)


def label_from_path(relative_path: str) -> int:
    """Match species in directory components only, including nested Kaggle roots."""
    matches = set()
    for component in PurePath(relative_path.replace("\\", "/")).parts[:-1]:
        words = set(re.findall(r"[a-z]+", component.lower()))
        matches.update(i for i, species in enumerate(CLASS_SLUGS) if species in words)
    if len(matches) != 1:
        raise ValueError(f"Cannot uniquely identify species directory for {relative_path!r}: {matches}")
    return matches.pop()
