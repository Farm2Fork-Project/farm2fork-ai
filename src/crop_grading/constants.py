"""Shared labels used by training, inference, and API layers."""

CROP_CLASSES = ("wheat", "rice", "mango", "maize", "cotton", "sugarcane")
GRADE_CLASSES = ("A", "B", "C", "D")

CROP_LABEL_TO_INDEX = {label: index for index, label in enumerate(CROP_CLASSES)}
CROP_INDEX_TO_LABEL = {index: label for label, index in CROP_LABEL_TO_INDEX.items()}

GRADE_LABEL_TO_INDEX = {label: index for index, label in enumerate(GRADE_CLASSES)}
GRADE_INDEX_TO_LABEL = {index: label for label, index in GRADE_LABEL_TO_INDEX.items()}
