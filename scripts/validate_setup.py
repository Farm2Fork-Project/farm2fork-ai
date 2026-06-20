"""Validate that the project scaffold and config are internally consistent."""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from crop_grading.utils.config import load_config


def main() -> None:
    config = load_config()
    print(f"Project: {config.project.name}")
    print(f"Crops: {', '.join(config.data.crops)}")
    print(f"Grades: {', '.join(config.data.grades)}")
    print("Config validation passed.")


if __name__ == "__main__":
    main()
