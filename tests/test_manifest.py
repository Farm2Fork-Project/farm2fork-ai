from pathlib import Path

from crop_grading.data.manifest import find_cross_split_duplicates, validate_manifest


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "image_path,crop_label,grade_label,source,split\n"
        + "\n".join(
            f"{row['image_path']},{row['crop_label']},{row['grade_label']},"
            f"{row['source']},{row['split']}"
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_manifest_accepts_valid_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "data" / "processed" / "train" / "rice" / "A" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"fake image bytes")

    manifest = tmp_path / "train_labels.csv"
    write_manifest(
        manifest,
        [
            {
                "image_path": "data/processed/train/rice/A/sample.jpg",
                "crop_label": "rice",
                "grade_label": "A",
                "source": "manual",
                "split": "train",
            }
        ],
    )

    summary = validate_manifest(manifest, project_root=tmp_path)

    assert summary.is_valid
    assert summary.rows == 1
    assert summary.crop_counts == {"rice": 1}
    assert summary.grade_counts == {"A": 1}


def test_validate_manifest_reports_bad_labels_and_missing_file(tmp_path: Path) -> None:
    manifest = tmp_path / "train_labels.csv"
    write_manifest(
        manifest,
        [
            {
                "image_path": "data/processed/train/banana/A/sample.jpg",
                "crop_label": "banana",
                "grade_label": "Z",
                "source": "",
                "split": "training",
            }
        ],
    )

    summary = validate_manifest(manifest, project_root=tmp_path)
    messages = [issue.message for issue in summary.issues]

    assert not summary.is_valid
    assert "image file not found: data/processed/train/banana/A/sample.jpg" in messages
    assert "invalid crop_label: banana" in messages
    assert "invalid grade_label: Z" in messages
    assert "invalid split: training" in messages
    assert "source is required" in messages


def test_find_cross_split_duplicates(tmp_path: Path) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    common_header = "image_path,crop_label,grade_label,source,split\n"
    (metadata_dir / "train_labels.csv").write_text(
        common_header + "same.jpg,rice,A,unit,train\n",
        encoding="utf-8",
    )
    (metadata_dir / "val_labels.csv").write_text(
        common_header + "same.jpg,rice,A,unit,val\n",
        encoding="utf-8",
    )
    (metadata_dir / "test_labels.csv").write_text(
        common_header + "other.jpg,rice,A,unit,test\n",
        encoding="utf-8",
    )

    duplicates = find_cross_split_duplicates(metadata_dir)

    assert duplicates == {"same.jpg": ["train", "val"]}
