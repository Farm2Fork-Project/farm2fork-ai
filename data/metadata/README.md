# Dataset Metadata

Store split label files here:

- `train_labels.csv`
- `val_labels.csv`
- `test_labels.csv`

Each CSV should use this schema:

```csv
image_path,crop_label,grade_label,source,split
processed/train/rice/A/example.jpg,rice,A,kaggle_rice,train
```

Validate one manifest:

```powershell
python scripts/validate_manifests.py data/metadata/train_labels.csv
```

Validate labels and schema before images are present:

```powershell
python scripts/validate_manifests.py data/metadata/train_labels.csv --no-file-check
```

Rules:

- `crop_label` must be one of `wheat`, `rice`, `mango`, `maize`, `cotton`, `sugarcane`.
- `grade_label` must be one of `A`, `B`, `C`, `D`.
- Keep original downloaded files in `data/raw`.
- Use `data/processed` for cleaned images and train/validation/test split folders.
