# GrainSet (wheat + maize) — download & placement guide

You already have the GrainSet **label lists** (`data/datalist/*.txt`) but **not the images**.
This guide gets the wheat + maize images into the exact structure the manifest builder
(`scripts/build_bootstrap_quality_manifests.py`) expects.

Source: *An annotated grain kernel image database for visual quality inspection*, Nature
Scientific Data (2023). Repo: https://github.com/hellodfan/GrainSet · CC BY 4.0.

---

## ⚠️ First: decide the scale (this matters)

The **full** GrainSet wheat set is **200,000 images**; maize is **19,000**. Your current
graded data is rice **4,237** + mango **600**. If you add full wheat, ~97% of the dataset
becomes wheat — the "multi-crop" story collapses and training on your M2 Pro gets very slow.

| Option | Wheat | Maize | Verdict |
|---|---|---|---|
| **GrainSet-tiny (recommended)** | ~4K | ~0.4K | Balanced with rice/mango; matches your original audit; fast on M2 Pro |
| Full wheat + maize | 200K | 19K | Only if you deliberately subsample (use the `*_bal.txt` balanced lists, or cap per class) |

**Recommendation:** use **GrainSet-tiny** for the paper's first results. You can scale up later
and report it as an ablation ("effect of training-set size"). Keep it balanced.

---

## Download links (Figshare, CC BY 4.0)

| Dataset | Images | DOI / link |
|---|---|---|
| **GrainSet-tiny** (recommended) | 6.5K (all 4 grains) | https://doi.org/10.6084/m9.figshare.22989029.v1 |
| Wheat (full) | 200K | https://doi.org/10.6084/m9.figshare.22992317.v2 |
| Maize (full) | 19K | https://doi.org/10.6084/m9.figshare.22987562.v2 |
| Rice (full, optional) | 31K | https://doi.org/10.6084/m9.figshare.22987292.v3 |
| Sorghum (not a target crop) | 102K | https://doi.org/10.6084/m9.figshare.22988981.v2 |

Open a DOI link in your browser → Figshare article page → **Download** the `.zip`.
(These are large public files; grab them via the browser, not a script.)

---

## Target structure the pipeline needs

The builder reads **only `train/` and `test/`** under each crop root, then re-splits
70/15/15 itself. Final layout must be:

```text
data/raw/wheat/
  train/{0_NOR,1_F_and_S,2_SD,3_MY,4_AP,5_BN,6_BP,7_IM}/*.png
  test/ {0_NOR,1_F_and_S,2_SD,3_MY,4_AP,5_BN,6_BP,7_IM}/*.png
data/raw/maize/
  train/{0_NOR,1_F_and_S,2_SD,3_MY,4_AP,5_BN,6_HD,7_IM}/*.png
  test/ {0_NOR,1_F_and_S,2_SD,3_MY,4_AP,5_BN,6_HD,7_IM}/*.png
```

**The one gotcha:** GrainSet ships the Fusarium class as **`1_F&S`**; the pipeline expects
**`1_F_and_S`**. That folder must be renamed (all other class names already match).
`7_IM` (impurities) is excluded by default — leave it; the builder skips it unless you pass
`--include-impurities`.

Grade mapping (already coded, for reference): `0_NOR→A`; `5_BN,6_BP/6_HD→B`;
`4_AP,2_SD→C`; `1_F_and_S,3_MY→D`.

---

## Two ways to place the files

### Path A — I do it for you (recommended, least error-prone)
1. Download the zip(s) above and unzip anywhere (e.g. `~/Downloads/grainset_tiny/`).
2. Tell me the extracted path.
3. I'll write a short script that reads your existing `data/datalist/*.txt` to copy each image
   into the correct `data/raw/<crop>/<split>/<class>/`, renaming `1_F&S → 1_F_and_S`
   automatically — the same way I organized rice and mango.

This is robust because the datalists already map every image to its split + class, so we don't
depend on guessing the archive's internal folder layout.

### Path B — manual
1. Unzip so wheat images land under `data/raw/wheat/` with `train/` and `test/` subfolders of
   class folders (as above); same for maize under `data/raw/maize/`.
2. Rename the Fusarium folders:
   ```bash
   cd /Users/macbook/UCP/FYP/farm2fork-ai/data/raw
   for c in wheat maize; do for s in train test; do
     [ -d "$c/$s/1_F&S" ] && mv "$c/$s/1_F&S" "$c/$s/1_F_and_S"
   done; done
   ```

---

## After placement: rebuild manifests & verify

```bash
cd /Users/macbook/UCP/FYP/farm2fork-ai
.venv/bin/python scripts/build_bootstrap_quality_manifests.py
```

It will now include wheat + maize automatically (no more `[skip]` lines) and print per-split
crop/grade counts with `valid=True`. Then retrain with all four crops:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python scripts/train.py --config configs/rice_mango_15ep.yaml
```

(Consider renaming that config to `four_crops_15ep.yaml` once wheat+maize are in.)
