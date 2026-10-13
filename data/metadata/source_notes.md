# Dataset Source Notes

This file tracks what each raw dataset can and cannot be used for.

## Rice Variety Dataset

- Local path: `data/raw/Rice/Rice_Image_Dataset`
- Source: Kaggle / Murat Koklu rice image dataset
- Labels: `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, `Karacadag`
- Use: rice crop recognition and visual feature learning
- Limitation: this is not an A/B/C/D quality dataset
- Citation note: keep `Rice_Citation_Request.txt` with the raw dataset

## Rice Quality Dataset

- Local path: `data/raw/Rice/rice_quality`
- Labels: `full`, `mixed`, `broken`
- Use: rice quality grading bootstrap data
- Grade mapping:
  - `full` -> `A`
  - `mixed` -> `C`
  - `broken` -> `D`
- Known gap: no reliable `B` class yet

Do not create synthetic `B` labels from this dataset. Add `B` only when we have a source that clearly represents mostly full rice with minor defects.

## Mango Classification Dataset

- Local path: `data/raw/mango/Classification_dataset`
- Labels: Pakistani mango varieties including `Anwar Ratool`, `Dosehri`, `Langra`, `Sindhri`, and Chaunsa variants
- Use: mango crop/variety recognition and feature learning
- Limitation: this is not an A/B/C/D quality dataset

## Mango Grading Dataset

- Local path: `data/raw/mango/Grading_dataset`
- Labels: `Extra_Class`, `Class_I`, `Class_II`
- Use: mango quality grading bootstrap data
- Grade mapping:
  - `Extra_Class` -> `A`
  - `Class_I` -> `B`
  - `Class_II` -> `C`
- Known gap: no reliable `D` class yet

## Wheat Dataset

- Local path: `data/raw/wheat`
- Labels observed: `0_NOR`, `1_F_and_S`, `2_SD`, `3_MY`, `4_AP`, `5_BN`, `6_BP`, `7_IM`
- XML metadata: `data/raw/wheat/wheat_tiny.xml`
- Use: wheat quality grading bootstrap data
- Confirmed: XML `DU_grain` labels match the image folder codes and all train/test IDs
- Grade mapping:
  - `0_NOR` -> `A`
  - `5_BN`, `6_BP` -> `B`
  - `4_AP`, `2_SD` -> `C`
- Folder rename for Kaggle: original GrainSet code `F&S` is stored locally as `1_F_and_S`
  - `1_F_and_S`, `3_MY` -> `D`
- Excluded by default: `7_IM`, because impurities are non-grain contaminants rather than crop quality examples
- Note: `mask/` appears to contain segmentation masks and should be excluded from classification manifests unless we intentionally build segmentation tooling

## Maize Dataset

- Local path: `data/raw/maize`
- Labels observed: `0_NOR`, `1_F_and_S`, `2_SD`, `3_MY`, `4_AP`, `5_BN`, `6_HD`, `7_IM`
- XML metadata: `data/raw/maize/maize_tiny.xml`
- Use: maize quality grading bootstrap data
- Confirmed: XML `DU_grain` labels match the image folder codes and all train/test IDs
- Grade mapping:
  - `0_NOR` -> `A`
  - `5_BN`, `6_HD` -> `B`
  - `4_AP`, `2_SD` -> `C`
- Folder rename for Kaggle: original GrainSet code `F&S` is stored locally as `1_F_and_S`
  - `1_F_and_S`, `3_MY` -> `D`
- Excluded by default: `7_IM`, because impurities are non-grain contaminants rather than crop quality examples
- Note: `mask/` appears to contain segmentation masks and should be excluded from classification manifests unless we intentionally build segmentation tooling
