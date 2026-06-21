# Dataset Audit

Last inspected: 2026-06-21

## Raw Folder Layout

```text
data/raw/
  Rice/
    Rice_Image_Dataset/
    rice_quality/
  mango/
    Classification_dataset/
    Grading_dataset/
  wheat/
    train/
    test/
    mask/
    wheat_tiny.xml
  maize/
    train/
    test/
    mask/
    maize_tiny.xml
```

## Usable Quality Sources

| Crop | Path | Classes | Quality Mapping | Notes |
|---|---|---|---|---|
| rice | `data/raw/Rice/rice_quality` | `full`, `mixed`, `broken` | `full -> A`, `mixed -> C`, `broken -> D` | No reliable B class |
| mango | `data/raw/mango/Grading_dataset` | `Extra_Class`, `Class_I`, `Class_II` | `Extra_Class -> A`, `Class_I -> B`, `Class_II -> C` | No D class |
| wheat | `data/raw/wheat/train`, `data/raw/wheat/test` | GrainSet DU classes | `NOR -> A`, `BN/BP -> B`, `AP/SD -> C`, `F&S/MY -> D` | `IM` excluded by default |
| maize | `data/raw/maize/train`, `data/raw/maize/test` | GrainSet DU classes | `NOR -> A`, `BN/HD -> B`, `AP/SD -> C`, `F&S/MY -> D` | `IM` excluded by default |

## Crop/Variety Recognition Sources

| Crop | Path | Classes | Notes |
|---|---|---|---|
| rice | `data/raw/Rice/Rice_Image_Dataset` | `Arborio`, `Basmati`, `Ipsala`, `Jasmine`, `Karacadag` | Variety classification only, not A/B/C/D grading |
| mango | `data/raw/mango/Classification_dataset` | 8 Pakistani mango varieties | Variety classification only, not A/B/C/D grading |

## Wheat/Maize XML Metadata

Wheat and maize include XML metadata files:

- `data/raw/wheat/wheat_tiny.xml`
- `data/raw/maize/maize_tiny.xml`

Each XML object includes:

- `ID`
- `species`
- `sub-species`
- `location`
- `time`
- `size`
- `DU_grain`
- `weight`

The XML confirms the `DU_grain` codes used by the folder labels. Every train/test image ID has a matching XML object, and every `mask/` image ID also has a matching XML object.

The XML confirms labels and numeric metadata. The DU class meanings used for grading are:

| Code | Full Name | Meaning |
|---|---|---|
| NOR | Normal | Healthy, undamaged grain |
| F&S | Fusarium & Shriveled | Fungal infection / shriveled kernels |
| SD | Sprouted | Germinated or sprouted kernels |
| MY | Moldy | Mold or fungal contamination |
| AP | Attacked by Pests | Insect/pest damage |
| BN | Broken | Fractured kernels |
| BP | Black Point | Wheat-specific black discoloration at germ end |
| HD | Heated | Maize-specific heat damage from storage/processing |
| IM | Impurities | Non-grain matter such as stones, chaff, dirt, or other seeds |

Observed wheat classes:

```text
0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, 5_BN, 6_BP, 7_IM
```

Observed maize classes:

```text
0_NOR, 1_F&S, 2_SD, 3_MY, 4_AP, 5_BN, 6_HD, 7_IM
```

Market-grade mapping:

- `NOR -> A`
- `BN`, `BP`, `HD -> B`
- `AP`, `SD -> C`
- `F&S`, `MY -> D`
- `IM` is excluded by default because it is contamination/non-grain matter, not a crop-quality image class

### XML `DU_grain` Counts

| Crop | DU_grain | Count |
|---|---|---:|
| wheat | NOR | 1198 |
| wheat | F&S | 145 |
| wheat | SD | 146 |
| wheat | MY | 113 |
| wheat | AP | 123 |
| wheat | BN | 147 |
| wheat | BP | 30 |
| wheat | IM | 98 |
| maize | NOR | 312 |
| maize | F&S | 25 |
| maize | SD | 27 |
| maize | MY | 29 |
| maize | AP | 30 |
| maize | BN | 35 |
| maize | HD | 30 |
| maize | IM | 82 |

### XML Match Check

| Crop | XML Objects | Mask Files Matching XML | Train/Test Files Matching XML |
|---|---:|---:|---:|
| wheat | 2000 | 2000 | 2000 |
| maize | 570 | 570 | 570 |

## Image Counts

### Rice Quality

| Class | Count |
|---|---:|
| full | 2159 |
| mixed | 1039 |
| broken | 1039 |

### Mango Grading

| Class | Count |
|---|---:|
| Extra_Class | 200 |
| Class_I | 200 |
| Class_II | 200 |

### Wheat

| Split | Class | Count |
|---|---|---:|
| train | 0_NOR | 1078 |
| train | 1_F&S | 137 |
| train | 2_SD | 133 |
| train | 3_MY | 105 |
| train | 4_AP | 114 |
| train | 5_BN | 133 |
| train | 6_BP | 29 |
| train | 7_IM | 89 |
| test | 0_NOR | 120 |
| test | 1_F&S | 8 |
| test | 2_SD | 13 |
| test | 3_MY | 8 |
| test | 4_AP | 9 |
| test | 5_BN | 14 |
| test | 6_BP | 1 |
| test | 7_IM | 9 |

### Maize

| Split | Class | Count |
|---|---|---:|
| train | 0_NOR | 278 |
| train | 1_F&S | 20 |
| train | 2_SD | 23 |
| train | 3_MY | 25 |
| train | 4_AP | 27 |
| train | 5_BN | 31 |
| train | 6_HD | 29 |
| train | 7_IM | 75 |
| test | 0_NOR | 34 |
| test | 1_F&S | 5 |
| test | 2_SD | 4 |
| test | 3_MY | 4 |
| test | 4_AP | 3 |
| test | 5_BN | 4 |
| test | 6_HD | 1 |
| test | 7_IM | 7 |

## Sample Image Dimensions

| Source | Example Size | Mode |
|---|---|---|
| rice quality | 256 x 256 | RGB |
| rice variety | 250 x 250 | RGB |
| mango grading | 718 x 430 | RGB |
| wheat | 340 x 290 | RGB |
| maize | 600 x 488 | RGB |
