# 01 · Dataset lineage: v1 → v2

*Analysis 2026-11-22. Raw output: [`data/dataset_lineage.json`](data/dataset_lineage.json).*

**Question:** how much of the previous (v1, 2024–25) dataset is in the data that trained the published v2 model (`diogocarapito/text-to-icpc2-v2`, tag `v1`: multilingual-e5-base retriever, run `ret-e5-base`, trained 2026-11-22 02:17–03:42 at commit `ab6069d`)?

**Short answer:** about **71% of the v1 dataset** was carried over. It makes up only about **19% of the v2 training data**, because v2 adds about 38k new Claude synthetic examples. All v1 synonyms and all official ICPC-2 texts are in v2. Only 39% of v1's ICD-10 descriptions are, because v2 rebuilt the ICD-10 part with stricter rules.

## 1. How much of v1 is in v2

v1 = `text-to-icpc2/data/data_pre_train.csv` (18,052 rows, 17,152 unique normalised texts; v1 repo commit `cffe197`, 2025-06-25). v2 = train + val rows of `data/processed/examples.parquet` (65,717 rows).

| v1 origin | v1 rows | same text in v2 | % | same text and code |
|---|---:|---:|---:|---:|
| gpt-4o-mini synonyms | 7,131 | 7,129 | 100 | 7,128 |
| gpt-4o-mini synonyms, human-edited | 189 | 189 | 100 | 189 |
| human-written synonyms | 35 | 35 | 100 | 35 |
| ICD-10 descriptions | 8,569 | 3,351 | **39.1** | 3,317 |
| ICPC-2 titles | 686 | 686 | 100 | 686 |
| ICPC-2 inclusion terms | 1,317 | 1,317 | 100 | 1,317 |
| ICPC-2 short titles | 125 | 125 | 100 | 125 |
| **total** | **18,052** | **12,832** | **71.1** | **12,797** |

The HF-format copy of the v1 dataset in the same repo (`data/data_pre_train_hf`, 17,342 rows, presumably after deduplication) gives the same picture: 12,687 rows (73.2%) carried over, ICD-10 41.5%.

### What happened to each part

- **v1 synonyms (gpt-4o-mini and human):** reused as is, from `data/legacy/v1_augmentation.csv` (a copy of v1's `data_augmentation.csv`, 11,020 rows). Only rows marked `include=1` in v1's manual review are kept (7,355). The 3,664 rows excluded in v1 stay excluded. 7,290 survive v2's cleaning (valid code, deduplication). Like the synthetic data, they are hash-split: **6,740 train / 550 val**.
- **ICPC-2 official texts:** not reused but **rebuilt** from the same ACSS ICPC-2e v4.4 file, so they come out identical. v2 adds 674 English titles.
- **ICD-10 descriptions:** rebuilt from the full ACSS ICD-10-CM PT/EN file with new rules (`configs/data.yaml`, `icd10`):
  - child codes included (e.g. A00.0, A000…), mapped by longest prefix;
  - **ambiguous ICD codes dropped** (3,467 of them map to more than one ICPC-2 code, which in v1 put identical text under different labels);
  - at most 40 texts per ICPC-2 code and language, so the injury chapters don't dominate;
  - an English version added.

  As a result, most of v1's ICD-10 texts were replaced. The 34 v1 ICD-10 texts that are in v2 with a different code come from the remapping.
- **Overlap with the test set:** 6 v1 texts (2 synonyms, 4 ICD-10) also appear in the physician-verified gold set. The leakage guard removed them from training, so they are not counted in the gold score.

## 2. What the v2 training data is made of

| v2 source family | rows (train+val) | share of v2 | text also in v1 |
|---|---:|---:|---:|
| Claude synthetic (`synth_*`) | 37,709 | 57.4% | 157 |
| ICD-10 PT + EN (rebuilt) | 17,924 | 27.3% | 3,058 |
| v1 synonyms (`legacy_*`) | 7,290 | 11.1% | 7,290 |
| ICPC-2 official texts (rebuilt) | 2,794 | 4.3% | 2,120 |
| **total** | **65,717** | 100% | **12,625 (19.2%)** |

The 157 synthetic texts that are also in v1 are short, common expressions that Claude produced independently (normalised exact match).

## 3. For the article

- v2's gain over v1 does not come from discarding the old data. All of v1's human-reviewed synonyms are kept. It comes from **adding realistic inputs** (Claude shorthand, lay wording, typos, English: 57% of v2), **removing contradictory ICD-10 labels**, and **evaluating on held-out, physician-verified data**.
- This matches the benchmark result that the same BERTimbau architecture goes from 0.346 (v1 data) to 0.896 (v2 data) top-1 on gold_test ([`../WRITEUP.md`](../WRITEUP.md) §5.1).
- Suggested figure: stacked bar of v1 (18k) vs v2 (66k) by source family, with the carried-over part shaded.

## 4. How to reproduce

```bash
make data                                         # rebuild data/processed/examples.parquet
.venv/bin/python scripts/dataset_lineage.py ../text-to-icpc2   # path to the v1 repo
```

Texts are compared after `norm_key` (lowercase, accents and punctuation stripped), the normalisation the v2 leakage guard uses. Inputs: `data/processed/synthetic.jsonl` (2026-11-21), `data/legacy/v1_augmentation.csv`, `data/raw/*`, `configs/data.yaml` at `ab6069d` or later.

## Caveats

- `examples.parquet` was rebuilt at 07:15 on 2026-11-22, after the e5-base run (02:17–03:42). No data input or build code changed after `ab6069d`, the commit that run used, so the training set should be identical. The run did not save a fingerprint of its data, so this cannot be proven. **[todo: record the training set's row count and a hash in each run's `run.json`]**
- The comparison is by normalised exact text. Near-duplicates (one word changed) count as new.
