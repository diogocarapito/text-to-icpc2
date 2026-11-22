# Lab notebook

A dated, public record of what was tried, what was measured, and what changed. It feeds the project write-up (2024 → today) and the paper. Newest entries are at the bottom. Numbers link to files under `experiments/`.

## 2024-02 → 2025-06 · v1 ([text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2), 173 commits)

- **2024-02-26:** project start. Goal: free-text diagnosis → ICPC-2 code for Portuguese primary care.
- **2024-03 → 2024-07:** ETL from the official ACSS files: the ICPC-2 v4.4 Excel and ICD-10-CM PT. Labels come from ICPC-2 titles, inclusion terms, and ICD-10 descriptions mapped through ICPC-2's ICD-10 column. A Streamlit front-end was built for data audit.
- **2024-08 → 2024-10:** data augmentation with gpt-4o-mini (5 synonyms per term, manually included or excluded in a Streamlit tool) gave about 11k rows. Training ran on Paperspace GPUs with MLflow and then W&B. Models were pushed to the HF Hub.
- **2024-09-05:** demo repo ([text-to-icpc2_demo](https://github.com/DiogoCarapito/text-to-icpc2_demo)). Streamlit on Streamlit Cloud, logging to Supabase, then Docker + Postgres + nginx on a Hetzner VPS (2026-04). It had a feedback system (correct / wrong / copy).
- **2025-04 → 2026-04:** BERTimbau (`neuralmind/bert-base-portuguese-cased`) with class-weighted loss became the best model: [`diogocarapito/text-to-icpc2-bert-base-portuguese-cased`](https://huggingface.co/diogocarapito/text-to-icpc2-bert-base-portuguese-cased). The reported top-1 was **0.787** (2026-04-08).

### v1 post-mortem (2026-11-21)

1. **Optimistic evaluation.** The 0.787 was measured on the full dataset, which included the training rows. There was no held-out set and no real inputs.
2. **Data distribution.** About 47% formal ICD-10 wording, about 40% gpt-4o-mini synonyms, and 35 human-written rows. No abbreviations, typos, or short notes like real problem lists.
3. **Contradictory labels.** ICD-10 codes mapped to several ICPC-2 codes (e.g. Z71.1) put the same text under different labels.
4. **No English data**, and a PT-BR-only base model.
5. **Lost usage data.** The demo server was gone by 2026-11, so the v1 interaction logs are not available.
6. **Demo bug.** "Copiar" copied to the server's clipboard instead of the user's.

## 2026-11-21 · v2 starts

**Decisions:** one multilingual model (PT-PT primary, EN); benchmark a classifier against a retriever; Claude for synthetic data; first training locally (Apple M3 Pro, MPS), then ephemeral DigitalOcean GPU Droplets; a new dedicated server for the demo; fully automatic, gated model promotion; everything public except user-submitted text.

**Dataset v2.0** (`icpc2 build-data`):
- The ICD-10 mapping now uses the longest prefix and includes ICD-10-CM child codes. **3,467 ambiguous ICD-10 codes dropped**, with a cap of 40 ICD texts per code and language.
- English from the ICD-10-CM English descriptions: about 8.6k rows.
- Hash-based splits on normalised text, plus a leakage guard. Official definitions always stay in train.

**Synthetic data**: Claude (`claude-opus-5-5`, Message Batches, structured JSON output). One request per code: 38 PT-PT examples across 5 styles (clinical / shorthand / lay / qualified / typo) and 22 EN examples across 4 styles, constrained by ICPC-2 inclusion/exclusion criteria. 686/686 requests succeeded. **40,037 examples** were kept after de-duplication and filtering (`data/processed/synthetic.jsonl`). The batch took under 10 minutes.

→ **65,750 examples, 686 codes. Minimum 54 training examples per code** (v1: 8 per code, and 1 per code before augmentation in the v2 rebuild).

**Gold test set (drafting):** Claude drafted 36 entries per chapter (17 chapters), 20% English and about 30% deliberately hard cases with a `confusable_with` code, using a prompt and task framing different from the training generator. **611 drafts covering 552 codes.** Every draft is to be verified by a family physician on a review page before it enters `gold.csv`.
- *Methods note:* 10 of 17 chapter requests were declined by a safety classifier (category `reasoning_extraction`) when each item asked for a free-text `rationale`. Without that field, all 17 completed. Rationales exist for 13 chapters only.
- *Methods note:* drafts and training synthetic data come from the same model family. Expect optimistic scores on drafts and on the LLM-drafted gold set. Physician-written entries and real usage (`feedback_test`) are needed as independent test sets.

**First honest v1 number:** the v1 checkpoint on the unverified drafts (eval-only, never trained on) scores **top-1 0.352** (PT 0.427, EN 0.042), against the 0.787 reported in 2024. On `val` it scores 0.436, but `val` includes v1's own gpt-4o-mini training rows, so that number is inflated for v1.

**Tooling:**
- `icpc2 sweep`: benchmark runner; results committed under `experiments/`.
- `icpc2 gpu run`: ephemeral DO droplet with a self-destruct timer.
- `icpc2 rescore`: re-scores saved predictions once gold is verified.

## 2026-11-22 · benchmark-01, first local runs (Apple M3 Pro, MPS)

| run | draft top-1 | top-3 | PT | EN | val top-1 | train |
|---|---|---|---|---|---|---|
| v1 checkpoint | 0.352 | 0.466 | 0.427 | 0.042 | 0.436 | – |
| clf-mmbert-small (v2 data, defaults) | **0.897** | 0.957 | 0.900 | 0.882 | 0.820 | 40 min |

- The smallest multilingual classifier trained on v2 data already improves draft top-1 by **+0.55** over the v1 checkpoint, and English goes from near-chance to 0.88. *Caveat:* the drafts share their generator (Claude) with the synthetic training data, so these numbers are likely optimistic. They will be re-scored on physician-verified gold and on physician-written entries.
- `ret-e5-base` (batch 128, 62k pairs) ran **out of memory on MPS** (18 GB unified memory). Moved to the GPU sweep.

## 2026-11-22 · gold set verified

A family physician reviewed all 611 Claude drafts on the review page (`scripts/build_gold_review.py`). Verdicts are in `data/gold/verdicts/`.

| verdict | n | % |
|---|---|---|
| correct as drafted | 600 | 98.2 |
| recoded | 2 | 0.3 |
| discarded (ambiguous / multi-problem / unrealistic) | 9 | 1.5 |

→ **602 gold entries**: `gold_test` 460 (374 PT, 86 EN) and `gold_dev` 142, split by hash of the draft id.

- The recodes reflect **Portuguese coding conventions**: "DM2 insulinotratada" → T89 (not T90), and "Fibromialgia" → A01 (not L18).
- Review pace: about 2 h for 611 items, median 2.7 s per item. *To do for the paper:* a second blind pass by the same physician or a second rater on a sample (e.g. 100 items), to report agreement (Cohen's κ).

**benchmark-01 on verified gold** (`icpc2 rescore`, predictions saved during the sweep):

| run | gold_test top-1 | top-3 | top-5 | PT | EN |
|---|---|---|---|---|---|
| v1-checkpoint | 0.346 | 0.465 | 0.511 | 0.417 | 0.035 |
| clf-mmbert-small | **0.887** | 0.954 | 0.961 | 0.896 | 0.849 |

## 2026-11-22 · benchmark-01, overnight on MPS (01:18 → 07:00)

Same data (dataset v2.0) for every run. Scores are on the physician-verified gold set (`experiments/benchmark-01/rescore_gold.csv`).

| run | gold_test top-1 | top-3 | top-5 | macro-F1 | PT | EN | gold_dev top-1 | train (M3 Pro) |
|---|---|---|---|---|---|---|---|---|
| v1-checkpoint (2024) | 0.346 | 0.465 | 0.511 | 0.271 | 0.417 | 0.035 | 0.366 | – |
| clf-mmbert-small | 0.887 | 0.954 | 0.961 | 0.831 | 0.896 | 0.849 | 0.908 | 40 min |
| clf-bertimbau-base | 0.896 | 0.941 | 0.954 | 0.845 | 0.922 | 0.779 | 0.873 | 57 min |
| **ret-e5-base** | **0.922** | **0.980** | **0.985** | **0.879** | 0.925 | **0.907** | **0.930** | 85 min |

- **The retriever leads** on every column, and its top-3 of 0.98 matters for the demo's "pick from the list" interface. It can also absorb confirmed feedback without retraining.
- BERTimbau (PT-BR only, v1's architecture) is the best classifier on PT (0.922), but loses about 0.07 on EN against the multilingual mmBERT-small. The v1 → v2 jump comes mostly from **data**, not architecture: the same BERTimbau goes from 0.346 to 0.896.
- n = 86 for EN gold_test, so EN differences under about 0.07 are within noise.

**Failed on MPS (18 GB unified memory), not on merit:**
- `clf-mmbert-base`, `clf-xlmr-base`: out of memory at batch 64. Fix: `max_device_batch: {mps: 16}` with gradient accumulation (same effective batch).
- `clf-albertina-ptpt-100m`: `Float vs BFloat16`. transformers v5 keeps the checkpoint dtype and Albertina ships bf16. Fix: load as float32.
- `ret-bge-m3`, `ret-e5-large`: out of memory even with the cached loss (mini-batch 16).
- `ret-qwen3-emb-0.6b`: ran, but at about 21 s/step (6+ h left at 28%), with the machine swapping. Stopped.

→ The 560M+ models need the GPU path.
