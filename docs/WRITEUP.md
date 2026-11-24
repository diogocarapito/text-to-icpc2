# Write-up material: text-to-icpc2, 2024 → 2026

Source material for (1) the public journey article and (2) the paper. It collects the facts, numbers, decisions and caveats in one place; the dated detail is in [JOURNAL.md](JOURNAL.md). Places marked **[cite]** need a reference; **[todo]** is work still to do before submission.

## 1. One-paragraph summary (draft)

Family doctors in Portugal code every problem in ICPC-2. Typing free text and choosing the code by hand is slow and inconsistent **[cite]**. In 2024 we fine-tuned a Portuguese BERT on official ICPC-2 and ICD-10 descriptions plus LLM synonyms, and reported 78.7% top-1 accuracy. On a physician-verified test set built in 2026, that model scores **34.6%**: the 2024 number was measured on its own training data. Version 2 rebuilds the dataset (Claude-generated clinical shorthand, lay wording, typos and English; leakage-guarded splits; ambiguous ICD mappings removed), benchmarks classifiers against an embedding retriever, and evaluates on 460 Claude-drafted, physician-verified entries. The best model, a fine-tuned multilingual-e5-base retriever, reaches **92.2% top-1 (95% CI 89.6–94.6) and 98.5% top-5**, with 90.7% on English. It is served publicly with a feedback loop that retrains and promotes models automatically behind regression gates. Code, data, experiments and models are open.

## 2. Timeline

| date | event |
|---|---|
| 2024-02-26 | v1 starts (repo `text-to-icpc2`) |
| 2024-03 → 07 | ETL from ACSS ICPC-2 v4.4 and ICD-10-CM PT; Streamlit data-audit tool |
| 2024-08 → 10 | gpt-4o-mini augmentation (5 synonyms per term, manually filtered), ~11k rows; Paperspace GPUs; MLflow → W&B |
| 2024-09-05 | demo repo `text-to-icpc2_demo` (Streamlit; Supabase logging; later Docker + Postgres + nginx on a VPS) |
| 2025-04 → 2026-04 | BERTimbau + class-weighted loss becomes the best v1 model; reported top-1 0.787 (2026-04-08) |
| 2026-11 | the v1 server is gone and so are its usage logs; v1 post-mortem |
| 2026-11-21 | v2: dataset v2.0, 40,037 Claude synthetic examples, 611 gold drafts |
| 2026-11-22 | the physician verifies the gold set (602 kept); benchmark-01 overnight on a laptop; **v1 of the v2 model published** on the HF Hub; public dataset published |

## 3. v1 post-mortem (the "why" of v2)

1. **Optimistic evaluation:** 0.787 was measured on data that included the training rows. There was no held-out set.
2. **Unrealistic inputs:** about 47% formal ICD-10 wording and about 40% gpt-4o-mini synonyms. Only 35 human-written rows, with no abbreviations, typos or problem-list shorthand.
3. **Contradictory labels:** ICD-10 codes that map to several ICPC-2 codes put identical text under different labels.
4. **No English**, and a PT-BR base model for PT-PT text.
5. **Lost usage data:** the demo logs died with the server. This is the reason v2 backs up feedback to a private HF repo.
6. **Demo bug:** "copy" copied to the server's clipboard.

The single most useful lesson for the paper: **the same architecture (BERTimbau) trained on v2 data goes from 0.346 to 0.896 on the same gold test.** Most of the gain comes from the data, not the model.

## 4. Methods

### 4.1 Label space
ICPC-2 components 1 (symptoms) and 7 (diagnoses): **686 codes**. Process codes (−30…−69) are out of scope. Source: ACSS ICPC-2e v4.4 (PT-PT), with titles, inclusion and exclusion terms, criteria and the ICD-10 mapping. English titles: currently Claude-generated **[todo: replace with official WONCA English rubrics]**.

### 4.2 Training data (dataset v2.0, `icpc2 build-data`)

| source | train rows | notes |
|---|---|---|
| Claude synthetic, 5 styles (PT) / 4 styles (EN) | 34,691 | shorthand 9,917 · clinical 9,285 · lay 6,334 · qualified 6,253 · typo 2,902 |
| ICD-10-CM descriptions, PT | 8,930 | longest-prefix mapping incl. child codes; **3,467 ambiguous ICD codes dropped**; ≤40 per code and language |
| ICD-10-CM descriptions, EN | 8,994 | same mapping |
| v1 gpt-4o-mini synonyms | 6,707 | 173 of them human-edited |
| ICPC-2 official texts | 2,794 | titles, short titles, inclusion terms, EN titles; **always in train** |
| v1 human-written | 33 | |
| **total train** | **62,149** | PT 39,418 / EN 22,731; **min 54 per code**, median 79 (v1: 8 per code) |
| val | 3,568 | hash split (8%) of the same distribution |

**Synthetic generation:** `claude-opus-5-5` via the Message Batches API with structured JSON output. One request per code (686/686 succeeded) produced 38 PT-PT examples (clinical 10, shorthand 10, lay 6, qualified 6, typo 6) and 22 EN examples (8/6/4/4), with the code's inclusion terms, exclusions and criteria in the prompt. Post-filter: exact or normalised duplicates and texts equal to another code's title are dropped, leaving 40,037 kept. The batch took under 10 minutes and cost **$14.79** (974k input / 1.28M output tokens).

**Lineage from v1:** 71% of the v1 dataset (12,832 of 18,052 rows) is in v2: all of its human-reviewed synonyms and official ICPC-2 texts, but only 39% of its ICD-10 descriptions, which were rebuilt with stricter rules. v1 makes up 19% of v2's train+val rows. Details: [memoire/01-dataset-lineage.md](memoire/01-dataset-lineage.md).

**Splits and leakage guard:** each text is normalised (`norm_key`: lowercase, accents and punctuation stripped). Splits come from a hash of the normalised text with a fixed seed. Priority `gold_test > gold_dev > feedback_test > val > train`: if a normalised text appears in a higher split, it is removed from the lower ones. Official ICPC-2 texts are exempt and always stay in train.

### 4.3 Gold evaluation set
- **Drafting:** Claude (a different prompt and task framing from the training generator) wrote 36 entries per chapter across 17 chapters, 20% English and about 30% deliberately hard cases with a `confusable_with` code, giving 611 drafts covering 552 codes.
- **Verification:** one family physician reviewed every draft on a purpose-built page (accept / recode / discard), taking about 2 h with a median of 2.7 s per item.

| verdict | n | % |
|---|---|---|
| accepted as drafted | 600 | 98.2 |
| recoded | 2 | 0.3 |
| discarded (ambiguous, multi-problem, unrealistic) | 9 | 1.5 |

- Recodes reflect Portuguese coding practice: insulin-treated T2DM → T89; fibromyalgia → A01.
- **602 entries**, split by hash of the draft id: **gold_test 460** (374 PT / 86 EN; 161 hard) and **gold_dev 142**.
- *Methods note (refusals):* 10 of 17 drafting requests were declined by a safety classifier (`reasoning_extraction`) when each item carried a free-text rationale field. Without that field, all 17 completed. The prompt was not reworded to get around it.

### 4.4 Models
- **Classifier:** encoder + linear head over 686 codes; weighted cross-entropy (sqrt inverse-frequency weights), label smoothing 0.1; 10 epochs, lr 5e-5, effective batch 64, max length 64 tokens; best epoch by val top-1.
- **Retriever:** multilingual embedding model fine-tuned with in-batch-negatives contrastive loss (MultipleNegativesRankingLoss; cached/GradCache variant on MPS). Pairs are (example text → its code's anchor text); no-duplicate batch sampler; 3 epochs, lr 3e-5, batch 128. Prediction: kNN (k = 25) over code-description anchors plus every labelled training example, scores aggregated per code. New confirmed examples can be added to the index **without retraining**.
- **Baseline:** the published v1 checkpoint, evaluated as is (eval-only).

### 4.5 Evaluation
Top-1/3/5 accuracy and macro-F1, overall and per language, on `val`, `gold_dev` and `gold_test`. 95% CIs from 10,000 bootstrap resamples, and paired bootstrap differences against the best model (`scripts/gold_stats.py`). Model selection uses `gold_dev` and `val`. `gold_test` is reported.

### 4.6 Serving, feedback and the loop
- **Serving:** FastAPI app with Postgres. Every request is logged with the masked text (e-mails, full dates and long or grouped digit runs replaced), the top-5, latency and model version. Users can give feedback: correct / wrong / corrected (with the right code) / copy. Feedback needs a per-request key returned only to the requester, is rate-limited, and is accepted once per kind per request.
- **Review before learning:** feedback never changes the model directly. A (text, code) pair reaches the live retriever index, and the training labels, only if (a) it confirms the model's own 1st suggestion and ≥3 distinct browser sessions agree, or (b) the physician approves it in the admin dashboard (any code other than the 1st suggestion). Session ids are client-chosen, so (a) is safe only because agreeing with the 1st suggestion cannot change it. Corrections outside the top-5 or in another chapter are flagged as strong.
- **Labels from feedback:** reviewed pairs only; one vote per browser session per text, weighted (corrected 3, correct 2, copy 1); ties are dropped. 20% of feedback texts (by hash) go to a `feedback_test` split that is never trained on. Because unreviewed corrections are excluded, `feedback_test` depends on the physician reviewing corrections: otherwise it holds mostly agreements and flatters the champion.
- **Retraining:** triggered by ≥50 new labels or 30 days. The candidate is trained on the current data and evaluated together with the champion on the same splits.
- **Promotion gates:** the candidate must match or beat the champion's top-1, with no regression beyond PT 0.01, EN 0.02 or `feedback_test` 0.01. Promoted models are tagged `vN` on the HF Hub, so rollback is a single change of revision.
- **Privacy:** user text goes only to a private HF repo. No IPs and no accounts are stored, and the proxy keeps no access log. The session cookie lasts 30 days. Names and addresses are not detected automatically; the page asks users not to type them.

## 5. Results

### 5.1 benchmark-01 (default hyper-parameters, Apple M3 Pro 18 GB, MPS)

| model | params | type | gold_test top-1 [95% CI] | top-3 | top-5 | macro-F1 | PT | EN | hard (n=161) | gold_dev | val | train |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 checkpoint (2024) | 110M | clf | 0.346 [0.302, 0.389] | 0.465 | 0.511 | 0.271 | 0.417 | 0.035 | 0.261 | 0.366 | 0.436* | – |
| mmBERT-small | 140M | clf | 0.887 [0.857, 0.915] | 0.954 | 0.961 | 0.831 | 0.896 | 0.849 | 0.839 | 0.908 | 0.820 | 40 min |
| BERTimbau-base | 110M | clf | 0.896 [0.867, 0.922] | 0.941 | 0.954 | 0.845 | **0.922** | 0.779 | 0.863 | 0.873 | 0.805 | 57 min |
| **multilingual-e5-base** | 278M | **ret** | **0.922 [0.896, 0.946]** | **0.980** | **0.985** | **0.879** | **0.925** | **0.907** | **0.901** | **0.930** | **0.861** | 85 min |

\* inflated: v1 was trained on part of `val`'s gpt-4o-mini rows.

Paired bootstrap on gold_test, e5-base minus the other model (b = only e5 correct, c = only the other correct):
- vs BERTimbau: **+0.026 [−0.002, +0.057]**, b=31, c=19. *Not significant at 95%.*
- vs mmBERT-small: +0.035 [+0.007, +0.065], b=32, c=16.
- vs v1: +0.576 [+0.528, +0.624], b=274, c=9.

Models that did not finish on 18 GB MPS (out of memory, a dtype bug, or too slow): mmBERT-base, XLM-R-base, Albertina-PT-PT-100M, bge-m3, e5-large, Qwen3-Embedding-0.6B. **[todo: rerun on a GPU]**

### 5.2 Error analysis (e5-base, gold_test, 36 errors)
- **27 of 36** errors have the right code in the top 3; **21 of 36** stay within the right chapter.
- The top-1 score separates right from wrong answers well: median 0.96 for correct, 0.63 for wrong. This supports an "uncertain" flag or abstention threshold in the UI **[todo: calibration / risk-coverage curve]**.
- Error themes:
  - **Psychological vs social problems:** grief → Z15 instead of P02; anxiety about a son's illness → Z18 instead of P01.
  - **Symptom vs diagnosis:** weak stream and nocturia → U05 instead of Y06; painful shoulder → L92 instead of L08.
  - **Sex-specific chapters:** gonococcal urethritis → X71 instead of Y71.
  - **Intentional overdose:** P77 vs A84, in both directions.
  - **Pregnancy vs female genital:** amenorrhoea with a negative pregnancy test → W01 instead of X05.
- Some gold labels are themselves debatable (e.g. benzodiazepine overdose: A84 vs P77). This is an argument for a second rater (κ).
- Seen in the live demo (outside gold): "lombalgia c/ irradiação MIE" → L03 (0.97), where L86 is correct. "dm2 insulinotratada" → T90 (0.78), with T89 second, whereas Portuguese practice (per the reviewer) is T89.

### 5.3 Speed and footprint
- The served model is 1.1 GB (fp32). On the M3 Pro's CPU, serving latency is **27–45 ms per request**, measured locally on 2026-11-22. **[todo: measure on the production server; int8/ONNX if > 150 ms]**

## 6. Threats to validity (for the discussion section)
1. **Same generator family:** the gold drafts and the synthetic training data both come from Claude, so the gold scores are probably optimistic. Mitigations still to do: physician-written entries, and real usage (`feedback_test`).
2. **Single rater, fast review:** a median of 2.7 s per item, and 98.2% accepted. **[todo: second blind pass or second rater on about 100 items; report Cohen's κ]**
3. **Small EN set:** 86 items, so EN differences under about 0.07 are noise.
4. **Hyper-parameters:** benchmark-01 used defaults for every model. The ranking might change after tuning (benchmark-02), and the e5 vs BERTimbau gap is not significant.
5. **Single-code outputs:** inputs that mention several problems are out of scope.
6. **English titles** are Claude-generated, not the official WONCA rubrics.
7. **ICPC-2 rule vs Portuguese practice:** some codes follow local convention rather than the ICPC-2 rubric (insulin-treated T2DM → T89 rather than T90; fibromyalgia → A01 rather than L18). Gold follows Portuguese practice, so a model faithful to the rubric scores lower on these items, and vice versa. **[todo: keep these divergences as a separate tagged test slice and report them apart]**

## 7. Engineering notes worth a paragraph (or an appendix)
- **Training on a laptop:** an M3 Pro (18 GB unified memory) trains base-size models in 40–85 min. Models of 560M parameters or more run out of memory, even with GradCache. Fixes that worked: gradient accumulation (same effective batch), eager attention (MPS SDPA has no dropout), and fp32 loading (transformers v5 keeps the checkpoint dtype).
- **Ephemeral cloud GPUs:** DigitalOcean GPU droplets bill while powered off, so the runner destroys them, and a self-destruct timer runs on the droplet itself. Attempts so far cost about $1, and the droplet booted without an NVIDIA driver. **[todo: resolve]**
- **Costs to date:** Claude synthetic data $14.79, gold drafting and previews about $5–8 (estimate), DigitalOcean about $1. Training cost $0 (laptop).

## 8. To do before submission
- [ ] Second rater / blind second pass on about 100 gold items, with κ (hide the drafting rationale during that pass)
- [ ] Separate test slice for ICPC-2-rule vs PT-practice divergences (T89/T90, A01/L18, …), reported apart from gold
- [ ] 100+ physician-written entries as an independent test set
- [ ] Real-usage evaluation (`feedback_test`) after a few weeks live
- [ ] benchmark-02 on a GPU: the large models, plus tuning of the top 2–3
- [ ] Calibration and risk-coverage curve; per-chapter accuracy figure
- [ ] Official WONCA English rubrics
- [ ] Ask WONCA about licensing for publishing the ICPC-2 rubrics in the dataset
- [ ] Related work **[cite]**: automatic ICPC-2 and ICD coding, Portuguese clinical NLP (BERTimbau, Albertina), LLM-generated training data, LLM-drafted / human-verified evaluation sets
- [ ] Venue shortlist (e.g. a medical informatics journal or conference; a Portuguese primary-care journal for the clinical audience)

## 9. Figures to make
1. Pipeline diagram: sources → build (leakage guard) → train → evaluate → serve → feedback → retrain/promote.
2. v1 reported vs v1 honest vs v2 (bar chart; the "evaluation inflation" story).
3. benchmark-01: gold_test top-1 with CIs per model, split by PT and EN.
4. Error-themes table, or a confusion matrix between chapters.
5. Usage over time and the feedback mix (after launch).

## 10. Artefacts
- Code, experiments, journal: https://github.com/diogocarapito/text-to-icpc2
- Model: https://huggingface.co/diogocarapito/text-to-icpc2 (tag `v1` = e5-base retriever, trained at commit `ab6069d`)
- Dataset: https://huggingface.co/datasets/diogocarapito/text-to-icpc2
- v1 model: https://huggingface.co/diogocarapito/text-to-icpc2-bert-base-portuguese-cased
- Per-run metrics and predictions: `experiments/benchmark-01/*/`; gold re-scoring: `rescore_gold.csv`; CIs: `gold_stats.json`
