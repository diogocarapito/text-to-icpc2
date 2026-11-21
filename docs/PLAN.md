# v2 plan

## Decisions (2026-11-21)

| topic | decision | why |
|---|---|---|
| languages | **one multilingual model** (PT-PT primary, EN) | one thing to serve and retrain; BERTimbau (v1) is PT-BR only |
| approach | **benchmark classifier vs retriever**, serve the winner | the retriever can learn new feedback instantly, without retraining |
| synthetic data | **Claude API** (Message Batches) | v1's gpt-4o-mini synonyms were formal; we need clinician shorthand, typos, lay wording, EN |
| first trainings | **local Mac M3 Pro (MPS)** | free; base-size encoders on short texts fit in 18 GB |
| triggered retrains | **remote GPU (Modal)** after deploy | runs from an API call, per-second billing, cron built in; Paperspace possible behind the same `backend` switch |
| hosting | **existing VPS + Docker**, same domain | reuses nginx/certbot/Postgres; CPU inference is enough for a ~100–300M-parameter encoder |
| promotion | **fully automatic, gated** | candidate vs champion on the same held-out data; per-language and real-user slices can't regress |
| repo | monorepo `text-to-icpc2-v2` | data, training, loop and web share one predictor code path |

## What v1 taught us

1. **Its evaluation was too optimistic.** The last run scored the full dataset, which includes the training rows (top-1 0.787). There was no held-out set, and nothing made of real inputs.
2. **The data doesn't look like real use.** About 47% is formal ICD-10 wording, 40% is gpt-4o-mini synonyms, and there are only 35 human rows: no abbreviations, typos or short notes.
3. **Contradictory labels.** ICD-10 codes such as Z71.1 map to several ICPC-2 codes, so the same text appeared with different labels. v2 drops 3,467 such codes.
4. **No English.** v2 gets about 9k English rows from the ICD-10-CM English descriptions, plus Claude-generated English.
5. **Demo bug:** `pyperclip` copied the code to the *server's* clipboard. v2 uses `navigator.clipboard`.

## Phases

### Phase 0: foundations ✅ (this commit)
- [x] Repo scaffold, configs, CLI
- [x] Dataset build with leakage guard, ambiguity filter and EN data → 27k rows
- [x] Claude synthetic generator (batch + preview)
- [x] Classifier + retriever trainers, shared predictor, evaluation
- [x] FastAPI web app with logging, feedback, export, reload and stats
- [x] Retrain loop with promotion gates; Modal remote backend
- [x] Tests (dataset leakage, gates, feedback rules, web API)

### Phase 1: data (next)
- [ ] Export v1 VPS logs (`scripts/export_v1_logs.sh`)
- [ ] **Gold set**: label ≥300 real/realistic entries (v1 logs + your own), ~20% English
- [ ] `icpc2 synth preview` on 5–10 tricky codes (K86/K87, R74/R78, P74/P76, L03/L86) and iterate on the prompt
- [ ] Full synth batch → expect ~40k new rows; spot-check 100 random rows
- [ ] Push the private data repo

### Phase 2: benchmark (local MPS)
- [ ] Classifier: mmBERT-base, XLM-R-base, Albertina PT-PT (PT reference), BERTimbau (v1 baseline)
- [ ] Retriever: multilingual-e5-base, bge-m3
- [ ] Compare on gold_test (top-1/top-3, PT vs EN, per chapter); read `errors.csv`
- [ ] Pick a winner → `loop.model_type`; `icpc2 promote` → **v1 on the HF Hub**

### Phase 3: deploy
- [ ] Stop v1, deploy v2 compose on the VPS; certificate; smoke test
- [ ] Set `ICPC2_ADMIN_TOKEN` on the VPS + Modal; check `/stats`
- [ ] Optional: ONNX/int8 export if CPU latency > 150 ms

### Phase 4: the loop
- [ ] `modal deploy` (nightly check + GPU training); `training.backend: modal`
- [ ] First automatic retrain once 50 labels have come in; read the history in `reports/loop_history.jsonl`
- [ ] Periodically move a sample of feedback into gold (labelled by you), so the gold set follows real use

## Open questions / later
- Hybrid serving: classifier probabilities blended with retriever kNN on recent feedback
- Active learning: show low-confidence inputs on `/stats` for you to label first
- Multi-label inputs ("HTA e DM2"): split on separators, or a multi-label head
- Process codes (−30…−69) for procedures — currently out of scope (components 1 and 7 only)
- Official WONCA English ICPC-2 rubrics to replace the Claude-translated `title_en`
