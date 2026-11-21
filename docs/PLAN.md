# v2 plan

See [JOURNAL.md](JOURNAL.md) for the dated record of what happened and the numbers.

## Decisions

| topic | decision | why |
|---|---|---|
| languages | **one multilingual model** (PT-PT primary, EN) | one thing to serve and retrain; BERTimbau (v1) is PT-BR only |
| approach | **benchmark classifier vs retriever**, serve the winner | the retriever can learn new feedback instantly, without retraining |
| synthetic data | **Claude API** (Message Batches) | clinician shorthand, typos, lay wording and EN, constrained by ICPC-2 criteria |
| test set | **Claude-drafted, physician-verified gold**, plus physician-written entries and real usage | v1 had no held-out set; the v1 logs were lost with the old server |
| first trainings | local Mac M3 Pro (MPS) | free; enough for smoke tests and small runs |
| benchmarks and triggered retrains | **ephemeral DigitalOcean GPU Droplets** | started from the API; GPU droplets bill while off, so each one is destroyed after its job |
| hosting | **new dedicated server**, Docker + Caddy, text-to-icpc2.com | the old v1 server is gone; isolated from other projects |
| promotion | **fully automatic, gated** | candidate vs champion on the same held-out data; no language or real-user slice may regress |
| openness | everything public except user-typed text | open science: the code, data, experiments and models back the paper |

## Phases

### Phase 0: foundations ✅
Scaffold, dataset build (leakage guard, ambiguity filter, EN), trainers, shared predictor, evaluation, web app, retrain loop, tests.

### Phase 1: data ✅ (verification pending)
- [x] Claude synthetic data for all 686 codes: 40,037 examples
- [x] 611 gold drafts across all 17 chapters
- [ ] **Physician verification** on the review page, then `icpc2 gold-import`
- [ ] Physician-written entries (aim for 100 or more; the least LLM-biased test set)

### Phase 2: benchmark
- [x] Sweep runner, DO GPU runner, v1 checkpoint as baseline (top-1 0.352 on the drafts)
- [ ] `benchmark-01`: 10 models, default hyper-parameters (GPU)
- [ ] `benchmark-02`: hyper-parameter search on the best 2–3
- [ ] `icpc2 rescore` on verified gold; pick the winner; `icpc2 promote` → **v1 on the HF Hub**; `icpc2 publish-dataset`

### Phase 3: deploy
- [ ] New server, DNS, `docker compose up`; smoke test; `/stats`
- [ ] Optional: ONNX/int8 if CPU latency is over 150 ms

### Phase 4: the loop and the paper
- [ ] Scheduler live (daily check → GPU retrain → gated promotion)
- [ ] Periodically move a sample of real inputs into gold (physician-labelled)
- [ ] Write-up of the 2024 → 2026 journey; paper draft (methods: data pipeline, LLM-drafted/physician-verified evaluation, v1 vs v2, classifier vs retriever, online learning from usage)

## Open questions / later
- Hybrid serving: classifier probabilities blended with retriever kNN on recent feedback
- Active learning: surface low-confidence real inputs for labelling first
- Multi-problem inputs ("HTA e DM2"): split, or a multi-label head
- Process codes (−30…−69) are out of scope for now
- Official WONCA English rubrics to replace the Claude-generated `title_en`
