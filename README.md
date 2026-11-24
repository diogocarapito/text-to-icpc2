# text-to-ICPC2 v2

Turns free-text clinical problems or diagnoses, in **PT-PT or English**, into **ICPC-2** codes. It covers the 686 symptom and diagnosis codes and is aimed at Portuguese primary care (MGF). It also learns from how people use it: every prediction and every piece of feedback is logged, and a retrain loop promotes a better model automatically.

This is a research demo, not a medical device.

This is v2 of the project, developed in [text-to-icpc2-v2](https://github.com/diogocarapito/text-to-icpc2-v2) (now archived) and merged back here with its full history. The v1 code (2024–25 BERT classifier) is kept at tag [`v1-legacy`](https://github.com/diogocarapito/text-to-icpc2/tree/v1-legacy); the old Streamlit demo is [text-to-icpc2_demo](https://github.com/DiogoCarapito/text-to-icpc2_demo). See [docs/PLAN.md](docs/PLAN.md) for the roadmap and the design decisions.

**Current model:** [`diogocarapito/text-to-icpc2`](https://huggingface.co/diogocarapito/text-to-icpc2) `v1`, a multilingual-e5-base retriever: **0.922 top-1 / 0.985 top-5** on 460 physician-verified entries (PT 0.925, EN 0.907). The 2024 model scores 0.346 on the same set.

```
            ┌──────── Mac (MPS) / ephemeral DigitalOcean GPU Droplet (created per job) ────────┐
 ACSS ICPC-2 + ICD-10 ─┐   │                                                                          │
 Claude synthetic ─────┼─► build-data ─► train (classifier | retriever) ─► evaluate vs champion ─► gates ─► HF Hub (vN)
 v1 synonyms ──────────┤   │                                                                          │        │
 user feedback ────────┘   └──────────────────────────────────────────────────────────────────────────┘        │ reload
        ▲                                                                                                        ▼
        └──────── /api/export ◄──── Postgres log ◄──── FastAPI web app (text-to-icpc2.com) ◄──── users
```

## Layout

| path | what |
|---|---|
| `configs/` | YAML configs: `data`, `synth`, `classifier`, `retriever`, `loop` |
| `src/icpc2/data/` | `build.py` (dataset and splits), `synth.py` (Claude batches), `feedback.py` (logs → labels) |
| `src/icpc2/train/` | `classifier.py` (encoder + softmax), `retriever.py` (contrastive embeddings + kNN) |
| `src/icpc2/predict.py` | one predictor interface, shared by evaluation **and** serving |
| `src/icpc2/eval/` | metrics per split, language and chapter; error dumps |
| `src/icpc2/loop/` | trigger check → train → champion comparison → automatic promotion |
| `src/icpc2/remote/do_gpu.py` | ephemeral DigitalOcean GPU Droplets: create → train → fetch → destroy |
| `src/icpc2/sweep.py` | benchmark sweeps (`configs/sweeps/`), results in `experiments/` |
| `experiments/` | **public lab results**: every benchmark run's metrics, config, hardware, predictions |
| `docs/JOURNAL.md` | dated lab notebook, 2024 → today |
| `docs/WRITEUP.md` | article and paper material: methods, results with CIs, error analysis, threats to validity |
| `docs/DEPLOY.md` | production checklist for text-to-icpc2.com |
| `src/icpc2/web/` | FastAPI app, interaction log, public API, hidden `/painel` dashboard; page text in `web/content/{pt,en}.md` |
| `deploy/` | docker-compose for the server: app, Postgres, Caddy (automatic HTTPS), backups, retrain scheduler |
| `scripts/` | gold review page builder, v1 log export |
| `data/raw/` | ACSS ICPC-2 (PT-PT), slim ICD-10-CM PT/EN |
| `data/gold/` | Claude-drafted, **physician-verified** test set (`drafts.jsonl` → review → `gold.csv`), never trained on |

## Data splits

| split | source | used for |
|---|---|---|
| `train` | official descriptions, synthetic, v1 synonyms, user feedback | training |
| `val` | 8% of synthetic/v1 (hash of normalised text) | model selection, proxy metric |
| `feedback_test` | 20% of user-confirmed labels | real-world metric that grows with use |
| `draft` | unverified gold drafts | eval-only proxy until verified |
| `gold_dev` / `gold_test` | physician-verified entries | **headline metric** and promotion gate |

A leakage guard removes any training text that also appears, after normalisation, in an evaluation split. v1 had no such guard: it validated on its own training data.

**Public by default; private only where users typed:** code, configs, synthetic data, the gold set, experiment results and models are public (GitHub, the [HF dataset](https://huggingface.co/datasets/diogocarapito/text-to-icpc2) and the [HF model](https://huggingface.co/diogocarapito/text-to-icpc2)). The demo's interaction logs contain text that users typed, so they live only in the **private** HF repo `diogocarapito/text-to-icpc2-v2-feedback`. Inputs are scrubbed of e-mails, full dates and long digit runs (SNS/NIF/phone numbers) before they are stored. User corrections change the model only after review: confirmations of the 1st suggestion need several independent sessions, any other code needs the physician's approval in the dashboard.

## Quick start (Mac)

```bash
make setup                      # .venv with train + synth + serve + dev extras
cp ~/somewhere/.env .env        # see "Environment variables" below
make data                       # build data/processed/examples.parquet
make test

# synthetic data (Claude)
.venv/bin/icpc2 synth preview --code K86     # look at quality for one code first
.venv/bin/icpc2 synth submit                 # batch for all 686 codes (50% price)
.venv/bin/icpc2 synth collect --wait && make data

# benchmark: a whole sweep on a fresh GPU droplet (destroyed afterwards)
.venv/bin/icpc2 gpu sizes
.venv/bin/icpc2 gpu run --size gpu-l40sx1-48gb -- icpc2 sweep configs/sweeps/benchmark-01.yaml
cat experiments/benchmark-01/summary.csv
# ...or locally:  .venv/bin/icpc2 sweep configs/sweeps/smoke.yaml

.venv/bin/icpc2 promote models/classifier/<run>/model   # first push to HF → v1
.venv/bin/icpc2 publish-dataset                          # public HF dataset

# web app locally
MODEL_PATH=models/classifier/<run>/model ICPC2_ADMIN_TOKEN=dev make serve
open http://127.0.0.1:8000  ·  http://127.0.0.1:8000/painel   # code: dev
```

## Gold set

1. `icpc2 gold-draft`: Claude drafts about 36 realistic entries per chapter (20% EN, about 30% hard cases), saved to `data/gold/drafts.jsonl`.
2. A physician verifies each draft on the review page (`scripts/build_gold_review.py` builds it as a claude.ai artifact). Each draft is marked correct, recoded, discarded or "unsure".
3. Export the verdicts and run `icpc2 gold-import <dir>` to write `data/gold/gold.csv` (80% test / 20% dev). Then `icpc2 rescore benchmark-01` re-scores every saved prediction without retraining.
4. Add physician-written entries straight to `gold.csv` (`source=manual`). They are the test set least biased towards the LLM that wrote the training data.

## Deploy (dedicated server)

A small Ubuntu server with Docker. Caddy gets HTTPS certificates automatically once the domain's DNS points at the server.

```bash
git clone https://github.com/diogocarapito/text-to-icpc2 && cd text-to-icpc2/deploy
nano .env        # DOMAIN, ACME_EMAIL, POSTGRES_PASSWORD, ICPC2_ADMIN_TOKEN, HF_TOKEN, DIGITALOCEAN_TOKEN
docker compose up -d --build
```

Services: `app` (FastAPI + model), `db` (Postgres), `caddy` (TLS), `backup` (nightly pg_dump, 14 days) and `scheduler` (daily retrain check).

## Automatic retraining

```bash
icpc2 loop                      # pull feedback → retrain if due → promote if the gates pass
icpc2 loop --backend do         # same, but training runs on an ephemeral DigitalOcean GPU Droplet
icpc2 loop --dry-run --force    # see what would happen
```

In production, the `scheduler` container runs `icpc2 loop --backend do` once a day. Every droplet is destroyed after its job, and a timer on the droplet destroys it anyway after `gpu.max_hours`.

Promotion is fully automatic, with gates (`configs/loop.yaml`). The candidate must match or beat the champion on gold (or val) top-1, with no language or `feedback_test` slice regressing past its tolerance. Both models are evaluated on the same current data. Every promoted model is tagged `vN` on the HF Hub, so you can roll back by setting `MODEL_REVISION=vN`.

## Environment variables

`.env` at the repo root (local) and `deploy/.env` (server). Never commit either.

| var | where | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Mac | synthetic data and gold drafts (or `ant auth login`) |
| `HF_TOKEN` | Mac, server | push models and datasets; private feedback repo |
| `DIGITALOCEAN_TOKEN` | Mac, server | GPU droplets (create, read and delete droplets; read SSH keys) |
| `ICPC2_ADMIN_TOKEN` | all | shared secret: `/api/export`, `/api/admin/reload` (Bearer) and the `/painel` dashboard code (`openssl rand -hex 32`) |
| `ICPC2_VPS_URL` | Mac, server | e.g. `https://text-to-icpc2.com` |
| `DOMAIN`, `ACME_EMAIL` | server | Caddy site and Let's Encrypt contact |
| `POSTGRES_PASSWORD` | server | database |
| `MODEL_REPO` / `MODEL_REVISION` | server | which model to serve (default `diogocarapito/text-to-icpc2@main`) |
