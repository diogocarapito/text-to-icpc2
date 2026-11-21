# text-to-ICPC2 v2

Turns free-text clinical problems or diagnoses, in **PT-PT or English**, into **ICPC-2** codes. It covers the 686 symptom and diagnosis codes and is aimed at Portuguese primary care (MGF). It also learns from how people use it: every prediction and every piece of feedback is logged, and a retrain loop promotes a better model automatically.

This is a research demo, not a medical device.

Successor to [text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2) and [text-to-icpc2_demo](https://github.com/DiogoCarapito/text-to-icpc2_demo). See [docs/PLAN.md](docs/PLAN.md) for the roadmap and the design decisions.

```
            ┌──────────── Mac (first trainings) / Modal GPU (after deploy) ───────────┐
 ACSS ICPC-2 + ICD-10 ─┐   │                                                                          │
 Claude synthetic ─────┼─► build-data ─► train (classifier | retriever) ─► evaluate vs champion ─► gates ─► HF Hub (vN)
 v1 synonyms ──────────┤   │                                                                          │        │
 user feedback ────────┘   └──────────────────────────────────────────────────────────────────────────┘        │ reload
        ▲                                                                                                        ▼
        └──────── /api/export ◄──── Postgres log ◄──── FastAPI web app (VPS, text-to-icpc2.com) ◄──── users
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
| `src/icpc2/remote/modal_app.py` | remote GPU training and nightly cron on Modal |
| `src/icpc2/web/` | FastAPI app, interaction log, export API, `/stats` usage page |
| `deploy/` | docker-compose (app, postgres, nginx, certbot, backups) for the VPS |
| `data/raw/` | ACSS ICPC-2 (PT-PT), slim ICD-10-CM PT/EN |
| `data/gold/` | **hand-labelled test set, never trained on** (private, see below) |

## Data splits

| split | source | used for |
|---|---|---|
| `train` | official descriptions, synthetic, v1 synonyms, user feedback | training |
| `val` | 8% of synthetic/v1 (hash of normalised text) | model selection, proxy metric |
| `feedback_test` | 20% of user-confirmed labels | real-world metric that grows with use |
| `gold_dev` / `gold_test` | hand-labelled real inputs | **headline metric** and promotion gate |

A leakage guard removes any training text that also appears, after normalisation, in an evaluation split. v1 had no such guard: it validated on its own training data.

**Privacy:** user-submitted text (feedback, gold) is git-ignored. It lives only in the **private** HF dataset `diogocarapito/text-to-icpc2-v2-data`. Inputs are scrubbed of long digit runs (SNS/NIF/phone numbers) before they are stored.

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

# benchmark (local MPS)
.venv/bin/icpc2 train-classifier --model jhu-clsp/mmBERT-base
.venv/bin/icpc2 train-classifier --model FacebookAI/xlm-roberta-base
.venv/bin/icpc2 train-retriever  --model intfloat/multilingual-e5-base
# reports/<run>/metrics.json + errors.csv for each

.venv/bin/icpc2 promote models/classifier/<run>/model   # first push to HF → v1

# web app locally
MODEL_PATH=models/classifier/<run>/model ICPC2_ADMIN_TOKEN=dev make serve
open http://127.0.0.1:8000  ·  http://127.0.0.1:8000/stats?token=dev
```

## Gold set (the most important thing to build)

1. `scripts/export_v1_logs.sh user@vps`: exports real inputs from the v1 demo's Postgres.
2. `icpc2 gold-template --src data/feedback/v1_logs.csv --model models/classifier/<run>/model`: creates `data/gold/to_label.csv`, pre-filled with model suggestions.
3. Correct the `code` column. Add your own realistic entries (abbreviations, typos, English). Append the rows to `data/gold/gold.csv` (format: `data/gold/gold.example.csv`).
4. `icpc2 data push`: uploads to the private HF data repo, so the remote trainer can see it.

Aim for at least 300 rows across chapters (≥100 is needed before gold becomes the promotion metric).

## Deploy (VPS)

Replaces the v1 demo on the same domain and certificates:

```bash
# on the VPS — first export the v1 logs (see above), then:
cd text-to-icpc2_demo && docker compose down          # keep v1 volumes until the export is safe
git clone https://github.com/DiogoCarapito/text-to-icpc2-v2 && cd text-to-icpc2-v2/deploy
nano .env                                             # POSTGRES_PASSWORD, ICPC2_ADMIN_TOKEN, MODEL_REPO
docker compose up -d --build
```

The certbot volumes are new in v2. Issue the certificate once, exactly as in the v1 README (`docker compose run --entrypoint certbot certbot certonly --webroot ...`).

## Automatic retraining

```bash
icpc2 loop                     # local backend: pull feedback → retrain if due → promote if gates pass
icpc2 loop --dry-run --force   # see what would happen

# after the demo is live: remote GPU on Modal, plus a nightly cron that doesn't need the Mac
pip install modal && modal setup
modal secret create icpc2 HF_TOKEN=... ICPC2_ADMIN_TOKEN=... ICPC2_VPS_URL=https://text-to-icpc2.com
modal deploy src/icpc2/remote/modal_app.py
# then set training.backend: modal in configs/loop.yaml
```

Promotion is fully automatic, with gates (`configs/loop.yaml`). The candidate must match or beat the champion on gold (or val) top-1, with no language or `feedback_test` slice regressing past its tolerance. Both models are evaluated on the same current data. Every promoted model is tagged `vN` on the HF Hub, so you can roll back by setting `MODEL_REVISION=vN`.

## Environment variables

`.env` at the repo root (local) and `deploy/.env` (VPS). Never commit either.

| var | where | purpose |
|---|---|---|
| `ANTHROPIC_API_KEY` | Mac | synthetic data (or `ant auth login`) |
| `HF_TOKEN` | Mac, Modal | push models and the private data repo |
| `ICPC2_ADMIN_TOKEN` | all | shared secret: `/api/export`, `/api/admin/reload`, `/stats` (`openssl rand -hex 32`) |
| `ICPC2_VPS_URL` | Mac, Modal | e.g. `https://text-to-icpc2.com` |
| `POSTGRES_PASSWORD` | VPS | database |
| `MODEL_REPO` / `MODEL_REVISION` | VPS | which model to serve (default `diogocarapito/text-to-icpc2-v2@main`) |
