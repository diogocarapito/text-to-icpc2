.PHONY: setup data synth-preview synth train-clf train-ret test lint serve loop docker-local

setup:            ## local dev env (Mac): training + synth + serve + dev tools
	python3.12 -m venv .venv && .venv/bin/pip install -U pip && \
	.venv/bin/pip install -e ".[train,synth,serve,dev]"

data:             ## build data/processed/examples.parquet
	.venv/bin/icpc2 build-data

synth-preview:
	.venv/bin/icpc2 synth preview --code K86

synth:            ## submit a full Claude batch (all 686 codes)
	.venv/bin/icpc2 synth submit

train-clf:
	.venv/bin/icpc2 train-classifier

train-ret:
	.venv/bin/icpc2 train-retriever

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check src tests

serve:            ## local web app (MODEL_PATH=models/.../model make serve)
	.venv/bin/icpc2 serve --reload

loop:
	.venv/bin/icpc2 loop

docker-local:     ## the web image alone, no nginx/postgres (SQLite inside the container)
	docker build -t icpc2-web . && docker run --rm -p 8000:8000 -e ICPC2_ADMIN_TOKEN=dev icpc2-web
