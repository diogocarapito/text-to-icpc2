import json
import logging
from pathlib import Path

import typer
from dotenv import load_dotenv

from icpc2.config import load_config, path

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

app = typer.Typer(no_args_is_help=True, help="text-to-icpc2 v2")
synth_app = typer.Typer(no_args_is_help=True, help="Synthetic data with the Claude API")
data_app = typer.Typer(no_args_is_help=True, help="Sync private data with the HF Hub")
app.add_typer(synth_app, name="synth")
gpu_app = typer.Typer(no_args_is_help=True, help="Ephemeral DigitalOcean GPU Droplets")
app.add_typer(data_app, name="data")
app.add_typer(gpu_app, name="gpu")


# ---------------------------------------------------------------- data
@app.command("build-data")
def build_data():
    """Build data/processed/examples.parquet from all sources."""
    from icpc2.data.build import build, summarize

    typer.echo(summarize(build()))


@synth_app.command("preview")
def synth_preview(code: str = "K86"):
    """One synchronous call for a single code — check quality before a full batch."""
    from icpc2.data.synth import preview

    typer.echo(json.dumps(preview(code.upper()), indent=2, ensure_ascii=False))


@synth_app.command("submit")
def synth_submit(codes: str | None = typer.Option(None, help="comma-separated, default all")):
    from icpc2.data.synth import submit

    typer.echo(submit(codes.upper().split(",") if codes else None))


@synth_app.command("status")
def synth_status():
    from icpc2.data.synth import status

    for s in status():
        typer.echo(f"{s['id']}  {s['status']:12s} {s['counts']}  collected={s['collected']}")


@synth_app.command("collect")
def synth_collect(wait: bool = typer.Option(False, help="block until batches end")):
    from icpc2.data.synth import collect

    typer.echo(f"{collect(wait=wait)} new examples")


@data_app.command("push")
def data_push():
    from icpc2.hub import push_data

    push_data("manual push")


@data_app.command("pull")
def data_pull():
    from icpc2.hub import pull_data

    pull_data()


@app.command("pull-feedback")
def pull_feedback():
    """Fetch new interaction logs from the web app and derive labels."""
    from icpc2.data import feedback

    feedback.pull()
    feedback.derive_labels()


@app.command("gold-draft")
def gold_draft(n_per_chapter: int = 36, en_share: float = 0.2, rationale: bool = True):
    """Claude drafts gold test candidates (data/gold/drafts.jsonl) for physician verification."""
    from icpc2.data.gold import draft

    typer.echo(f"{draft(n_per_chapter, en_share, with_rationale=rationale)} drafts")


@app.command("gold-template")
def gold_template(src: Path = Path("data/feedback/v1_logs.csv"), model: str | None = None,
                  n: int = 500):
    """Make data/gold/to_label.csv from real inputs (v1 logs / events), pre-filled with
    model suggestions. Fix the `code` column, then append the rows to data/gold/gold.csv."""
    import pandas as pd

    from icpc2.text import as_icpc2_code, norm_key

    df = pd.read_csv(path(src), dtype=str).fillna("")
    col = "text_input" if "text_input" in df else "text"
    df = df[[col]].rename(columns={col: "text"})
    df = df[~df["text"].map(lambda t: bool(as_icpc2_code(t))) & (df["text"].str.len() > 1)]
    df = df.assign(k=df["text"].map(norm_key)).drop_duplicates("k").drop(columns="k").head(n)
    df["code"], df["model_suggestion"] = "", ""
    if model:
        from icpc2.predict import load_predictor

        preds = load_predictor(model).predict(df["text"].tolist(), k=3)
        df["model_suggestion"] = [" | ".join(f"{c} {s:.2f}" for c, s in p) for p in preds]
        df["code"] = [p[0][0] for p in preds]
    df["lang"], df["split"], df["source"], df["notes"] = "pt", "test", "v1_logs", ""
    out = path("data/gold/to_label.csv")
    df.to_csv(out, index=False)
    typer.echo(f"{len(df)} rows -> {out}")


# ---------------------------------------------------------------- train / eval
@app.command("train-classifier")
def train_classifier(model: str | None = None, epochs: int | None = None,
                     limit: int | None = typer.Option(None, help="subsample for smoke tests"),
                     evaluate: bool = True):
    from icpc2.train.classifier import train

    d = train(model, epochs, limit)
    if evaluate:
        _eval(str(d))


@app.command("train-retriever")
def train_retriever(model: str | None = None, epochs: int | None = None,
                    limit: int | None = None, evaluate: bool = True):
    from icpc2.train.retriever import train

    d = train(model, epochs, limit)
    if evaluate:
        _eval(str(d))


def _eval(model: str, revision: str | None = None):
    from icpc2.eval.evaluate import evaluate, headline, report_dir
    from icpc2.predict import load_predictor
    from icpc2.train.common import pick_device

    pred = load_predictor(model, revision=revision, device=pick_device("auto"))
    name = Path(model).parent.name if Path(model).exists() else model.replace("/", "_")
    rep = evaluate(pred, report_dir(name))
    typer.echo(headline(rep))


@app.command("evaluate")
def evaluate_cmd(model: str, revision: str | None = None):
    """Evaluate a local model dir or HF repo on all held-out splits."""
    _eval(model, revision)


@app.command("promote")
def promote(model_dir: Path, force: bool = typer.Option(False, help="skip gates")):
    """Evaluate a trained model against the current champion and push it to HF if it wins."""
    from icpc2 import hub
    from icpc2.eval.evaluate import evaluate, report_dir
    from icpc2.loop.retrain import _gates, _notify_reload
    from icpc2.predict import load_predictor
    from icpc2.train.common import pick_device

    cfg = load_config("loop")
    dev = pick_device("auto")
    cand = evaluate(load_predictor(str(model_dir), device=dev), report_dir("promote-candidate"))
    v = hub.champion_version()
    champ = evaluate(load_predictor(cfg["hf"]["model_repo"], "main", dev), report_dir("champion")) if v else None
    ok, reasons = _gates(cand, champ, cfg)
    typer.echo("\n".join(reasons))
    if ok or force:
        typer.echo(f"Promoted {hub.push_model(model_dir, v + 1, cand)}")
        _notify_reload()
    else:
        typer.echo("Not promoted.")


@app.command("loop")
def loop(backend: str | None = typer.Option(None, help="local | do (default from loop.yaml)"),
         force: bool = False, dry_run: bool = False,
         no_pull: bool = typer.Option(False, help="don't pull feedback from the web app")):
    """Check feedback, retrain if due, promote automatically if the gates pass."""
    from icpc2.loop.retrain import run

    backend = backend or load_config("loop")["training"]["backend"]
    typer.echo(json.dumps(run(backend, force=force, pull=not no_pull, dry_run=dry_run), indent=2))


@app.command("sweep")
def sweep(spec: str, only: str | None = typer.Option(None, help="comma-separated run names"),
          keep_models: int = 1):
    """Train + evaluate every run in a sweep spec; results -> experiments/<sweep>/."""
    from icpc2.sweep import run_sweep

    df = run_sweep(spec, only.split(",") if only else None, keep_models)
    cols = [c for c in df.columns if c in ("run", "train_min") or c.endswith("_top1") and c.count("_") == 1]
    typer.echo(df[cols].to_string(index=False) if not df.empty else "no runs")


@app.command("rescore")
def rescore_cmd(sweep_name: str):
    """Re-score a sweep's saved predictions on the physician-verified gold.csv."""
    from icpc2.sweep import rescore

    typer.echo(rescore(sweep_name).to_string(index=False))


@gpu_app.command("run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def gpu_run(ctx: typer.Context, size: str | None = None, region: str | None = None,
            keep: bool = typer.Option(False, help="don't destroy (it still self-destructs)")):
    """Run a command on a fresh GPU droplet, fetch results, destroy it.
    Example: icpc2 gpu run -- icpc2 sweep configs/sweeps/benchmark-01.yaml"""
    from icpc2.remote import do_gpu

    raise typer.Exit(do_gpu.run(ctx.args, size=size, region=region, keep=keep))


@gpu_app.command("sizes")
def gpu_sizes():
    from icpc2.remote import do_gpu

    for s in do_gpu.sizes():
        typer.echo(f"{s['slug']:28s} ${s['price_hourly']:>6.2f}/h  {','.join(s['regions']) or '-'}")


@gpu_app.command("list")
def gpu_list():
    from icpc2.remote import do_gpu

    for d in do_gpu.list_droplets():
        typer.echo(f"{d['id']}  {d['name']}  {d['size_slug']}  {d['status']}  {d['created_at']}")


@gpu_app.command("cleanup")
def gpu_cleanup():
    """Destroy every droplet tagged icpc2-train."""
    from icpc2.remote import do_gpu

    typer.echo(f"destroyed {do_gpu.cleanup()}")


@app.command("gold-import")
def gold_import(verdict_dir: Path, test_share: float = 0.8):
    """Turn exported review verdicts (JSON files, one per draft) into data/gold/gold.csv."""
    import pandas as pd

    from icpc2.text import hash_fraction

    drafts = {json.loads(x)["id"]: json.loads(x) for x in open(path("data/gold/drafts.jsonl"))}
    rows, rejected = [], []
    for f in sorted(Path(verdict_dir).rglob("*.json")):
        v = json.loads(f.read_text())
        v = v.get("data", v)
        d = drafts.get(f.stem)
        if not d:
            continue
        if v.get("status") not in ("ok", "fixed"):
            rejected.append({"draft_id": d["id"], "status": v.get("status"), "text": d["text"],
                             "proposed": d["code"], "note": v.get("note", "")})
            continue
        rows.append({"text": d["text"], "code": v["code"], "lang": d["lang"],
                     "split": "test" if hash_fraction(d["id"], "gold") < test_share else "dev",
                     "source": "claude_draft_" + v["status"], "notes": v.get("note", ""),
                     "draft_id": d["id"], "proposed": d["code"], "difficulty": d["difficulty"]})
    df = pd.DataFrame(rows)
    df.to_csv(path("data/gold/gold.csv"), index=False)
    pd.DataFrame(rejected).to_csv(path("data/gold/rejected.csv"), index=False)
    n_fixed = (df["source"] == "claude_draft_fixed").sum() if len(df) else 0
    typer.echo(f"{len(df)} verified gold rows ({n_fixed} with corrected code) -> data/gold/gold.csv")


@app.command("publish-dataset")
def publish_dataset():
    """Push the public training/eval dataset (no user text) to the HF Hub."""
    from icpc2.hub import push_public_dataset

    push_public_dataset()


@app.command("serve")
def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    import uvicorn

    uvicorn.run("icpc2.web.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
