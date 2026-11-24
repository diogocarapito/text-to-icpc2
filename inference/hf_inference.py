# python inference/hf_inference.py --text "Hipertensão arterial"

import click
import torch
import pandas as pd
from transformers import pipeline


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@click.command()
@click.option(
    "--text",
    default="Hipertensão arterial",
    help="Clinical text to classify",
    required=False,
)
@click.option(
    "--model",
    default="diogocarapito/text-to-icpc2-bert-base-portuguese-cased",
    help="HuggingFace model repo id",
    required=False,
)
@click.option(
    "--top_k",
    default=5,
    help="Number of top predictions to return",
    required=False,
)
def main(text, model, top_k):
    device = get_device()
    click.echo(f"Device: {device}")
    click.echo(f"Loading model: {model}")

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer="neuralmind/bert-base-portuguese-cased",
        device=device,
        top_k=top_k,
    )

    click.echo(f"\nInput: {text}\n")

    code_descriptions = pd.read_csv("data/code_text_label.csv").set_index("code")["text"]

    results = pipe(text)[0]

    click.echo(f"{'Rank':<6}{'Code':<10}{'Score':>8}  {'Description'}")
    click.echo("-" * 60)
    for i, result in enumerate(results, 1):
        code = result["label"]
        description = code_descriptions.get(code, "")
        click.echo(f"{i:<6}{code:<10}{result['score']:>8.4f}  {description}")


if __name__ == "__main__":
    main()
