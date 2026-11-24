# pylint: disable=W1203

import click
import logging
import datetime
import pandas as pd
import os
import torch
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import f1_score, precision_score, recall_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def inference(
    text_input=None,
    k=5,
    model_id="diogocarapito/text-to-icpc2-bert-base-portuguese-cased",
):
    logging.info("Starting inference")

    device = get_device()
    logging.info(f"Using device: {device}")

    if text_input is None:
        text_input = ["Hipertensão arterial", "Diabetes Mellitus"]

    if isinstance(text_input, pd.Series):
        list_inputs = list(text_input)
    elif not isinstance(text_input, list):
        list_inputs = [text_input]
    else:
        list_inputs = list(text_input)

    logging.info(f"Loading model: {model_id}")
    pipe = pipeline(
        "text-classification",
        model=model_id,
        tokenizer="neuralmind/bert-base-portuguese-cased",
        device=device,
        top_k=k,
    )

    code_descriptions = pd.read_csv("data/code_text_label.csv").set_index("code")["text"]

    results = []
    batch_size = 500
    for i in range(0, len(list_inputs), batch_size):
        batch = list_inputs[i : i + batch_size]
        logging.info(f"Performing inference on batch {i // batch_size + 1}")
        batch_preds = pipe(batch)
        for text, preds in zip(batch, batch_preds):
            row = {"input": text}
            for j, pred in enumerate(preds, 1):
                code = pred["label"]
                row[f"code_{j}"] = code
                row[f"value_{j}"] = pred["score"]
                row[f"text_{j}"] = code_descriptions.get(code, None)
            results.append(row)

    logging.info("Inference finished successfully!")
    return pd.DataFrame(results)


def validation(
    model_id="diogocarapito/text-to-icpc2-bert-base-portuguese-cased",
    save=True,
):
    logging.info("Loading dataset")
    dataset = load_dataset("diogocarapito/text-to-icpc2")
    df_dataset = pd.DataFrame(dataset["train"])

    logging.info("Performing validation")
    results = inference(df_dataset["text"], model_id=model_id)

    final_results = pd.merge(
        df_dataset,
        results,
        left_on="text",
        right_on="input",
        how="left",
    )
    final_results.drop(columns=["input"], inplace=True)

    print(final_results)

    logging.info("Calculating metrics")

    y_true = final_results["code"]
    y_pred = final_results["code_1"]

    top1_acc = (y_true == y_pred).mean()

    top3_acc = final_results.apply(
        lambda row: row["code"] in [row["code_1"], row["code_2"], row["code_3"]],
        axis=1,
    ).mean()

    top5_acc = final_results.apply(
        lambda row: row["code"]
        in [row["code_1"], row["code_2"], row["code_3"], row["code_4"], row["code_5"]],
        axis=1,
    ).mean()

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)

    per_chapter_acc = (
        final_results.groupby("chapter")
        .apply(lambda g: (g["code"] == g["code_1"]).mean())
        .rename("top1_accuracy")
        .reset_index()
    )

    final_results["correct_top1"] = final_results["code"] == final_results["code_1"]
    per_code_acc = (
        final_results.groupby("code")
        .agg(
            count=("correct_top1", "count"),
            top1_accuracy=("correct_top1", "mean"),
        )
        .reset_index()
    )

    metrics = {
        "model_id": model_id,
        "top1_accuracy": round(top1_acc, 4),
        "top3_accuracy": round(top3_acc, 4),
        "top5_accuracy": round(top5_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
    }

    logging.info(metrics)

    if save:
        date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_name = model_id.split("/")[-1]
        folder_name = f"validation/validation_runs/{date_time}_{model_name}"
        os.makedirs(folder_name)

        logging.info(f"Saving results to {folder_name}")

        pd.DataFrame([metrics]).to_csv(folder_name + "/metrics.csv", index=False)
        final_results.to_csv(folder_name + "/results.csv", index=False)
        per_chapter_acc.to_csv(folder_name + "/per_chapter_accuracy.csv", index=False)
        per_code_acc.to_csv(folder_name + "/per_code_accuracy.csv", index=False)

        logging.info(f"Validation results saved to {folder_name}")


@click.command()
@click.option(
    "--model_id",
    type=str,
    required=False,
    default="diogocarapito/text-to-icpc2-bert-base-portuguese-cased",
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Save the validation results and metrics (default: True).",
)
def main(model_id, save):
    validation(model_id=model_id, save=save)


if __name__ == "__main__":
    main()
