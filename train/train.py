# python train/train.py --size full --model neuralmind/bert-base-portuguese-cased --dev cuda --hf True

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    # pipeline,
)
import logging
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

# from typing import List, Tuple
import os
import click
import wandb
from dotenv import load_dotenv
from huggingface_hub import login

# from validation import validation


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(outputs.logits.device)
        )
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def experiment_size(size, model_name):
    # remove the path from the model name
    model_name = model_name.split("/")[-1]
    if size == "full":
        return f"text-to-icpc2-{model_name}"
    else:
        return f"text-to-icpc2-{size}-{model_name}"


@click.command()
@click.option(
    "--size",
    default="small",
    help="size of the dataset to be used",
    required=False,
)
@click.option(
    "--model",
    default="neuralmind/bert-base-portuguese-cased",
    help="pretraines model name",
    required=False,
)
@click.option(
    "--dev",
    default="cuda",
    required=True,
    help="device to be used",
)
@click.option(
    "--hf", default=True, help="publish to huggingface model", required=False
)
# @click.option("--val", default=False, help="perform validation", required=False)
def main(
    size="small", model="neuralmind/bert-base-portuguese-cased", dev="cuda", hf=True
):
    # distilbert/distilbert-base-uncased
    # distilbert/distilbert-base-multilingual-cased
    # google-bert/bert-base-uncased
    # google-bert/bert-base-cased
    # google-bert/bert-base-multilingual-uncased
    # google-bert/bert-base-multilingual-cased
    # google-bert/bert-large-uncased
    # google-bert/bert-large-cased

    load_dotenv()

    experiment_name = experiment_size(size, model_name=model)
    model_name = model

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Initialize W&B
    logging.info("Initializing W&B")
    # wandb.init(project="text-to-icpc2", name=experiment_name)
    run = wandb.init(project="text-to-icpc2")

    # seting up the device cuda, mps or cpu
    device = torch.device(device=dev)
    logging.info("Using the device '%s'", device)

    # Load the dataset
    logging.info("Loading dataset")
    if size == "small":
        dataset = load_dataset("diogocarapito/text-to-icpc2-nano")
    elif size == "medium" or size == "full":
        dataset = load_dataset("diogocarapito/text-to-icpc2")

    # get the distribution of the labels
    logging.info("Getting the distribution of the labels")
    features = dataset["train"].features

    number_of_labels = len(features["label"].names)

    logging.info(
        "Getting the distribution of the labels as a dictionary id : label and label : id"
    )
    # get the distribution of the labels as a dictionary id : label and  label : id
    id2label = {idx: features["label"].int2str(idx) for idx in range(number_of_labels)}
    lable2id = {v: k for k, v in id2label.items()}

    # model name
    # model_name = "bert-base-uncased"
    # model_name = "distilbert-base-uncased"
    logging.info("Using the model '%s'", model_name)

    # Load the tokenizer
    logging.info("Loading the tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a function to tokenize the data
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

    # Tokenize the data
    logging.info("Tokenize the data")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define the filter function
    logging.info("Applying the filter function for smaller size dataset")

    def filter_chapter(example):
        if size == "medium":
            return example["chapter"] == "K"
        else:
            return example

    # Select a small subset of the data
    tokenized_dataset = tokenized_dataset["train"].filter(filter_chapter)

    tokenize_dataset_validation = tokenized_dataset

    # Split the dataset into training and evaluation
    logging.info("Splitting the dataset into training and evaluation")
    test_size = 0.2
    stratify_by_column = "label"
    seed = 42

    tokenized_dataset_split = tokenized_dataset.train_test_split(
        test_size=test_size,
        stratify_by_column=stratify_by_column,
        seed=seed,
    )

    # Get the training and evaluation datasets separately
    tokenized_dataset_split_train = tokenized_dataset_split["train"]
    tokenized_dataset_split_test = tokenized_dataset_split["test"]

    logging.info(
        "The size of the training dataset is %s", len(tokenized_dataset_split_train)
    )
    logging.info(
        "The size of the evaluation dataset is %s", len(tokenized_dataset_split_test)
    )

    # Compute class weights from training labels to up-weight underrepresented codes
    logging.info("Computing class weights from training label distribution")
    train_labels = np.array(tokenized_dataset_split_train["label"])
    present_classes = np.unique(train_labels)
    weights_for_present = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=train_labels,
    )
    # Default weight 1.0 for any class absent from the training split
    class_weights = np.ones(number_of_labels)
    class_weights[present_classes] = weights_for_present
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    logging.info(
        "Class weights — min: %.4f, max: %.4f, mean: %.4f",
        class_weights.min(),
        class_weights.max(),
        class_weights.mean(),
    )

    # Load the model
    logging.info("Loading the model")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=number_of_labels,
        id2label=id2label,
        label2id=lable2id,
        ignore_mismatched_sizes=True,  # warning supression: Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight'])
    )

    # Define a function for calculating our defined target optimization metric during training
    logging.info("Defining the compute_metrics function")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        # top-1 predictions (argmax)
        top1_preds = np.argmax(logits, axis=-1)

        # top-k indices sorted by descending score
        topk_indices = np.argsort(logits, axis=-1)[:, ::-1]

        top1_acc = (top1_preds == labels).mean()
        top3_acc = np.any(topk_indices[:, :3] == labels[:, None], axis=1).mean()
        top5_acc = np.any(topk_indices[:, :5] == labels[:, None], axis=1).mean()

        macro_f1 = f1_score(labels, top1_preds, average="macro", zero_division=0)

        return {
            "accuracy": top1_acc,
            "top3_accuracy": top3_acc,
            "top5_accuracy": top5_acc,
            "macro_f1": macro_f1,
        }

    logging.info("Setting up the training output directory")
    # if t == "full" and hf:
    #     training_output_dir = "/tmp/text-to-icpc2"
    # else:
    training_output_dir = f"/tmp/{experiment_name}"

    if size == "small":
        n_epochs = 3
    else:
        n_epochs = 12

    # Checkpoints will be output to this `training_output_dir`.
    logging.info("Defining the training arguments")
    training_args = TrainingArguments(
        output_dir=training_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        run_name=experiment_name,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_steps=200,
        seed=42,
        num_train_epochs=n_epochs,
        # torch_compile=True,
        logging_dir="./logs",
        report_to="wandb",  # Report to W&B
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        weight_decay=0.01,
    )

    # Instantiate a WeightedTrainer to apply class-weighted cross-entropy loss
    logging.info("Instantiating the WeightedTrainer")
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset_split_train,
        eval_dataset=tokenized_dataset_split_test,
        compute_metrics=compute_metrics,
    )

    # Train the model
    logging.info("Training the model")
    trainer.train()

    # Evaluate the model by using on the full tokenized_dataset
    logging.info(
        "Evaluating the model with the full dataset: %s",
        len(tokenize_dataset_validation),
    )
    eval_results = trainer.evaluate(tokenize_dataset_validation)
    wandb.log(eval_results)

    # logging.info("F1: %s", eval_results["eval_f1"])
    logging.info("Accuracy: %s", eval_results["eval_accuracy"])
    # logging.info("Precision: %s", eval_results["eval_precision"])
    # logging.info("Recall: %s", eval_results["eval_recall"])

    # push the model to huggingface
    if hf:
        hf_token = os.getenv("huggingface_token")
        login(token=hf_token)
        trainer.push_to_hub()

    # Save the model
    model_dir = "/tmp/saved_model"
    trainer.save_model(model_dir)

    # Show the model saved in model_dir
    logging.info("Model saved in %s", model_dir)

    # Log the model as an artifact to W&B
    logging.info("Logging the model as an artifact to W&B")
    artifact = wandb.Artifact(name=experiment_name, type="model")
    artifact.add_file(f"{model_dir}/model.safetensors")
    run.log_artifact(artifact)

    # Save the model as a PyTorch model
    # logging.info("Saving the model as PyTorch")
    # pt_model_path = f"{model_dir}/model.pth"
    # torch.save(trainer.model.state_dict(), pt_model_path)

    # Log the PyTorch model using W&B
    # logging.info("Logging the PyTorch model to W&B")
    # pt_artifact = wandb.Artifact(name=f"{experiment_name}-pytorch", type="model")
    # pt_artifact.add_file(pt_model_path)
    # run.log_artifact(pt_artifact)

    # Link the artifact to the model registry
    # run.link_artifact(
    #     artifact=pt_artifact,
    #     target_path=f"mgf_nlp/{experiment_name}/text-to-icpc2:latest",
    #     aliases=[model_name, size],
    # )
    # logging.info("Model logged to W&B model registry")

    # # Define a class for the inference model
    # class ModelInference:
    #     def __init__(self, model_dir):
    #         # Load the tokenizer and model
    #         self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #         self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    #         self.model.eval()  # Set the model to evaluation mode

    #     def predict(self, text, top_k=5):
    #         # Tokenize the input text
    #         inputs = self.tokenizer(text, return_tensors="pt")

    #         # Perform inference
    #         with torch.no_grad():
    #             outputs = self.model(**inputs)
    #             probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    #             topk_values, topk_indices = torch.topk(probabilities, k=top_k, dim=-1)

    #         # Convert indices to labels
    #         topk_labels = [
    #             self.model.config.id2label[idx.item()] for idx in topk_indices[0]
    #         ]

    #         return topk_values[0], topk_labels

    # # Save the model to a directory as ONNX
    # logging.info("Saving the model as ONNX")

    # onnx_model_path = f"{model_dir}/model.onnx"
    # dummy_model_input = tokenizer(
    #     "Hipertensão Arterial",
    #     return_tensors="pt").to(device)

    # torch.onnx.export(
    #     model,
    #     tuple(dummy_model_input.values()),
    #     f=onnx_model_path,
    #     input_names=["input_ids", "attention_mask"],
    #     output_names=["logits"],
    #     dynamic_axes={
    #         "input_ids": {0: "batch_size", 1: "sequence_length"},
    #         "attention_mask": {0: "batch_size", 1: "sequence_length"},
    #         "logits": {0: "batch_size"},
    #     },
    #     opset_version=11,
    # )

    # # Log the ONNX model using W&B
    # logging.info("Logging the model to W&B")
    # artifact = wandb.Artifact(name=experiment_name, type="model")
    # artifact.add_file(onnx_model_path)
    # run.log_artifact(artifact)

    # # Link the artifact to the model registry
    # run.link_artifact(
    #     artifact=artifact,
    #     target_path=f"mgf_nlp/{experiment_name}/text-to-icpc2:latest",
    #     aliases=[model_name, size],
    # )

    run.finish()
    logging.info("Training Finished Successfully!!")


if __name__ == "__main__":
    main()
