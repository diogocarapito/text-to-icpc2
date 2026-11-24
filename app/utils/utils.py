import streamlit as st
from transformers import pipeline
import torch
import pandas as pd


def device_cuda_mps_cpu(force_cpu=False):
    if force_cpu:
        device = "cpu"
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return device


@st.cache_resource()
def load_model(model_name, available_device):
    # Load model and create a text-classification pipeline
    # model_name_or_path = "diogocarapito/text-to-icpc2" # loading the medium size model trained only with K (cardiovascular) codes

    # prepare the pipeline
    pipe = pipeline(
        "text-classification",
        model=model_name,
        tokenizer="neuralmind/bert-base-portuguese-cased",
        device=available_device,
    )
    return pipe


@st.cache_data()
def load_csv_github(github_raw_url):
    df = pd.read_csv(github_raw_url)
    return df


def prediction_display(prediction, labels_dataframe):
    for each in prediction:
        # label = each["label"]
        label = each["code"]

        description = labels_dataframe[labels_dataframe["cod"] == label]["nome"].values[
            0
        ]

        st.write(f"## {label} - {description}")

        include = labels_dataframe[labels_dataframe["cod"] == label]["incl"].values[0]

        st.write("### Inclui")
        st.write(include)

        exclude = labels_dataframe[labels_dataframe["cod"] == label]["excl"].values[0]

        st.write("### Exclui")
        st.write(exclude)

        criteria = labels_dataframe[labels_dataframe["cod"] == label]["crit"].values[0]

        st.write("### Critérios")
        st.write(criteria)

        st.write("### ICD-10")
        icd_10_code = (
            labels_dataframe[labels_dataframe["cod"] == label]["ICD_10_new"]
            .values[0]
            .split(",")
        )
        # remove all [, ] and ' from the list
        icd_10_code = [
            x.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
            for x in icd_10_code
        ]
        icd_10_description = (
            labels_dataframe[labels_dataframe["cod"] == label][
                "ICD_10_list_description"
            ]
            .values[0]
            .split("',")
        )
        icd_10_description = [
            x.replace("['", "").replace("']", "").replace(" '", "")
            for x in icd_10_description
        ]

        for each in zip(icd_10_code, icd_10_description):
            st.write(f"{each[0]} - *{each[1]}*")


def load_predictions_labels(runid="862e53bb1e7a4c05ab8a049c5a97a257"):
    # load the correct predictions
    # df_correct_current_model = load_csv_github("https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/correct_predictions/correct_predictions_862e53bb1e7a4c05ab8a049c5a97a257.csv")
    df_model_predictions = pd.read_csv(
        f"https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/correct_predictions/correct_predictions_{runid}.csv"
    )

    # Create a list of codes that are present in correct_prediction
    df_correct_current_model = df_model_predictions[["code", "top_prediction"]]

    df_correct_current_model = df_correct_current_model.rename(
        columns={"top_prediction": "is_correct"}
    )
    df_correct_current_model["is_correct"] = True

    return df_correct_current_model


def func():
    return None
