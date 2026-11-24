import streamlit as st
import pyperclip
import pandas as pd
import re
import timeit
import time
import os

from utils.utils import (
    device_cuda_mps_cpu,
    load_model,
    prediction_display,
    load_csv_github,
)


def send_feedback(text, result, type_feedback, value):
    print(f"Feedback sent: {text}, {result}, {type_feedback}, {value}")


def inference_result(results):
    for i, result in enumerate(results):
        code = result["code"]
        diagnosis = result["text"]
        value = result["score"]

        # get the top code
        if i == 0:
            top_code = code
            top_diagnosis = diagnosis
            top_value = value

        if value > 0.4:
            font_size = 30
        elif value > 0.2:
            font_size = 24
        else:
            font_size = 18

        with st.container(border=True):
            col0, col1, col2, col3, col4, col5 = st.columns([1, 3, 1, 1, 1, 1])
            with col0:
                st.markdown(
                    f"<span style='font-size:{font_size}px; font-weight:bold;'>{code}</span>",
                    unsafe_allow_html=True,
                )
            with col1:
                # st.write(f"**{label}**: {value:.3f}%")
                st.write(diagnosis)
            with col2:
                st.progress(value)
            with col3:
                st.write(f"{value:.2f}%")
            with col4:
                feedback = st.feedback("thumbs", key=f"feedback_{i}")
                if feedback == 1:
                    send_feedback(text, code, "thumbs", "up")
                elif feedback == 0:
                    send_feedback(text, code, "thumbs", "down")
                # elif feedback == None:

            with col5:
                if st.button("Copiar", key=f"select_{i}"):
                    pyperclip.copy(code)
                    send_feedback(text, code, "copy", "copy")

    return top_code, top_diagnosis, top_value


st.title("text-to-icpc2")

text = st.text_input("Coloque aqui o texto que deseja converter para ICPC2")

# get availabel device
availabel_device = device_cuda_mps_cpu(force_cpu=True)

# choose model
# model_chosen = "text-to-icpc2_bert-base-uncased"
model_chosen = "text-to-icpc2"

pipe = load_model(f"diogocarapito/{model_chosen}", availabel_device)

if text:
    labels_dataframe = pd.read_csv(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/main/data/icpc2_processed.csv"
    )

    label_code_dict = pd.read_csv(
        "https://raw.githubusercontent.com/DiogoCarapito/text-to-icpc2/refs/heads/main/data/code_text_label.csv"
    )

    # check if text corresponds to a letter and 2 algarisms with re
    if bool(re.match(r"[A-Za-z]\d\d", text)):
        # make the text uppercase
        text = text.upper()

        try:
            # get the code
            description = labels_dataframe.loc[
                labels_dataframe["cod"] == text, "nome"
            ].values[0]

            st.write(f"## {text} - {description}")
        except IndexError:
            st.warning("Código não encontrado")

    else:
        # Record the start time
        start_time = timeit.default_timer()

        # Execute prediction
        predictions = pipe(text, top_k=5)

        # Record the end time
        end_time = timeit.default_timer()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        for each in predictions:
            # transform the label into a icpc2 code
            if each["label"].startswith("LABEL_"):
                # split on the "_" and get the 2nd part and make it integer
                each["label"] = int(each["label"].split("_")[1])

                # get the code
                each["code"] = label_code_dict.loc[
                    label_code_dict["label"] == each["label"], "code"
                ].values[0]

                # get the desctição
                each["text"] = label_code_dict.loc[
                    label_code_dict["label"] == each["label"], "text"
                ].values[0]

        # Display the elapsed time
        st.write(
            f"Tempo necessário para classificação: **{elapsed_time:.4f} segundos** com **'{availabel_device}'**"
        )

        code, diagnosis, value = inference_result(predictions)

        correct_code = st.text_input(
            "Se não acertou, qual é o código correto?", key="correct_code"
        )

        if correct_code and code:
            send_feedback(text, code, "correct_code", correct_code)
