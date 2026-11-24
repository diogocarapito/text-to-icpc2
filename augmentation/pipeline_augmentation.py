# pylint: disable=W1203

# python augmentation/pipeline_augmentation.py --threshold 15 --target 15
# Generates augmented text for ICPC-2 codes with fewer than --threshold samples.
# Results are saved to data/data_augmentation.csv with include=False for manual review.
# After review in the Streamlit app, run etl/etl.py to rebuild the training dataset.

import os
import re
import time
import logging

import click
import dotenv
import pandas as pd
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

CHAPTER_NAMES = {
    "NE": "não especificada",
    "(B)": "por Sangue, órgãos hematopoiéticos e linfáticos",
    "(D)": "por Aparelho digestivo e Gastrenterologia",
    "(F)": "por Olhos e Oftalmologia",
    "(H)": "por Ouvidos e Otorrinolaringologia",
    "(K)": "por sistema cardiovascular e aparelho circulatório",
    "(L)": "por Sistema musculo-esquelético",
    "(N)": "por Sistema nervoso e Neurologia",
    "(P)": "por Piscologico e Psiquiátrico",
    "(R)": "por Aparelho respiratório e Penumologia",
    "(S)": "por Pele e Dermatologia",
    "(T)": "por Endócrino, metabólico e nutricional",
    "(U)": "por Aparelho urinário",
    "(W)": "por Gravidez e planeamento familiar",
    "(X)": "por Aparelho genital feminino (incluíndo mama)",
    "(Y)": "por Aparelho genital masculino",
    "(Z)": "por Problemas sociais",
}


def process_icpc2_description_text(text):
    if str(text) == "nan":
        return ""
    cleaned = re.sub(r"\b[A-Z]\d{2}\b", "", str(text))
    cleaned = re.sub(r" ; se o doente tem a doença, codifique-a", "", cleaned)
    cleaned = re.sub(r" ; ", "; ", cleaned)
    return cleaned.strip()


def chapters_substitution(text):
    for key, value in CHAPTER_NAMES.items():
        text = text.replace(key, value)
    return text


def prompt_design(code, icpc2_df):
    row = icpc2_df[icpc2_df["cod"] == code]
    if row.empty:
        return None, None

    labels = "; ".join(row["nome"].values)
    labels = chapters_substitution(labels)

    inclui = process_icpc2_description_text(row["incl"].values[0])
    exclui = process_icpc2_description_text(row["excl"].values[0])
    icd10 = process_icpc2_description_text(row["ICD_10_list_description_join"].values[0])
    criterios = process_icpc2_description_text(row["crit"].values[0])

    parts = [labels]
    if inclui:
        parts.append(inclui)
    if icd10:
        parts.append(icd10)
    if criterios:
        parts.append(f"Critérios: {criterios}")
    if exclui:
        parts.append(f"Exclui: {exclui}")

    prompt_text = "; ".join(parts)

    context_text = (
        "Estou a fazer um processo de Data Augmentation e preciso de encontrar sinónimos "
        "para a seguinte doença ou problema de saúde em português de Portugal. "
        "Vou te dar as expressões que já tenho, encontra-me variações que sejam sinónimos "
        "ou outras formas de expressão similares. Dá-me entre 10 e 50 resultados. "
        "Devolve apenas a lista de expressões separadas por ';'"
    )

    return prompt_text, context_text


def call_openai(client, prompt_text, context_text, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": context_text},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
    )
    return response.choices[0].message.content


def parse_response(raw_text):
    raw_text = raw_text.strip().rstrip(".")
    items = [item.strip() for item in raw_text.split(";") if item.strip()]
    seen = set()
    unique_items = []
    for item in items:
        if item.lower() not in seen:
            seen.add(item.lower())
            unique_items.append(item)
    return unique_items


def append_to_augmentation_csv(rows, path="data/data_augmentation.csv"):
    existing = pd.read_csv(path)
    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["code", "text"])
    combined.to_csv(path, index=False)
    return len(combined) - len(existing)


@click.command()
@click.option(
    "--threshold",
    default=15,
    show_default=True,
    help="Augment codes with fewer than this many samples.",
)
@click.option(
    "--target",
    default=15,
    show_default=True,
    help="Target number of samples per code (used for logging only; API returns variable counts).",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    show_default=True,
    help="OpenAI model to use.",
)
@click.option(
    "--delay",
    default=1.0,
    show_default=True,
    help="Seconds to wait between API calls to avoid rate limits.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print underrepresented codes and their counts without calling the API.",
)
@click.option(
    "--auto-include",
    is_flag=True,
    default=False,
    help="Save rows with include=True (skip manual review). Default is False — review in Streamlit app first.",
)
def main(threshold=15, target=15, model="gpt-4o-mini", delay=1.0, dry_run=False, auto_include=False):
    logging.info("Loading data_pre_train.csv")
    df_train = pd.read_csv("data/data_pre_train.csv")
    counts = df_train.groupby("code").size()

    underrepresented = counts[counts < threshold].sort_values()
    logging.info(
        f"{len(underrepresented)} codes have fewer than {threshold} samples (target: {target})"
    )

    if dry_run:
        print(f"\n{'Code':<8} {'Current samples':>15}")
        print("-" * 25)
        for code, count in underrepresented.items():
            print(f"{code:<8} {count:>15}")
        print(f"\nTotal codes to augment: {len(underrepresented)}")
        return

    dotenv.load_dotenv()
    logging.info("Loading icpc2_processed.csv")
    icpc2_df = pd.read_csv("data/icpc2_processed.csv")

    client = OpenAI(
        project=os.getenv("OPENAI_PROJECT_ID"),
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    total_added = 0

    for i, (code, current_count) in enumerate(underrepresented.items(), 1):
        logging.info(
            f"[{i}/{len(underrepresented)}] {code}: {current_count} samples, target {target}"
        )

        prompt_text, context_text = prompt_design(code, icpc2_df)
        if prompt_text is None:
            logging.warning(f"  No ICPC-2 data found for {code}, skipping")
            continue

        try:
            raw = call_openai(client, prompt_text, context_text, model=model)
            items = parse_response(raw)
            logging.info(f"  Got {len(items)} variations from API")

            rows = [
                {
                    "code": code,
                    "text": text,
                    "origin": model,
                    "include": auto_include,
                    "context": context_text,
                    "prompt": prompt_text,
                }
                for text in items
            ]

            added = append_to_augmentation_csv(rows)
            total_added += added
            logging.info(f"  Saved {added} new rows to data/data_augmentation.csv")

        except Exception as e:
            logging.error(f"  API call failed for {code}: {e}")

        if i < len(underrepresented):
            time.sleep(delay)

    logging.info(f"Done. Total new rows added: {total_added}")
    if not auto_include:
        logging.info(
            "Rows saved with include=False. Review in the Streamlit app, "
            "then run etl/etl.py to rebuild the dataset."
        )
    else:
        logging.info("Rows saved with include=True. Run etl/etl.py to rebuild the dataset.")


if __name__ == "__main__":
    main()
