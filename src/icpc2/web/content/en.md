---
# Page text in English.
# Edit the text in quotes; keep the names to the left of the colons.
# Everything below the second "---" is the "About the project" section, in Markdown.

title: "text-to-ICPC2"
description: "Suggests ICPC-2 codes from diagnoses or problems written as free text, in Portuguese or English."
subtitle: "Type a diagnosis or problem and get its ICPC-2 code."
placeholder: "e.g. uncontrolled hypertension, low back pain, T2DM"
button: "Classify"
notice: "Do not enter patient-identifying data. Submitted texts and feedback are logged to improve the model."

# results
copy: "Copy"
copied: "Copied"
correct: "Correct"
wrong: "Wrong"
fix_label: "What would be the correct code?"
fix_placeholder: "K86 or hypertension…"
save: "Save"
thanks: "Thanks — feedback recorded."
model: "model {version}"
model_loading: "model loading…"
code_lookup: "code lookup"

# error messages
error_unknown_code: "Code {code} does not exist."
error_model_loading: "The model is loading. Try again in a moment."

# footer and controls
disclaimer: "Research demo, not a medical device."
theme: "Toggle light/dark mode"
about_title: "About the project"
---

**text-to-ICPC2** suggests ICPC-2 codes from diagnoses or problems written as free text, in Portuguese or English, to support coding in primary care. It covers the 686 symptom/complaint and diagnosis codes of ICPC-2 (ACSS, Portugal).

The current model is a retriever based on **multilingual-e5-base**, trained on the official ICPC-2 descriptions and their ICD-10 mappings, synthetic data and feedback from people using this page. On 460 physician-verified entries, its first suggestion is correct 92.2% of the time, and the right code is in the top 5 for 98.5%.

- Model: [huggingface.co/diogocarapito/text-to-icpc2-v2](https://huggingface.co/diogocarapito/text-to-icpc2-v2)
- Dataset: [huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset](https://huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset)
- Code: [github.com/DiogoCarapito/text-to-icpc2-v2](https://github.com/DiogoCarapito/text-to-icpc2-v2)

Developed by Diogo Carapito with the support of a research grant from [AICIB](https://aicib.pt/) and [APMGF](https://apmgf.pt/) (the Portuguese Association of General and Family Medicine), as part of the General and Family Medicine residency programme. It is the second version of [text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2).
