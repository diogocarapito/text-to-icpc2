---
# Page text in English.
# Edit the text in quotes; keep the names to the left of the colons.
# Below the second "---" is the text of the sections, in Markdown.
# Each "# ..." heading is a section that opens and closes (e.g. "# About the project", "# API").

title: "text-to-ICPC2"
description: "Suggests ICPC-2 codes from diagnoses or problems written as free text, in Portuguese or English."
subtitle: "Type a diagnosis or problem and get its ICPC-2 code."
placeholder: "e.g. uncontrolled hypertension, low back pain, T2DM"
button: "Classify"
notice: "Do not type anything that identifies the patient (names, dates, addresses, record numbers). Texts and feedback are logged, without IP addresses, to improve the model; corrections are used only after review."

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
error_rate_limit: "Too many requests. Wait a minute and try again."

# footer and controls
disclaimer: "Research demo, not a medical device."
theme: "Toggle light/dark mode"
---

# About the project

**text-to-ICPC2** suggests ICPC-2 codes from diagnoses or problems written as free text, in Portuguese or English, to support coding in primary care. It covers the 686 symptom/complaint and diagnosis codes of ICPC-2 (ACSS, Portugal).

The current model is a retriever based on **multilingual-e5-base**, trained on the official ICPC-2 descriptions and their ICD-10 mappings, synthetic data and feedback from people using this page. On 460 physician-verified entries, its first suggestion is correct 92.2% of the time, and the right code is in the top 5 for 98.5%.

- Model: [huggingface.co/diogocarapito/text-to-icpc2](https://huggingface.co/diogocarapito/text-to-icpc2)
- Dataset: [huggingface.co/datasets/diogocarapito/text-to-icpc2](https://huggingface.co/datasets/diogocarapito/text-to-icpc2)
- Code: [github.com/DiogoCarapito/text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2)

Developed by Diogo Carapito with the support of a research grant from [AICIB](https://aicib.pt/) and [APMGF](https://apmgf.pt/) (the Portuguese Association of General and Family Medicine), as part of the General and Family Medicine residency programme. It is the second version of [text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2).

# API

The API is public and free, with no sign-up. Submitted texts are logged just like the ones typed on this page, so **do not send patient-identifying data**. Limit: 120 requests per minute per address.

**Classify a text**: `POST /api/predict`

```bash
curl -s https://text-to-icpc2.com/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "uncontrolled hypertension", "lang": "en"}'
```

- `text`: the diagnosis or problem, 1 to 300 characters.
- `lang`: language of the text, `pt` (default) or `en`.

Response (shortened):

```json
{
  "request_id": 1234,
  "feedback_key": "9f2c…",
  "model_version": "v1",
  "latency_ms": 31.2,
  "predictions": [
    {"code": "K86", "score": 0.97,
     "title_pt": "Hipertensão sem complicações",
     "title_en": "Hypertension uncomplicated",
     "inclusion": ["hipertensão essencial", "…"]}
  ]
}
```

`predictions` holds the 5 most likely codes, best first. `score` (0 to 1) is a similarity score, not a probability. If the text is a code (e.g. `K86`), the response is that code and `model_version` is `null`.

**Send feedback**: `POST /api/feedback` with `{"request_id": 1234, "feedback_key": "…", "kind": "correct", "code": "K86"}`, using the `request_id` and `feedback_key` returned by `/api/predict`. `kind` is one of `correct`, `wrong`, `corrected` (with the right code) or `copy`; one of each per request. Corrections to a code other than the 1st suggestion change the model only after a physician reviews them.

**Other:** `GET /api/codes` (the 686 codes with titles) · `GET /health` (model version).

Interactive documentation: [/docs](/docs).
