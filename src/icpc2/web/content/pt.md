---
# Textos da página em português (PT-PT).
# Edite o texto entre aspas; mantenha os nomes à esquerda dos dois pontos.
# Abaixo do segundo "---" está o texto das secções, em Markdown.
# Cada título "# ..." é uma secção que abre e fecha (ex.: "# Sobre o projeto", "# API").

title: "text-to-ICPC2"
description: "Sugere códigos ICPC-2 a partir de diagnósticos ou problemas escritos em texto livre, em português ou inglês."
subtitle: "Escreva um diagnóstico ou problema e obtenha o código ICPC-2."
placeholder: "ex.: HTA não controlada, lombalgia mecânica, DM2"
button: "Classificar"
notice: "Não escreva dados que identifiquem o doente (nomes, datas, moradas, números de utente). Os textos e o feedback são registados, sem endereço IP, para melhorar o modelo; as correções só são usadas depois de revistas."

# resultados
copy: "Copiar"
copied: "Copiado"
correct: "Correto"
wrong: "Errado"
fix_label: "Qual seria o código correto?"
fix_placeholder: "K86 ou hipertensão…"
save: "Guardar"
thanks: "Obrigado — o feedback foi registado."
model: "modelo {version}"
model_loading: "modelo a carregar…"
code_lookup: "pesquisa por código"

# mensagens de erro
error_unknown_code: "O código {code} não existe."
error_model_loading: "O modelo está a carregar. Tente dentro de momentos."
error_rate_limit: "Demasiados pedidos. Aguarde um minuto e tente de novo."

# rodapé e controlos
disclaimer: "Demo de investigação, não é um dispositivo médico."
theme: "Alternar modo claro/escuro"
---

# Sobre o projeto

O **text-to-ICPC2** sugere códigos ICPC-2 a partir de diagnósticos ou problemas escritos em texto livre, em português ou inglês, para apoiar a codificação nos cuidados de saúde primários. Cobre os 686 códigos de sintomas/queixas e diagnósticos da ICPC-2 (ACSS).

O modelo atual é um *retriever* baseado no **multilingual-e5-base**, treinado com as descrições oficiais da ICPC-2 e as correspondências com a ICD-10, dados sintéticos e o feedback de quem usa esta página. Num conjunto de 460 entradas verificadas por um médico, acerta no código em 1.º lugar em 92,2% dos casos e coloca-o entre os 5 primeiros em 98,5%.

- Modelo: [huggingface.co/diogocarapito/text-to-icpc2-v2](https://huggingface.co/diogocarapito/text-to-icpc2-v2)
- Dataset: [huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset](https://huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset)
- Código: [github.com/DiogoCarapito/text-to-icpc2-v2](https://github.com/DiogoCarapito/text-to-icpc2-v2)

Este projeto foi desenvolvido por Diogo Carapito com o apoio de uma bolsa de investigação da [AICIB](https://aicib.pt/) e da [APMGF](https://apmgf.pt/), no âmbito do internato médico de Medicina Geral e Familiar. É a segunda versão do [text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2).

# API

A API é pública e gratuita, sem registo. Os textos enviados são registados tal como os desta página, por isso **não envie dados identificativos de doentes**. Limite: 120 pedidos por minuto por endereço.

**Classificar um texto**: `POST /api/predict`

```bash
curl -s https://text-to-icpc2.com/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "HTA não controlada", "lang": "pt"}'
```

- `text`: o diagnóstico ou problema, 1 a 300 caracteres.
- `lang`: idioma do texto, `pt` (por omissão) ou `en`.

Resposta (abreviada):

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

`predictions` traz os 5 códigos mais prováveis, do melhor para o pior. O `score` (0 a 1) é uma pontuação de semelhança, não uma probabilidade. Se o texto for um código (ex.: `K86`), a resposta é esse código e `model_version` é `null`.

**Enviar feedback**: `POST /api/feedback` com `{"request_id": 1234, "feedback_key": "…", "kind": "correct", "code": "K86"}`, usando o `request_id` e a `feedback_key` devolvidos pelo `/api/predict`. `kind` pode ser `correct`, `wrong`, `corrected` (com o código certo) ou `copy`; um de cada por pedido. As correções para um código diferente do 1.º sugerido só alteram o modelo depois de revistas por um médico.

**Outros:** `GET /api/codes` (lista dos 686 códigos com títulos) · `GET /health` (versão do modelo).

Documentação interativa: [/docs](/docs).
