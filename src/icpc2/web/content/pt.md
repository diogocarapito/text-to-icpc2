---
# Textos da página em português (PT-PT).
# Edite o texto entre aspas; mantenha os nomes à esquerda dos dois pontos.
# Tudo o que está abaixo do segundo "---" é a secção "Sobre o projeto", em Markdown.

title: "text-to-ICPC2"
description: "Sugere códigos ICPC-2 a partir de diagnósticos ou problemas escritos em texto livre, em português ou inglês."
subtitle: "Escreva um diagnóstico ou problema e obtenha o código ICPC-2."
placeholder: "ex.: HTA não controlada, lombalgia mecânica, DM2"
button: "Classificar"
notice: "Não introduza dados identificativos de doentes. Os textos submetidos e o seu feedback são registados para melhorar o modelo."

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

# rodapé e controlos
disclaimer: "Demo de investigação, não é um dispositivo médico."
theme: "Alternar modo claro/escuro"
about_title: "Sobre o projeto"
---

O **text-to-ICPC2** sugere códigos ICPC-2 a partir de diagnósticos ou problemas escritos em texto livre, em português ou inglês, para apoiar a codificação nos cuidados de saúde primários. Cobre os 686 códigos de sintomas/queixas e diagnósticos da ICPC-2 (ACSS).

O modelo atual é um *retriever* baseado no **multilingual-e5-base**, treinado com as descrições oficiais da ICPC-2 e as correspondências com a ICD-10, dados sintéticos e o feedback de quem usa esta página. Num conjunto de 460 entradas verificadas por um médico, acerta no código em 1.º lugar em 92,2% dos casos e coloca-o entre os 5 primeiros em 98,5%.

- Modelo: [huggingface.co/diogocarapito/text-to-icpc2-v2](https://huggingface.co/diogocarapito/text-to-icpc2-v2)
- Dataset: [huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset](https://huggingface.co/datasets/diogocarapito/text-to-icpc2-v2-dataset)
- Código: [github.com/DiogoCarapito/text-to-icpc2-v2](https://github.com/DiogoCarapito/text-to-icpc2-v2)

Este projeto foi desenvolvido por Diogo Carapito com o apoio de uma bolsa de investigação da [AICIB](https://aicib.pt/) e da [APMGF](https://apmgf.pt/), no âmbito do internato médico de Medicina Geral e Familiar. É a segunda versão do [text-to-icpc2](https://github.com/DiogoCarapito/text-to-icpc2).
