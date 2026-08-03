import os

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from ingest import access_store

model = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash"),
    temperature=0.5,
    max_output_tokens=1024,
)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)


def search_prompt(question):
    store = access_store()
    docs = store.similarity_search(question, k=4)

    contexto = "\n\n".join(d.page_content for d in docs)

    formatted = prompt.format(contexto=contexto, pergunta=question)
    answer = model.invoke(formatted)

    return answer.content
