import os

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from ingest import access_store


model = ChatOpenAI(
    model=os.getenv("LLM_LOCAL_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "lm-studio"),
    temperature=0.5,
    max_tokens=1024,
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
    results = store.similarity_search_with_score(query=question, k=10)

    contexto = "\n\n".join(doc.page_content for doc, score in results)

    formatted = prompt.format(contexto=contexto, pergunta=question)
    answer = model.invoke(formatted)

    return answer.content
