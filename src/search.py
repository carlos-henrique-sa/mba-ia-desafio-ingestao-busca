import os

from langchain.chat_models import init_chat_model
from ingest import access_store
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI

model_openai = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "quantfactory/gpt2-xl"), disable_streaming=True)

message_openai = model_openai.invoke("hello world openai")
print(message_openai.content)

# model = init_chat_model(
#     model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash"),
#     model_provider="google_genai",
#     temperature=0.5,
# )

# answer_gemini = model.invoke("hello world gemini")
# print(answer_gemini.content)

store = access_store()

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
prompt = PromptTemplate.from_template(PROMPT_TEMPLATE, input_variables=["contexto", "pergunta"])

agent_chain = create_react_agent(llm=model_openai, tools=[], prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain, 
    tools=[], 
    handling_parsing_errors=True,
    verbose=True,
    max_iterations=3
)

def search_prompt(question=None):
    return prompt