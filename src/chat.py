from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from search import search_prompt
from dotenv import load_dotenv
from ingest import ingest_pdf
import os

load_dotenv()

ingest_pdf()

model_openai = ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "bzl/gpt-5.4-nano"), temperature=0.5)

message_openai = model_openai.invoke("hello world openai")
print(message_openai.content)


model = init_chat_model(
    model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash"),
    model_provider="google_genai",
    temperature=0.5,
)

answer_gemini = model.invoke("hello world gemini")
print(answer_gemini.content)

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    pass

if __name__ == "__main__":
    main()