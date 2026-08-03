import os

from dotenv import load_dotenv

load_dotenv()


def _is_lm_studio_placeholder(key):
    return not key or key.strip().lower() == "lm-studio"


def detect_provider():
    openai_key = os.getenv("OPENAI_API_KEY", "")
    google_key = os.getenv("GOOGLE_API_KEY", "")

    if not _is_lm_studio_placeholder(openai_key):
        return "openai"

    if google_key.strip():
        return "google"

    return "local"


def build_chat_model():
    provider = detect_provider()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.5,
            max_tokens=1024,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.5,
            max_tokens=1024,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("LLM_LOCAL_MODEL", "local-model"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        temperature=0.5,
        max_tokens=1024,
    )


def build_embeddings():
    provider = detect_provider()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            check_embedding_ctx_length=False,
        )

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL", "models/embedding-001"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5"),
        base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1"),
        api_key="lm-studio",
        check_embedding_ctx_length=False,
    )
