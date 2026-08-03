import os

from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

for k in ("OPENAI_API_KEY", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise ValueError(f"Missing required environment variable: {k}")



PDF_PATH = os.getenv("PDF_PATH")
loader = PyPDFLoader(PDF_PATH)

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100, add_start_index=False)

chunks = splitter.split_documents(docs)

if not chunks:
    raise SystemExit("No chunks were created from the PDF. Please check the PDF file and try again.")


# enriched_document = []
# for d in chunks:
#     metadata = {k: v for k, v in d.metadata.items() if v not in ("", None)}
#     new_doc = Document(page_content=d.page_content, metadata=metadata)
#     enriched_document.append(new_doc)

enriched_document = [
    Document(
        page_content=d.page_content,
        metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
    )
    for d in chunks
]

ids = [f"doc-{i}" for i in range(len(enriched_document))]


# situação diferente por conta de usar o lm studio, openai ficou me bloqueando para pagar.
embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDINGS_MODEL", "text-embedding-3-small"), check_embedding_ctx_length=False)

store = PGVector(
    embeddings=embeddings,
    connection=os.getenv("PGVECTOR_URL"),
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    use_jsonb=True
)

store.add_documents(enriched_document, ids=ids)

def access_store():
    return store


if __name__ == "__main__":
    pass