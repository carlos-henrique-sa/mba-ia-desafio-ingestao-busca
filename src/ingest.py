import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from dotenv import load_dotenv
from providers import build_embeddings, detect_provider

load_dotenv()

for k in ("PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise ValueError(f"Missing required environment variable: {k}")

_store = None

def access_store():
    global _store
    if _store is None:
        embeddings = build_embeddings()
        _store = PGVector(
            embeddings=embeddings,
            connection=os.getenv("PGVECTOR_URL"),
            collection_name=os.getenv("PGVECTOR_COLLECTION"),
            use_jsonb=True,
        )
    return _store


def ingest_pdf():
    pdf_path = os.getenv("PDF_PATH")
    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, add_start_index=False
    )

    chunks = splitter.split_documents(docs)

    if not chunks:
        raise SystemExit(
            "No chunks were created from the PDF. Please check the PDF file and try again."
        )

    enriched_document = [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)},
        )
        for d in chunks
    ]

    ids = [f"doc-{i}" for i in range(len(enriched_document))]

    store = access_store()
    store.add_documents(enriched_document, ids=ids)

    return len(enriched_document)


if __name__ == "__main__":
    count = ingest_pdf()
    print(f"{count} chunks ingeridos no PGVector.")
