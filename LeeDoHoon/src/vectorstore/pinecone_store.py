from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_DIMENSION,
    PINECONE_INDEX_NAME,
    PINECONE_METRIC,
    PINECONE_REGION,
)
from src.vectorstore.embeddings import get_embeddings_model


def get_pinecone_client() -> Pinecone:
    """Pinecone 클라이언트를 초기화합니다."""
    return Pinecone(api_key=PINECONE_API_KEY)


def create_index_if_not_exists():
    """Pinecone 인덱스가 없으면 생성합니다."""
    pc = get_pinecone_client()
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        print(f"인덱스 '{PINECONE_INDEX_NAME}' 생성 완료")
    else:
        print(f"인덱스 '{PINECONE_INDEX_NAME}' 이미 존재")


def get_vector_store() -> PineconeVectorStore:
    """기존 Pinecone 벡터스토어 인스턴스를 반환합니다."""
    embeddings = get_embeddings_model()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )


def ingest_documents(documents: list[Document], batch_size: int = 100):
    """문서를 배치 단위로 Pinecone에 업로드합니다."""
    embeddings = get_embeddings_model()
    create_index_if_not_exists()

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(documents) + batch_size - 1) // batch_size
        print(f"  배치 {batch_num}/{total_batches} 업로드 중 ({len(batch)}건)...")

        if i == 0:
            vector_store = PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME,
                pinecone_api_key=PINECONE_API_KEY,
            )
        else:
            vector_store = get_vector_store()
            vector_store.add_documents(documents=batch)

    return vector_store
