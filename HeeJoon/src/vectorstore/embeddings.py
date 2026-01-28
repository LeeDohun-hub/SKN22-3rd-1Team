from langchain_openai import OpenAIEmbeddings

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY


def get_embeddings_model() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델을 초기화하여 반환합니다."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )
