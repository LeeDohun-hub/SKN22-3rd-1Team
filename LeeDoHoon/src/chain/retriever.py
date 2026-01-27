import json
import re
from typing import Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import SEARCH_K
from src.vectorstore.pinecone_store import get_vector_store

# 한국어 조사 패턴
_PARTICLES = re.compile(
    r"(의|은|는|이|가|을|를|에|에서|로|으로|와|과|도|만|부터|까지|에게|한테|께)$"
)


def _extract_keywords(query: str) -> list[str]:
    """쿼리에서 조사를 제거하고 핵심 키워드를 추출합니다."""
    query = re.sub(r"[?!.,]", "", query)
    words = query.split()
    keywords = []
    for word in words:
        stem = _PARTICLES.sub("", word)
        if stem and len(stem) >= 2:
            keywords.append(stem)
    return keywords


class DrugNameRetriever(BaseRetriever):
    """제품명 + 효능 키워드 매칭 기반 Retriever."""

    documents: list[Document]
    k: int = SEARCH_K

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> list[Document]:
        keywords = _extract_keywords(query)
        scored_docs = []

        for doc in self.documents:
            efcy_text = doc.page_content  # 효능 텍스트만 포함
            item_name = doc.metadata.get("item_name", "")
            score = 0

            for kw in keywords:
                # 제품명에 키워드 포함 시 높은 점수
                if kw in item_name:
                    score += 10
                # 효능 텍스트에 키워드 포함 시 낮은 점수
                elif kw in efcy_text:
                    score += 1

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[: self.k]]


def _load_documents(path: str = "data/raw/drugs_raw.json") -> list[Document]:
    """원본 데이터에서 Document 객체를 생성합니다."""
    from src.data.preprocessor import preprocess_all
    from src.data.loader import create_documents

    with open(path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    processed = preprocess_all(raw_items)
    return create_documents(processed)


def get_retriever() -> BaseRetriever:
    """
    제품명 키워드 매칭 + Pinecone 벡터 검색 Ensemble Retriever를 반환합니다.
    키워드 매칭은 약품명 검색에, 벡터 검색은 의미적 유사도에 활용됩니다.
    """
    from langchain_classic.retrievers import EnsembleRetriever

    # 벡터 검색 Retriever
    vector_store = get_vector_store()
    vector_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": SEARCH_K},
    )

    # 제품명 키워드 매칭 Retriever
    documents = _load_documents()
    keyword_retriever = DrugNameRetriever(documents=documents, k=SEARCH_K)

    # Ensemble: 키워드 가중치 0.7 + 벡터 가중치 0.3
    ensemble_retriever = EnsembleRetriever(
        retrievers=[keyword_retriever, vector_retriever],
        weights=[0.7, 0.3],
    )

    return ensemble_retriever
