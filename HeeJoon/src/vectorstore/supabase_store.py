import warnings
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from supabase import Client, create_client

from src.config import (
    SUPABASE_KEY,
    SUPABASE_QUERY_NAME,
    SUPABASE_TABLE_NAME,
    SUPABASE_URL,
)
from src.vectorstore.embeddings import get_embeddings_model


class PatchedSupabaseVectorStore(SupabaseVectorStore):
    """postgrest 2.x 호환 패치: .params.set() → 메서드 체이닝."""

    def similarity_search_by_vector_with_relevance_scores(
        self,
        query: List[float],
        k: int,
        filter: Optional[Dict[str, Any]] = None,
        postgrest_filter: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Tuple[Document, float]]:
        match_documents_params = self.match_args(query, filter)
        query_builder = self._client.rpc(self.query_name, match_documents_params)

        if postgrest_filter:
            query_builder = query_builder.filter("and", f"({postgrest_filter})")

        query_builder = query_builder.limit(k)

        res = query_builder.execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        if score_threshold is not None:
            match_result = [
                (doc, similarity)
                for doc, similarity in match_result
                if similarity >= score_threshold
            ]
            if len(match_result) == 0:
                warnings.warn(
                    "No relevant docs were retrieved using the relevance score"
                    f" threshold {score_threshold}"
                )

        return match_result


def get_supabase_client() -> Client:
    """Supabase 클라이언트를 초기화합니다."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_vector_store() -> PatchedSupabaseVectorStore:
    """PatchedSupabaseVectorStore 인스턴스를 반환합니다."""
    client = get_supabase_client()
    embeddings = get_embeddings_model()
    return PatchedSupabaseVectorStore(
        client=client,
        embedding=embeddings,
        table_name=SUPABASE_TABLE_NAME,
        query_name=SUPABASE_QUERY_NAME,
    )


def ingest_documents(documents: list[Document], batch_size: int = 100):
    """문서를 배치 단위로 Supabase documents 테이블에 업로드합니다."""
    client = get_supabase_client()
    embeddings = get_embeddings_model()

    total_batches = (len(documents) + batch_size - 1) // batch_size

    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  배치 {batch_num}/{total_batches} 업로드 중 ({len(batch)}건)...")

        if i == 0:
            vector_store = SupabaseVectorStore.from_documents(
                documents=batch,
                embedding=embeddings,
                client=client,
                table_name=SUPABASE_TABLE_NAME,
                query_name=SUPABASE_QUERY_NAME,
            )
        else:
            vector_store.add_documents(documents=batch)

    print(f"  벡터 임베딩 업로드 완료: {len(documents)}건")
    return vector_store


def upload_drugs_to_supabase(drug_rows: list[dict], batch_size: int = 500):
    """병합된 약품 데이터를 Supabase drugs 테이블에 upsert합니다."""
    client = get_supabase_client()

    total_batches = (len(drug_rows) + batch_size - 1) // batch_size

    for i in range(0, len(drug_rows), batch_size):
        batch = drug_rows[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  drugs 배치 {batch_num}/{total_batches} upsert 중 ({len(batch)}건)...")
        client.table("drugs").upsert(batch, on_conflict="item_seq").execute()

    print(f"  drugs 테이블 업로드 완료: {len(drug_rows)}건")


