"""전체 데이터 적재 파이프라인: API -> 전처리 -> 임베딩 -> Pinecone"""

from src.data.collector import fetch_all_drugs
from src.data.loader import create_documents, split_documents
from src.data.preprocessor import preprocess_all
from src.vectorstore.pinecone_store import ingest_documents


def run_ingestion_pipeline(raw_data_path: str = "data/raw/drugs_raw.json"):
    """전체 데이터 적재 파이프라인을 실행합니다."""
    print("[1/4] API에서 데이터 수집 중...")
    raw_items = fetch_all_drugs(save_path=raw_data_path)
    print(f"  수집 완료: {len(raw_items)}건")

    print("[2/4] 데이터 전처리 중...")
    processed = preprocess_all(raw_items)
    print(f"  전처리 완료: {len(processed)}건")

    print("[3/4] LangChain 문서 생성 중...")
    documents = create_documents(processed)
    split_docs = split_documents(documents)
    print(f"  문서 생성 완료: {len(split_docs)}개 청크")

    print("[4/4] Pinecone에 업로드 중...")
    vector_store = ingest_documents(split_docs)
    print("적재 파이프라인 완료!")

    return vector_store


if __name__ == "__main__":
    run_ingestion_pipeline()
