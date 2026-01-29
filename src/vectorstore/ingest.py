"""전체 데이터 적재 파이프라인: API 1 + API 2 수집 -> 병합 -> 전처리 -> Supabase 업로드"""

from src.data.loader import create_documents, split_documents
from src.data.preprocessor import (
    merge_api1_api2,
    prepare_drugs_for_db,
    preprocess_all,
)
from src.vectorstore.supabase_store import (
    ingest_documents,
    upload_drugs_to_supabase,
)


def run_ingestion_pipeline(raw_dir: str = "data/raw"):
    """전체 데이터 적재 파이프라인을 실행합니다.
    
    기존 수집된 JSON 파일(drugs_raw.json, approval_filtered.json)을 사용합니다.
    """
    import json
    import os

    # [1/5] 데이터 로드
    print("=" * 60)
    print("[1/5] 기존 수집 데이터 로드 중...")
    print("=" * 60)
    
    with open(f"{raw_dir}/drugs_raw.json", "r", encoding="utf-8") as f:
        api1_items = json.load(f)
    print(f"  API 1 (e약은요): {len(api1_items)}건")
    
    # 필터링된 approval 파일 사용
    approval_path = f"{raw_dir}/approval_filtered.json"
    if not os.path.exists(approval_path):
        raise FileNotFoundError(f"{approval_path} 파일이 없습니다. 먼저 필터링을 수행해주세요.")
    
    with open(approval_path, "r", encoding="utf-8") as f:
        api2_items = json.load(f)
    print(f"  API 2 (허가정보 필터링): {len(api2_items)}건")

    # [2/5] API 1 + API 2 데이터 병합
    print()
    print("=" * 60)
    print("[2/5] API 1 + API 2 데이터 병합 (item_seq 기준)...")
    print("=" * 60)
    merged_items = merge_api1_api2(api1_items, api2_items)

    # [3/5] 데이터 전처리
    print()
    print("=" * 60)
    print("[3/5] 데이터 전처리 중...")
    print("=" * 60)
    processed = preprocess_all(merged_items)
    print(f"  전처리 완료: {len(processed)}건")

    # [4/5] Supabase drugs 테이블에 업로드
    print()
    print("=" * 60)
    print("[4/5] Supabase drugs 테이블에 업로드 중...")
    print("=" * 60)
    drug_rows = prepare_drugs_for_db(merged_items)
    upload_drugs_to_supabase(drug_rows)

    # [5/5] LangChain 문서 생성 + 벡터 임베딩 업로드
    print()
    print("=" * 60)
    print("[5/5] 문서 생성 및 벡터 임베딩 업로드 중...")
    print("=" * 60)
    documents = create_documents(processed)
    split_docs = split_documents(documents)
    print(f"  문서 청크: {len(split_docs)}개")
    
    vector_store = ingest_documents(split_docs)

    print()
    print("=" * 60)
    print("✅ 적재 파이프라인 완료!")
    print("=" * 60)

    return vector_store


if __name__ == "__main__":
    run_ingestion_pipeline()
