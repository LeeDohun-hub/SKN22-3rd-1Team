"""전체 데이터 적재 파이프라인을 실행합니다: 수집 -> 병합 -> 전처리 -> Supabase 업로드"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore.ingest import run_ingestion_pipeline

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    run_ingestion_pipeline()
