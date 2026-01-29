import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# HeeJoon 디렉토리를 Python 경로에 추가 (모듈 import 가능하도록)
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import psycopg2
import requests

from src.config import (
    SUPABASE_URL,
    MIXTURE_API_BASE_URL,
    MIXTURE_API_NUM_OF_ROWS,
    MIXTURE_API_SERVICE_KEY,
)
from src.vectorstore.supabase_store import get_supabase_client

# Mixture 테이블 컬럼
MIXTURE_COLUMNS = [
    "TYPE_NAME",
    "MIX_TYPE",
    "INGR_CODE",
    "INGR_ENG_NAME",
    "INGR_KOR_NAME",
    "MIX",
    "ORI",
    "CLASS",
    "MIXTURE_MIX_TYPE",
    "MIXTURE_INGR_CODE",
    "MIXTURE_INGR_ENG_NAME",
    "MIXTURE_INGR_KOR_NAME",
    "MIXTURE_MIX",
    "MIXTURE_ORI",
    "MIXTURE_CLASS",
    "NOTIFICATION_DATE",
    "PROHBT_CONTENT",
    "REMARK",
    "DEL_YN",
]


def fetch_page(base_url: str, page_no: int, num_of_rows: int = MIXTURE_API_NUM_OF_ROWS,
               service_key: Optional[str] = None) -> dict:
    """API에서 데이터 1페이지를 가져옵니다."""
    service_key = service_key or MIXTURE_API_SERVICE_KEY
    params = {
        "serviceKey": service_key,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "type": "json",
    }
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_from_api(base_url: str, num_of_rows: int = MIXTURE_API_NUM_OF_ROWS,
                       service_key: Optional[str] = None,
                       label: str = "혼합성분 데이터") -> list[dict]:
    """API에서 전체 데이터를 페이지네이션으로 수집합니다."""
    service_key = service_key or MIXTURE_API_SERVICE_KEY
    
    first_page = fetch_page(base_url, page_no=1, num_of_rows=1, service_key=service_key)
    total_count = first_page.get("body", {}).get("totalCount", 0)
    total_pages = math.ceil(total_count / num_of_rows) if num_of_rows > 0 else 1

    print(f"  [{label}] 총 {total_count}건, {total_pages}페이지 수집 시작...")

    all_items = []
    for page in range(1, total_pages + 1):
        data = fetch_page(base_url, page_no=page, num_of_rows=num_of_rows, service_key=service_key)
        items = data.get("body", {}).get("items", [])
        # API 응답이 {"item": {...}} 형태로 nested되어 있으므로 추출
        for item_wrapper in items:
            if isinstance(item_wrapper, dict) and "item" in item_wrapper:
                all_items.append(item_wrapper["item"])
            else:
                all_items.append(item_wrapper)
        if page % 10 == 0 or page == total_pages:
            print(f"    페이지 {page}/{total_pages} - 누적 {len(all_items)}건")
        time.sleep(0.3)

    print(f"  [{label}] 수집 완료: 총 {len(all_items)}건")
    return all_items


def _parse_date_yyyymmdd(value: str) -> Optional[str]:
    """YYYYMMDD 형식을 ISO 날짜로 변환합니다."""
    if not value or value is None:
        return None
    s = str(value).strip()
    if len(s) == 8 and s.isdigit():
        try:
            return f"{s[:4]}-{s[4:6]}-{s[6:8]}"  # YYYYMMDD → YYYY-MM-DD
        except Exception:
            pass
    return s


def clean_record(raw: dict) -> dict:
    """원본 레코드를 정제합니다. 대소문자 자동 처리 + DEL_YN 문자열 변환."""
    out = {}
    
    for col in MIXTURE_COLUMNS:
        # 원본 필드명, 소문자, 대문자 순서로 찾기
        val = None
        if col in raw:
            val = raw[col]
        elif col.lower() in raw:
            val = raw[col.lower()]
        elif col.upper() in raw:
            val = raw[col.upper()]
        
        # NOTIFICATION_DATE는 YYYYMMDD → YYYY-MM-DD로 변환
        if col == "NOTIFICATION_DATE":
            out[col] = _parse_date_yyyymmdd(val)
        # DEL_YN은 문자열 "정상" → False, 나머지 → True로 변환
        elif col == "DEL_YN":
            if val == "정상" or val in (False, "f", "false", "False", "N", "n", "0", 0):
                out[col] = False
            else:
                out[col] = True
        # 나머지는 string으로 변환 (None은 빈 문자열)
        else:
            out[col] = str(val).strip() if val is not None else ""
    return out


def ensure_table_exists(table_name: str = "mixtures") -> None:
    """Supabase 테이블이 존재하는지 확인합니다.
    (테이블 생성은 Supabase 대시보드에서 수동으로 진행)
    """
    print(f"  '{table_name}' 테이블이 Supabase에 이미 생성되어 있다고 가정합니다.")


def upsert_to_supabase(rows: list[dict], table_name: str = "mixtures", batch_size: int = 500) -> None:
    """정제된 데이터를 Supabase에 upsert합니다."""
    client = get_supabase_client()
    total = len(rows)
    if total == 0:
        print("  upsert할 데이터 없음")
        return

    batches = math.ceil(total / batch_size)
    for i in range(batches):
        start = i * batch_size
        end = start + batch_size
        batch = rows[start:end]
        print(f"  배치 {i+1}/{batches} upsert 중 ({len(batch)}건)...")
        client.table(table_name).upsert(batch).execute()

    print(f"  upsert 완료: {total}건")


def fetch_mixture_data(save_path: Optional[str] = None) -> list[dict]:
    """API에서 혼합성분 데이터를 수집합니다."""
    if not MIXTURE_API_BASE_URL:
        raise ValueError(".env에서 MIXTURE_API_BASE_URL을 설정해주세요")
    if not MIXTURE_API_SERVICE_KEY:
        raise ValueError(".env에서 MIXTURE_API_SERVICE_KEY를 설정해주세요")

    items = fetch_all_from_api(MIXTURE_API_BASE_URL, label="혼합성분")

    if save_path:
        # 디렉토리 자동 생성
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  저장 완료: {save_path}")

    return items


def ingest_mixture_to_supabase(save_raw: bool = False) -> None:
    """혼합성분 데이터 수집 → 정제 → Supabase 업로드 마스터 함수."""
    print("=" * 60)
    print("혼합성분 데이터 수집 및 업로드")
    print("=" * 60)

    # 1) 수집
    raw_path = "data/raw/mixture_raw.json" if save_raw else None
    raw_items = fetch_mixture_data(save_path=raw_path)
    print()

    # 2) 정제
    print("데이터 정제 중...")
    cleaned_items = [clean_record(item) for item in raw_items]
    print(f"  정제 완료: {len(cleaned_items)}건")
    print()

    # 3) 테이블 생성
    print("테이블 생성 확인 중...")
    ensure_table_exists(table_name="mixtures")
    print()

    # 4) Supabase에 upsert
    print("Supabase에 데이터 업로드 중...")
    upsert_to_supabase(cleaned_items, table_name="mixtures")
    print()
    print("=" * 60)
    print("완료!")
    print("=" * 60)


if __name__ == "__main__":
    ingest_mixture_to_supabase(save_raw=True)
