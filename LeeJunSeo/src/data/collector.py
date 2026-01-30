import json
import math
import time
from typing import Optional

import requests

from src.config import (
    DRUG_API_BASE_URL,
    DRUG_API_NUM_OF_ROWS,
    DRUG_APPROVAL_API_BASE_URL,
    MC_DATA_API,
)


def fetch_page(base_url: str, page_no: int, num_of_rows: int = DRUG_API_NUM_OF_ROWS,
               extra_params: dict | None = None) -> dict:
    """API에서 데이터 1페이지를 가져옵니다."""
    params = {
        "serviceKey": MC_DATA_API,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "type": "json",
    }
    if extra_params:
        params.update(extra_params)
    response = requests.get(base_url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_from_api(base_url: str, num_of_rows: int = DRUG_API_NUM_OF_ROWS,
                       extra_params: dict | None = None,
                       label: str = "데이터") -> list[dict]:
    """API에서 전체 데이터를 페이지네이션으로 수집합니다."""
    first_page = fetch_page(base_url, page_no=1, num_of_rows=1,
                            extra_params=extra_params)
    total_count = first_page["body"]["totalCount"]
    total_pages = math.ceil(total_count / num_of_rows)

    print(f"  [{label}] 총 {total_count}건, {total_pages}페이지 수집 시작...")

    all_items = []
    for page in range(1, total_pages + 1):
        data = fetch_page(base_url, page_no=page, num_of_rows=num_of_rows,
                          extra_params=extra_params)
        items = data["body"]["items"]
        all_items.extend(items)
        if page % 10 == 0 or page == total_pages:
            print(f"    페이지 {page}/{total_pages} - 누적 {len(all_items)}건")
        time.sleep(0.3)

    print(f"  [{label}] 수집 완료: 총 {len(all_items)}건")
    return all_items


def fetch_api1_easy_drug(save_path: Optional[str] = None) -> list[dict]:
    """API 1: e약은요 전체 약품 데이터를 수집합니다 (~4,740건)."""
    items = fetch_all_from_api(DRUG_API_BASE_URL, label="e약은요")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  저장 완료: {save_path}")

    return items


def fetch_api2_approval_info(save_path: Optional[str] = None) -> list[dict]:
    """API 2: 의약품 허가정보를 수집합니다 (~70,000건)."""
    items = fetch_all_from_api(DRUG_APPROVAL_API_BASE_URL, label="허가정보")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  저장 완료: {save_path}")

    return items


def fetch_all_data(raw_dir: str = "data/raw") -> dict:
    """2개 API 전체 수집 마스터 함수."""
    import os
    os.makedirs(raw_dir, exist_ok=True)

    print("=" * 60)
    print("API 1: e약은요 수집")
    print("=" * 60)
    api1_items = fetch_api1_easy_drug(save_path=f"{raw_dir}/drugs_raw.json")

    print()
    print("=" * 60)
    print("API 2: 허가정보 수집")
    print("=" * 60)
    api2_items = fetch_api2_approval_info(save_path=f"{raw_dir}/approval_raw.json")

    return {
        "api1": api1_items,
        "api2": api2_items,
    }


# 하위 호환성
def fetch_all_drugs(save_path: Optional[str] = None) -> list[dict]:
    """하위 호환성을 위한 래퍼 (API 1만 수집)."""
    return fetch_api1_easy_drug(save_path=save_path)
