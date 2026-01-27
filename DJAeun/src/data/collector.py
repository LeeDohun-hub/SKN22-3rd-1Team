import json
import math
import time
from typing import Optional

import requests

from src.config import DRUG_API_BASE_URL, DRUG_API_NUM_OF_ROWS, MC_DATA_API


def fetch_drug_page(page_no: int, num_of_rows: int = DRUG_API_NUM_OF_ROWS) -> dict:
    """API에서 약품 데이터 1페이지를 가져옵니다."""
    params = {
        "serviceKey": MC_DATA_API,
        "pageNo": page_no,
        "numOfRows": num_of_rows,
        "type": "json",
    }
    response = requests.get(DRUG_API_BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_all_drugs(save_path: Optional[str] = None) -> list[dict]:
    """
    API에서 전체 약품 데이터를 페이지네이션으로 수집합니다.
    총 4,740건, 100건/페이지 = 48페이지.
    """
    first_page = fetch_drug_page(page_no=1, num_of_rows=1)
    total_count = first_page["body"]["totalCount"]
    total_pages = math.ceil(total_count / DRUG_API_NUM_OF_ROWS)

    print(f"총 {total_count}건, {total_pages}페이지 수집 시작...")

    all_items = []
    for page in range(1, total_pages + 1):
        data = fetch_drug_page(page_no=page)
        items = data["body"]["items"]
        all_items.extend(items)
        print(f"  페이지 {page}/{total_pages} - {len(items)}건 수집")
        time.sleep(0.3)

    print(f"수집 완료: 총 {len(all_items)}건")

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        print(f"저장 완료: {save_path}")

    return all_items
