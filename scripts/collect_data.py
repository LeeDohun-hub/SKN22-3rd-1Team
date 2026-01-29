"""API에서 전체 약품 데이터를 수집하여 JSON 파일로 저장합니다."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.collector import fetch_all_drugs

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    fetch_all_drugs(save_path="data/raw/drugs_raw.json")
