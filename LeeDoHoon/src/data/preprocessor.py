import re
from typing import Optional

FIELD_LABELS = {
    "efcyQesitm": "효능",
    "useMethodQesitm": "사용법",
    "atpnWarnQesitm": "주의사항 경고",
    "atpnQesitm": "주의사항",
    "intrcQesitm": "상호작용",
    "seQesitm": "부작용",
    "depositMethodQesitm": "보관법",
}


def clean_text(text: Optional[str]) -> str:
    """텍스트 필드를 정제합니다: HTML 태그 제거, 공백 정규화."""
    if text is None or str(text).strip() in ("", "None"):
        return ""
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compose_drug_document(item: dict) -> str:
    """약품 1건을 구조화된 텍스트 문서로 구성합니다."""
    lines = []
    lines.append(f"제품명: {item.get('itemName', '정보 없음')}")
    lines.append(f"업체명: {item.get('entpName', '정보 없음')}")
    lines.append(f"품목기준코드: {item.get('itemSeq', '정보 없음')}")
    lines.append("")

    for field_key, label in FIELD_LABELS.items():
        value = clean_text(item.get(field_key))
        if value:
            lines.append(f"[{label}]")
            lines.append(value)
            lines.append("")

    return "\n".join(lines).strip()


def compose_efficacy_document(item: dict) -> str:
    """약품 1건의 제품명 + 효능(efcyQesitm)만 텍스트로 구성합니다."""
    item_name = item.get("itemName", "정보 없음")
    efcy = clean_text(item.get("efcyQesitm"))
    if not efcy:
        return f"제품명: {item_name}"
    return f"제품명: {item_name}\n\n[효능]\n{efcy}"


def extract_metadata(item: dict) -> dict:
    """Pinecone에 저장할 메타데이터를 추출합니다. None 값은 빈 문자열로 변환."""
    return {
        "item_name": item.get("itemName") or "",
        "entp_name": item.get("entpName") or "",
        "item_seq": item.get("itemSeq") or "",
        "open_de": item.get("openDe") or "",
        "update_de": item.get("updateDe") or "",
        "item_image": item.get("itemImage") or "",
    }


def preprocess_all(raw_items: list[dict]) -> list[dict]:
    """전체 원본 데이터를 문서 생성 가능한 형태로 전처리합니다.

    각 항목은 다음 키를 포함합니다:
    - text: 효능 텍스트 (검색/임베딩 대상)
    - metadata: Pinecone 메타데이터 + full_text (LLM 컨텍스트용 전체 텍스트)
    """
    processed = []
    for item in raw_items:
        full_text = compose_drug_document(item)
        efcy_text = compose_efficacy_document(item)
        metadata = extract_metadata(item)
        metadata["full_text"] = full_text
        if full_text and metadata.get("item_name"):
            processed.append({"text": efcy_text, "metadata": metadata})
    return processed
