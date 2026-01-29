import re
from typing import Optional

# API 1 필드 라벨
FIELD_LABELS = {
    "efcyQesitm": "효능",
    "useMethodQesitm": "사용법",
    "atpnWarnQesitm": "주의사항 경고",
    "atpnQesitm": "주의사항",
    "intrcQesitm": "상호작용",
    "seQesitm": "부작용",
    "depositMethodQesitm": "보관법",
}

# API 2 필드 라벨
APPROVAL_FIELD_LABELS = {
    "CHART": "성상",
    "MAIN_ITEM_INGR": "주성분",
    "INGR_NAME": "원료성분",
    "PACK_UNIT": "포장단위",
    "STORAGE_METHOD": "저장방법",
    "VALID_TERM": "유효기간",
    "SPCLTY_PBLC": "전문/일반구분",
    "ITEM_ENG_NAME": "영문제품명",
    "PRDUCT_PRMISN_NO": "허가번호",
    "ITEM_PERMIT_DATE": "허가일자",
    "PERMIT_KIND_NAME": "허가종류",
    "CNSGN_MANUF": "위탁제조업체",
    "RARE_DRUG_YN": "희귀의약품 여부",
}


def clean_text(text: Optional[str]) -> str:
    """텍스트 필드를 정제합니다: HTML 태그 제거, 공백 정규화."""
    if text is None or str(text).strip() in ("", "None"):
        return ""
    text = str(text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def merge_api1_api2(api1_items: list[dict], api2_items: list[dict]) -> list[dict]:
    """API 1(e약은요)과 API 2(허가정보)를 item_seq 기준으로 LEFT JOIN 병합합니다.

    API 1 기준으로 API 2 데이터를 보강합니다. API 1에 없는 API 2 데이터는 무시됩니다.
    """
    # API 2를 itemSeq 기준 dict로 변환
    api2_lookup = {}
    for item in api2_items:
        seq = item.get("ITEM_SEQ") or item.get("itemSeq")
        if seq:
            api2_lookup[str(seq)] = item

    merged = []
    matched = 0
    for item in api1_items:
        item_seq = str(item.get("itemSeq", ""))
        api2_item = api2_lookup.get(item_seq, {})
        if api2_item:
            matched += 1

        # API 2 필드를 API 1 항목에 병합
        item["_api2"] = {
            "ITEM_ENG_NAME": api2_item.get("ITEM_ENG_NAME") or "",
            "CHART": api2_item.get("CHART") or "",
            "MAIN_ITEM_INGR": api2_item.get("MAIN_ITEM_INGR") or "",
            "INGR_NAME": api2_item.get("INGR_NAME") or "",
            "PACK_UNIT": api2_item.get("PACK_UNIT") or "",
            "STORAGE_METHOD": api2_item.get("STORAGE_METHOD") or "",
            "VALID_TERM": api2_item.get("VALID_TERM") or "",
            "SPCLTY_PBLC": api2_item.get("SPCLTY_PBLC") or "",
            "PRDUCT_PRMISN_NO": api2_item.get("PRDUCT_PRMISN_NO") or "",
            "ITEM_PERMIT_DATE": api2_item.get("ITEM_PERMIT_DATE") or "",
            "PERMIT_KIND_NAME": api2_item.get("PERMIT_KIND_NAME") or "",
            "CNSGN_MANUF": api2_item.get("CNSGN_MANUF") or "",
            "RARE_DRUG_YN": api2_item.get("RARE_DRUG_YN") or "",
            "CANCEL_DATE": api2_item.get("CANCEL_DATE") or "",
            "CANCEL_NAME": api2_item.get("CANCEL_NAME") or "",
        }
        merged.append(item)

    print(f"  병합 완료: API 1 {len(api1_items)}건 중 {matched}건 매칭")
    return merged


def compose_drug_document(item: dict) -> str:
    """약품 1건을 구조화된 텍스트 문서로 구성합니다 (API 1 + API 2 병합 데이터)."""
    lines = []
    lines.append(f"제품명: {item.get('itemName', '정보 없음')}")
    lines.append(f"업체명: {item.get('entpName', '정보 없음')}")
    lines.append(f"품목기준코드: {item.get('itemSeq', '정보 없음')}")

    # API 2 기본 정보
    api2 = item.get("_api2", {})
    main_ingr = clean_text(api2.get("MAIN_ITEM_INGR"))
    if main_ingr:
        lines.append(f"주성분: {main_ingr}")
    chart = clean_text(api2.get("CHART"))
    if chart:
        lines.append(f"성상: {chart}")
    spclty = clean_text(api2.get("SPCLTY_PBLC"))
    if spclty:
        lines.append(f"전문/일반: {spclty}")
    permit_date = clean_text(api2.get("ITEM_PERMIT_DATE"))
    if permit_date:
        lines.append(f"허가일자: {permit_date}")
    rare = clean_text(api2.get("RARE_DRUG_YN"))
    if rare:
        lines.append(f"희귀의약품: {rare}")

    lines.append("")

    # API 1 필드
    for field_key, label in FIELD_LABELS.items():
        value = clean_text(item.get(field_key))
        if value:
            lines.append(f"[{label}]")
            lines.append(value)
            lines.append("")

    # API 2 추가 필드
    storage = clean_text(api2.get("STORAGE_METHOD"))
    if storage:
        lines.append(f"[저장방법]")
        lines.append(storage)
        lines.append("")
    valid_term = clean_text(api2.get("VALID_TERM"))
    if valid_term:
        lines.append(f"[유효기간]")
        lines.append(valid_term)
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
    """벡터 저장소에 저장할 메타데이터를 추출합니다. None 값은 빈 문자열로 변환."""
    api2 = item.get("_api2", {})
    return {
        "item_name": item.get("itemName") or "",
        "entp_name": item.get("entpName") or "",
        "item_seq": item.get("itemSeq") or "",
        "open_de": item.get("openDe") or "",
        "update_de": item.get("updateDe") or "",
        "item_image": item.get("itemImage") or "",
        "main_item_ingr": clean_text(api2.get("MAIN_ITEM_INGR")),
        "spclty_pblc": clean_text(api2.get("SPCLTY_PBLC")),
        "item_permit_date": clean_text(api2.get("ITEM_PERMIT_DATE")),
    }


def preprocess_all(raw_items: list[dict]) -> list[dict]:
    """전체 원본 데이터를 문서 생성 가능한 형태로 전처리합니다.

    각 항목은 다음 키를 포함합니다:
    - text: 효능 텍스트 (검색/임베딩 대상)
    - metadata: 메타데이터 + full_text (LLM 컨텍스트용 전체 텍스트)
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


def prepare_drugs_for_db(merged_items: list[dict]) -> list[dict]:
    """병합된 약품 데이터를 Supabase drugs 테이블에 맞는 dict 리스트로 변환합니다."""
    rows = []
    for item in merged_items:
        api2 = item.get("_api2", {})
        row = {
            "item_seq": str(item.get("itemSeq", "")),
            "item_name": item.get("itemName") or "",
            "entp_name": item.get("entpName") or "",
            "efcy_qesitm": clean_text(item.get("efcyQesitm")),
            "use_method_qesitm": clean_text(item.get("useMethodQesitm")),
            "atpn_warn_qesitm": clean_text(item.get("atpnWarnQesitm")),
            "atpn_qesitm": clean_text(item.get("atpnQesitm")),
            "intrc_qesitm": clean_text(item.get("intrcQesitm")),
            "se_qesitm": clean_text(item.get("seQesitm")),
            "deposit_method_qesitm": clean_text(item.get("depositMethodQesitm")),
            "open_de": item.get("openDe") or "",
            "update_de": item.get("updateDe") or "",
            "item_image": item.get("itemImage") or "",
            "bizrno": item.get("bizrno") or "",
            # API 2 필드
            "item_eng_name": api2.get("ITEM_ENG_NAME") or "",
            "chart": clean_text(api2.get("CHART")),
            "main_item_ingr": clean_text(api2.get("MAIN_ITEM_INGR")),
            "ingr_name": clean_text(api2.get("INGR_NAME")),
            "pack_unit": clean_text(api2.get("PACK_UNIT")),
            "storage_method": clean_text(api2.get("STORAGE_METHOD")),
            "valid_term": clean_text(api2.get("VALID_TERM")),
            "spclty_pblc": clean_text(api2.get("SPCLTY_PBLC")),
            "prduct_prmisn_no": api2.get("PRDUCT_PRMISN_NO") or "",
            "item_permit_date": api2.get("ITEM_PERMIT_DATE") or "",
            "permit_kind_name": clean_text(api2.get("PERMIT_KIND_NAME")),
            "cnsgn_manuf": clean_text(api2.get("CNSGN_MANUF")),
            "rare_drug_yn": api2.get("RARE_DRUG_YN") or "",
            "cancel_date": api2.get("CANCEL_DATE") or "",
            "cancel_name": api2.get("CANCEL_NAME") or "",
        }
        if row["item_seq"]:
            rows.append(row)
    return rows


