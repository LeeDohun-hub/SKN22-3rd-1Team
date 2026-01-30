from src.config import SEARCH_LIMIT, SUPABASE_KEY, SUPABASE_URL
from supabase import create_client

# 분류 카테고리 → Supabase drugs 테이블 컬럼 매핑
CATEGORY_COLUMN_MAP = {
    "product_name": "item_name",
    "ingredient": "main_item_ingr",
    "efficacy": "efcy_qesitm",
}

# 행 데이터를 텍스트로 변환할 때 사용할 필드 라벨
FIELD_LABELS = {
    "item_name": "제품명",
    "entp_name": "업체명",
    "item_seq": "품목기준코드",
    "main_item_ingr": "주성분",
    "chart": "성상",
    "spclty_pblc": "전문/일반",
    "item_permit_date": "허가일자",
    "efcy_qesitm": "효능",
    "use_method_qesitm": "사용법",
    "atpn_warn_qesitm": "주의사항 경고",
    "atpn_qesitm": "주의사항",
    "intrc_qesitm": "상호작용",
    "se_qesitm": "부작용",
    "deposit_method_qesitm": "보관법",
    "storage_method": "저장방법",
    "valid_term": "유효기간",
}


def _get_client():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def search_drugs(category: str, keyword: str) -> list[dict]:
    """drugs 테이블에서 category에 해당하는 컬럼을 keyword로 ILIKE 검색합니다."""
    column = CATEGORY_COLUMN_MAP.get(category)
    if not column:
        return []

    client = _get_client()
    res = (
        client.table("drugs")
        .select("*")
        .ilike(column, f"%{keyword}%")
        .limit(SEARCH_LIMIT)
        .execute()
    )
    return res.data or []


def format_drug_info(row: dict) -> str:
    """drugs 테이블의 행 1건을 읽기 좋은 텍스트로 포맷합니다.
    HTML 태그 및 마크다운 취소선 형식만 제거."""
    import re
    
    def clean_value(value):
        """값에서 HTML 태그 및 마크다운 취소선만 제거."""
        if not value:
            return ""
        value = str(value).strip()
        
        # 1단계: HTML 태그 제거
        value = re.sub(r"<[^>]+>", "", value)
        
        # 2단계: 마크다운 취소선 형식만 제거 (~~텍스트~~)
        value = re.sub(r"~~[^~]+~~", "", value)
        
        # 3단계: HTML 엔티티 제거
        value = value.replace("&nbsp;", " ")
        value = value.replace("&lt;", "<")
        value = value.replace("&gt;", ">")
        value = value.replace("&amp;", "&")
        
        # 4단계: 공백 정규화
        value = re.sub(r"\s+", " ", value)
        
        return value.strip()
    
    lines = []
    for key, label in FIELD_LABELS.items():
        value = clean_value(row.get(key, ""))
        if value:
            lines.append(f"[{label}] {value}")
    return "\n".join(lines)


def format_search_results(rows: list[dict]) -> str:
    """검색 결과 전체를 하나의 컨텍스트 문자열로 포맷합니다."""
    if not rows:
        return "(검색 결과 없음)"
    parts = []
    for i, row in enumerate(rows, 1):
        header = f"── 검색 결과 {i} ──"
        body = format_drug_info(row)
        parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)
