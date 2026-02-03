"""FDA 의약품 정보 Q&A - 통합 Streamlit 앱 (app_3.py)"""
import re
import streamlit as st
from src.chain.rag_chain import prepare_context, stream_answer
from src.config import CLASSIFIER_MODEL, LLM_MODEL
from src.security import validate_user_input  # app.py의 보안 기능 유지

# 1. 페이지 설정
st.set_page_config(
    page_title="FDA 의약품 정보 Q&A",
    page_icon="💊",
    layout="wide",
)

# 2. CSS 스타일
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. 성분 섹션 후처리 함수
def _truncate_ingredient_section(answer: str) -> str:
    pattern = r"(###\s*💊\s*관련 성분 및 효능\n)(.*?)(?=\n###|\Z)"
    match = re.search(pattern, answer, flags=re.DOTALL)
    if not match:
        return answer

    header = match.group(1)
    body = match.group(2)
    lines = [line for line in body.strip().split('\n') if line.strip()]
    
    ingredient_lines = [line for line in lines if line.strip().startswith("- **")]
    
    if len(ingredient_lines) < 4:
        return answer
    
    first_three = ingredient_lines[:3]
    remaining = ingredient_lines[3:]
    
    new_body = "\n".join(first_three)
    expander_block = (
        f"\n\n**📋 나머지 성분 목록 (외 {len(remaining)}종)**\n\n"
        + "\n".join(remaining)
        + "\n\n---\n"
    )
    
    before_section = answer[:match.start()]
    after_section = answer[match.end():]
    
    updated = before_section + header + new_body + expander_block + after_section
    return updated

# 4. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 5. 사이드바 구성 (app.py의 상세 정보 포함)
with st.sidebar:
    st.title("💊 FDA 의약품 Q&A")
    st.markdown("---")

    st.markdown("### 사용 방법")
    st.markdown("""
    OpenFDA 데이터베이스를 실시간으로 검색하여
    FDA 승인 의약품 정보를 제공합니다.

    **검색 가능 항목:**
    - 브랜드명 (예: Tylenol, Advil)
    - 성분명 (예: acetaminophen, ibuprofen)
    - 증상/효능 (예: headache, pain)
    """)

    st.markdown("---")
    st.markdown("### 질문 예시")

    example_questions = [
        "Tylenol은 어떤 약인가요?",
        "ibuprofen 복용 시 주의사항은?",
        "두통약 추천해주세요",
        "aspirin과 함께 먹으면 안 되는 약은?",
        "acetaminophen 임산부 복용 가능한가요?",
    ]

    for q in example_questions:
        if st.button(q, key=f"example_{q}"):
            st.session_state.pending_question = q

    st.markdown("---")
    st.caption(f"분류 모델: {CLASSIFIER_MODEL}")
    st.caption(f"답변 모델: {LLM_MODEL}")
    st.caption("데이터: OpenFDA (api.fda.gov)")

    st.markdown("---")
    st.warning(
        "⚠️ 이 시스템은 일반적인 의약품 정보를 제공하며, "
        "전문적인 의료 조언을 대체하지 않습니다. "
        "정확한 복용은 의사 또는 약사와 상담하세요."
    )

    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 6. 메인 영역 제목
st.title("💊 FDA 의약품 정보 Q&A")
st.caption("OpenFDA 데이터베이스 실시간 검색 기반")

# 7. 대화 기록 표시 (상세 출처 표시 로직 포함)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        
        # 나머지 성분 부분을 expander로 분리하여 렌더링
        if "**📋 나머지 성분 목록" in content:
            parts = content.split("**📋 나머지 성분 목록")
            st.markdown(parts[0], unsafe_allow_html=True)
            
            expander_content = "**📋 나머지 성분 목록" + parts[1].split("---")[0]
            remaining_content = "---".join(parts[1].split("---")[1:]) if "---" in parts[1] else ""
            
            title_line = expander_content.split("\n")[0]
            items = "\n".join([line for line in expander_content.split("\n")[1:] if line.strip()])
            
            with st.expander(title_line):
                st.markdown(items, unsafe_allow_html=True)
            
            if remaining_content.strip():
                st.markdown(remaining_content, unsafe_allow_html=True)
        else:
            st.markdown(content, unsafe_allow_html=True)
        
        # Assistant 메시지인 경우 검색 정보와 원본 데이터 표시 (app.py 기능)
        if message["role"] == "assistant":
            if "search_info" in message:
                info = message["search_info"]
                st.caption(f"🔍 검색: {info['category']} → \"{info['keyword']}\"")

            if "sources" in message and message["sources"]:
                with st.expander("📋 원본 데이터 보기"):
                    for i, src in enumerate(message["sources"][:3], 1):
                        openfda = src.get("openfda", {})
                        brand = openfda.get("brand_name", ["N/A"])[0] if openfda.get("brand_name") else "N/A"
                        generic = openfda.get("generic_name", ["N/A"])[0] if openfda.get("generic_name") else "N/A"
                        manufacturer = openfda.get("manufacturer_name", ["N/A"])[0] if openfda.get("manufacturer_name") else "N/A"
                        st.markdown(f"**{i}. {brand}** ({generic})")
                        st.caption(f"제조사: {manufacturer}")

# 8. 공통 답변 생성 로직 함수 (중복 제거를 위해 정의)
def process_user_input(user_query):
    # 입력 검증 (보안)
    validation = validate_user_input(user_query)
    if not validation.is_valid:
        st.warning(f"입력 오류: {validation.error_message}")
        return

    safe_input = validation.sanitized_input

    # 사용자 메시지 추가 및 표시
    st.session_state.messages.append({"role": "user", "content": safe_input})
    with st.chat_message("user"):
        st.markdown(safe_input)

    # 답변 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("OpenFDA 데이터베이스 검색 중..."):
            context_data = prepare_context(safe_input)

        response_placeholder = st.empty()
        full_response = ""

        for chunk in stream_answer(context_data):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True)

        full_response = _truncate_ingredient_section(full_response)
        response_placeholder.markdown(full_response, unsafe_allow_html=True)

    # 메시지 저장 (출처 정보 포함)
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": context_data.get("raw_results", [])[:5],
        "search_info": {
            "category": context_data["category"],
            "keyword": context_data["keyword"],
        }
    })
    st.rerun()

# 9. 입력 이벤트 처리
# 예시 질문 클릭 시
if "pending_question" in st.session_state:
    pending_q = st.session_state.pending_question
    del st.session_state.pending_question
    process_user_input(pending_q)

# 채팅창 입력 시
if user_input := st.chat_input("약품이나 증상에 대해 질문하세요..."):
    process_user_input(user_input)