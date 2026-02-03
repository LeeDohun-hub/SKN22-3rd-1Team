"""FDA 의약품 정보 Q&A - 통합 Streamlit 앱"""
import re
import streamlit as st
from src.chain.rag_chain import prepare_context, stream_answer
from src.config import CLASSIFIER_MODEL, LLM_MODEL, validate_env
from src.security import validate_user_input

# 환경 변수 검증
validate_env()

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
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


# 면책동의 다이얼로그
@st.dialog(title="⚠️ 면책사항 동의", width="large")
def disclaimer_dialog():
    """첫 진입 시 표시되는 면책사항 동의 팝업"""
    st.markdown(
        """

# **서비스 이용 약관 및 법적 면책 고지**

### 본 서비스는 OpenFDA 공공데이터를 기반으로 정보를 제공하는 데이터 검색 보조 도구입니다.

### 사용자는 본 서비스를 이용함과 동시에 아래의 모든 사항에 **동의**한 것으로 간주됩니다.

---

## **1. 의료 행위의 부인**

### 본 시스템이 제공하는 모든 정보는 일반적인 정보 제공만을 목적으로 하며, 의학적 진단, 치료, 처방 또는 복약 지도를 대신할 수 없습니다.

> **AI가 생성한 답변을 근거로 스스로 질병을 진단하거나 약물을 선택하여 복용하지 마십시오. 이는 오남용으로 인한 심각한 부작용을 초래할 수 있습니다.**

## **2. 정보의 정확성 및 최신성 보장 불가**

### 본 서비스는 생성형 AI(RAG) 기술을 사용합니다.

### AI의 특성상 환각 현상(Hallucination)이 발생할 수 있으며, OpenFDA 데이터의 내용과 다른 부정확하거나 왜곡된 정보를 제공할 가능성이 항상 존재합니다.

> 데이터베이스 업데이트 지연으로 인해 최신 의약품 정보나 허가 취소 사항이 반영되지 않았을 수 있습니다.

> 정보의 최종 확인은 반드시 공식적인 FDA 웹사이트(fda.gov) 또는 전문가를 통해 확인하시기 바랍니다.

## **3. 책임의 제한**

### 서비스 운영 주체는 본 서비스가 제공한 정보의 오류, 누락, 지연으로 인해 발생하는 어떠한 형태의 직접적·간접적·결과적 손해(신체적 부상, 질환의 악화, 경제적 손실 등)에 대해서도 법적 책임을 지지 않습니다.

> **사용자가 본 시스템의 정보를 신뢰하여 행한 모든 결정 및 행동에 대한 책임은 전적으로 사용자 본인에게 있습니다.**

## **4. 전문가 상담 필수**

### 증상이 있거나 의약품 성분에 대해 궁금한 점이 있을 경우, 반드시 전문의 또는 약사와 상담하십시오.

### 응급 상황이 발생한 경우, 본 시스템에 의존하지 말고 즉시 응급 의료 기관(911 등)에 연락하십시오.

## **5. 데이터 출처 및 오용 금지**

### 본 서비스는 OpenFDA의 공공데이터를 인용하나, FDA가 본 서비스의 운영이나 결과물을 보증하는 것은 아닙니다.

> 사용자는 본 서비스의 결과를 상업적으로 이용하거나, 타인에게 의학적 권고로 전달하여 발생하는 모든 법적 문제에 대해 단독으로 책임을 집니다.

---

## **6. 확인 및 동의**

본인은 위 면책 고지 사항을 충분히 숙지하였으며, 본 서비스가 제공하는 정보는 참고용일 뿐 의료 전문가의 조언을 대체할 수 없음에 동의합니다.

또한, 이를 어기고 발생한 모든 결과에 대해 서비스 제공자에게 책임을 묻지 않을 것을 서약합니다.
        """
    )
    
    # 체크박스 상태 확인
    checked = st.checkbox("**내용을 꼼꼼히 확인 했습니다.**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 동의합니다", type="primary", use_container_width=True, disabled=not checked):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    with col2:
        if st.button("❌ 거부합니다", use_container_width=True):
            # 거부 시 Google로 리다이렉트
            st.markdown(
                """
                <meta http-equiv="refresh" content="0; url=https://www.google.com">
                <script>window.location.href = 'https://www.google.com';</script>
                """,
                unsafe_allow_html=True,
            )
            st.stop()


# 면책동의 확인 - 동의하지 않으면 팝업 표시 후 중단
if not st.session_state.disclaimer_accepted:
    disclaimer_dialog()
    st.stop()

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
        "이부프로펜은 어떤 성분인가요??",
        "머리 아플때는 어떤 성분이 도움이 되나요??"
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