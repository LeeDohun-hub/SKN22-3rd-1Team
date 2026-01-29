import streamlit as st
import streamlit.components.v1 as components

from src.chain.rag_chain import build_rag_chain_with_sources, prepare_context, stream_answer
from src.config import CLASSIFIER_MODEL, LLM_MODEL

# 페이지 설정
st.set_page_config(
    page_title="의약품 정보 Q&A",
    page_icon="💊",
    layout="wide",
)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = build_rag_chain_with_sources()
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False


# 면책동의 다이얼로그
@st.dialog(title="⚠️ 면책사항 동의", width="large")
def disclaimer_dialog():
    """첫 진입 시 표시되는 면책사항 동의 팝업"""
    st.markdown(
        """
        ### 📋 서비스 이용 전 안내사항
        
        이 시스템은 **식품의약품안전처 공공데이터**를 기반으로 일반적인 의약품 정보를 제공합니다.
        
        ---
        
        #### ⚠️ 중요 주의사항
        
        🔴 이 시스템의 응답은 AI가 공공 데이터를 기반으로 생성한 것으로, **정확성을 보장하지 않습니다.**
        
        🔴 복약지시나 진단으로 해석될 수 있는 답변이 출력될 경우, 이는 **시스템 오류이며 의도된 것이 아닙니다.**
        
        🔴 **모든 의약품 복용 및 건강 관련 결정은 반드시 의사 또는 약사와 상담 후 진행하세요.**
        
        🔴 본 시스템 사용으로 인한 **어떠한** 직접적, 간접적 **피해**에 대해서도 **책임지지 않습니다.**
        
        ---
        
        위 내용을 이해하고 동의하시면 서비스를 이용하실 수 있습니다.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ 동의합니다", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    with col2:
        if st.button("❌ 거부합니다", use_container_width=True):
            # 거부 시 Google로 리다이렉트 (브라우저에서 window.close()는 제한적)
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


# 클립보드 복사 버튼 생성 함수
def copy_button(text: str, button_text: str):
    """클릭 시 텍스트를 클립보드에 복사하는 버튼 생성"""
    html_code = f"""
    <button onclick="navigator.clipboard.writeText('{text}').then(() => {{
        this.innerHTML = '✅ 복사됨!';
        setTimeout(() => {{ this.innerHTML = '{button_text}'; }}, 1500);
    }})" style="
        padding: 8px 12px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        width: 100%;
        margin: 4px 0;
        transition: transform 0.2s, box-shadow 0.2s;
    " onmouseover="this.style.transform='scale(1.02)'; this.style.boxShadow='0 4px 12px rgba(102,126,234,0.4)';"
       onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
        {button_text}
    </button>
    """
    components.html(html_code, height=50)


# 사이드바
with st.sidebar:
    st.title("의약품 정보 Q&A 시스템")
    st.text("사용 안내:")
    st.text(
        """
    이 시스템은 식품의약품안전처 공공데이터의 의약품 정보를 제공합니다.
    """
    )

    st.text("📝 질문 예시 (클릭하여 복사):")
    copy_button("타이레놀의 효능은 무엇인가요?", "💊 타이레놀의 효능은 무엇인가요?")
    copy_button("아세트아미노펜이 포함된 약은?", "🧪 아세트아미노펜이 포함된 약은?")
    copy_button("두통에 효과있는 약은?", "🩹 두통에 효과있는 약은?")
    st.caption(f"분류기: {CLASSIFIER_MODEL}")
    st.caption(f"답변 생성: {LLM_MODEL}")
    st.caption("데이터: 식품의약품안전처 e약은요 + 허가정보")
    st.warning(
        "⚠️ 이 시스템은 일반적인 의약품 정보를 제공하며, "
        "의학적 진단이나 처방을 대체하지 않습니다. "
        "반드시 의사 또는 약사와 상담하세요."
    )
    if st.button("대화 초기화"):
        st.session_state.messages = []
        st.rerun()

# 메인 채팅 인터페이스
st.title("💊 의약품 정보 Q&A")
st.caption("식품의약품안전처 e약은요 + 허가정보 데이터 기반 시스템")

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📋 참고 자료 보기"):
                for src in message["sources"]:
                    st.text(
                        f"{src['item_name']} | "
                        f"업체: {src['entp_name']} | "
                        f"품목코드: {src['item_seq']}"
                    )

# 채팅 입력
if user_input := st.chat_input("의약품에 대해 궁금한 점을 질문해주세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # 답변 생성
    with st.chat_message("assistant"):
        # 1단계: 분류 + 검색 (일괄 처리)
        with st.spinner("질문을 분석하고 정보를 검색하고 있습니다..."):
            prepared = prepare_context(user_input)
            source_drugs = prepared["source_drugs"]

        # 2단계: 답변 생성 (스트리밍)
        answer_placeholder = st.empty()
        full_answer = ""

        for chunk in stream_answer(prepared):
            full_answer += chunk
            answer_placeholder.text(full_answer)

        # 검색 정보 표시
        if prepared.get("category") and prepared.get("keyword"):
            category_labels = {
                "product_name": "제품명",
                "ingredient": "성분",
                "efficacy": "효능",
            }
            cat_label = category_labels.get(prepared["category"], prepared["category"])
            st.caption(f"🔍 검색 과정: {cat_label} → \"{prepared['keyword']}\"")

        # 출처 표시
        sources = []
        if source_drugs:
            with st.expander("📋 관련 의약품 정보"):
                for drug in source_drugs:
                    source_info = {
                        "item_name": drug.get("item_name", ""),
                        "entp_name": drug.get("entp_name", ""),
                        "item_seq": drug.get("item_seq", ""),
                        "main_item_ingr": drug.get("main_item_ingr", ""),
                    }
                    sources.append(source_info)
                    st.text(
                        f"{source_info['item_name']} | "
                        f"성분: {source_info['main_item_ingr']} | "
                        f"업체: {source_info['entp_name']}"
                    )

    # 어시스턴트 메시지 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
        }
    )
