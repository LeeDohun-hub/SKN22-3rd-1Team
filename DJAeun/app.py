import streamlit as st

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

# 사이드바
with st.sidebar:
    st.title("의약품 정보 Q&A 시스템")
    st.text("사용 안내:")
    st.text(
        """
    이 시스템은 식품의약품안전처 공공데이터의 의약품 정보를 제공합니다.

    질문 예시:
    - 타이레놀의 효능은 무엇인가요?
    - 아세트아미노펜이 포함된 약은?
    - 두통에 효과있는 약은?
    """
    )
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
