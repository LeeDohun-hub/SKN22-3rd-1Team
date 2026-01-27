import streamlit as st

from src.chain.rag_chain import build_rag_chain_with_sources
from src.config import LLM_MODEL

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
    st.markdown("---")
    st.markdown("### 사용 안내")
    st.markdown(
        """
    이 시스템은 식품의약품안전처의 **e약은요** 데이터를 기반으로
    의약품 정보를 제공합니다.

    **질문 예시:**
    - "타이레놀의 효능은 무엇인가요?"
    - "아스피린의 부작용은?"
    - "활명수는 어떻게 복용하나요?"
    - "겔포스와 함께 먹으면 안 되는 약은?"
    """
    )
    st.markdown("---")
    st.caption(f"모델: {LLM_MODEL}")
    st.caption("데이터: 식품의약품안전처 e약은요 (4,740건)")
    st.markdown("---")
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
st.caption("한국 식품의약품안전처 e약은요 데이터 기반 RAG 시스템")

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📋 참고 자료 보기"):
                for src in message["sources"]:
                    st.markdown(
                        f"**{src['item_name']}** | "
                        f"업체: {src['entp_name']} | "
                        f"품목코드: {src['item_seq']}"
                    )

# 채팅 입력
if user_input := st.chat_input("의약품에 대해 궁금한 점을 질문해주세요..."):
    # 사용자 메시지 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 답변 생성
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하고 있습니다..."):
            result = st.session_state.chain.invoke(user_input)
            answer = result["answer"]
            source_docs = result["source_docs"]

            st.markdown(answer)

            # 출처 표시
            sources = []
            if source_docs:
                with st.expander("📋 참고 자료 보기"):
                    for doc in source_docs:
                        meta = doc.metadata
                        source_info = {
                            "item_name": meta.get("item_name", ""),
                            "entp_name": meta.get("entp_name", ""),
                            "item_seq": meta.get("item_seq", ""),
                        }
                        sources.append(source_info)
                        st.markdown(
                            f"**{source_info['item_name']}** | "
                            f"업체: {source_info['entp_name']} | "
                            f"품목코드: {source_info['item_seq']}"
                        )

    # 어시스턴트 메시지 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
