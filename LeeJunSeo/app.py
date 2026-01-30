import streamlit as st

from src.chain.rag_chain import build_rag_chain_with_sources
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
    st.markdown("---")
    st.markdown("### 사용 안내")
    st.markdown(
        """
    이 시스템은 식품의약품안전처의 **e약은요**, **의약품 허가정보**
    데이터를 기반으로 의약품 정보를 제공합니다.

    **질문 예시:**
    - "타이레놀의 효능은 무엇인가요?"
    - "아세트아미노펜이 포함된 약은?"
    - "두통에 효과있는 약은?"
    - "아스피린의 부작용은?"
    - "겔포스와 함께 먹으면 안 되는 약은?"
    """
    )
    st.markdown("---")
    st.caption(f"분류기: {CLASSIFIER_MODEL}")
    st.caption(f"답변 생성: {LLM_MODEL}")
    st.caption("데이터: 식품의약품안전처 e약은요 + 허가정보")
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
st.caption("식품의약품안전처 e약은요 + 허가정보 데이터 기반 시스템")

# 대화 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.text(message["content"])
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
        with st.spinner("질문을 분석하고 답변을 생성하고 있습니다..."):
            result = st.session_state.chain.invoke(user_input)
            answer = result["answer"]
            source_drugs = result["source_drugs"]

            # 일반 텍스트로 표시 (markdown 해석 방지)
            st.text(answer.replace('\\n', '\n'))

            # 검색 정보 표시
            if result.get("category") and result.get("keyword"):
                category_labels = {
                    "product_name": "제품명",
                    "ingredient": "성분",
                    "efficacy": "효능",
                }
                cat_label = category_labels.get(result["category"], result["category"])
                st.caption(f"🔍 검색: {cat_label} → \"{result['keyword']}\"")

            # Top 5 결과 표시
            if source_drugs:
                st.subheader("📋 해당 의약품 정보 (Top 5)")
                for idx, drug in enumerate(source_drugs, 1):
                    with st.expander(f"{idx}. {drug.get('item_name', '정보없음')} - {drug.get('entp_name', '')}"):
                        # 주요 정보만 간단히 표시 (text로만 표시 - markdown 비활성화)
                        cols = st.columns(2)
                        with cols[0]:
                            st.text(f"제품명: {drug.get('item_name', '')}")
                            st.text(f"업체: {drug.get('entp_name', '')}")
                            st.text(f"품목코드: {drug.get('item_seq', '')}")
                        with cols[1]:
                            st.text(f"성상: {drug.get('chart', '-')}")
                            st.text(f"주성분: {drug.get('main_item_ingr', '-')}")
                            st.text(f"구분: {drug.get('spclty_pblc', '-')}")
                        
                        # 상세 정보 (모두 text로 표시, 이스케이프 문자 처리)
                        st.divider()
                        if drug.get('efcy_qesitm'):
                            st.subheader("효능", divider=False)
                            # \n을 실제 줄바꿈으로 변환
                            st.text(drug.get('efcy_qesitm', '').replace('\\n', '\n'))
                        if drug.get('use_method_qesitm'):
                            st.subheader("용법·용량", divider=False)
                            st.text(drug.get('use_method_qesitm', '').replace('\\n', '\n'))
                        if drug.get('se_qesitm'):
                            st.subheader("부작용", divider=False)
                            st.text(drug.get('se_qesitm', '').replace('\\n', '\n'))
                        if drug.get('atpn_qesitm'):
                            st.subheader("주의사항", divider=False)
                            st.text(drug.get('atpn_qesitm', '').replace('\\n', '\n'))
                
                # 출처 정보 저장
                sources = []
                for drug in source_drugs:
                    source_info = {
                        "item_name": drug.get("item_name", ""),
                        "entp_name": drug.get("entp_name", ""),
                        "item_seq": drug.get("item_seq", ""),
                    }
                    sources.append(source_info)
            else:
                sources = []

    # 어시스턴트 메시지 저장
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
