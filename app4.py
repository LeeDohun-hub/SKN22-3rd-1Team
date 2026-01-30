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


# --- [수정 부분 1: CSS 스타일] ---
st.markdown("""
    <style>
    .chat-bubble {
        background-color: white;
        padding: 15px 20px;
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        display: inline-block;
        color: black;
        font-family: sans-serif;
        white-space: pre-wrap; 
        box-shadow: 1px 1px 5px rgba(0,0,0,0.05);
        word-break: break-all;
        line-height: 1.6; 
    }
    .user-message-group {
        display: flex;
        align-items: flex-start;
        justify-content: flex-end; 
        gap: 10px;
        width: 100%;
        margin-bottom: 20px;
    }
    .user-icon {
        width: 35px;
        height: 35px;
        background-color: #FF4B4B;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        order: 2; 
    }
    .user-bubble-container { order: 1; }
    </style>
""", unsafe_allow_html=True)


# --- [수정 부분 2: 가공 함수 보강] ---
def format_answer(text):
    """
    텍스트에 이미 줄바꿈이 섞여 있어도 강제로 '성분명:' 앞에 
    빈 줄을 만들어주는 더 강력한 로직입니다.
    """
    if not text:
        return text
    
    # 1. 모든 '성분명:' 앞에 줄바꿈 두 개(\n\n)를 넣습니다.
    text = text.replace("성분명:", "\n\n성분명:")
    
    # 2. 맨 처음에 오는 성분명 때문에 생긴 맨 위의 빈 줄만 분석 결과로 변경.
    return f'💉분석 결과\n {text}'


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


# 메인 UI
st.title("💊 의약품 정보 Q&A")
st.caption("식품의약품안전처 e약은요 + 허가정보 데이터 기반 시스템")


# --- [수정 부분 3: 대화 기록 표시 시 format_answer 적용] ---
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
            <div class="user-message-group">
                <div class="user-icon">👤</div>
                <div class="user-bubble-container">
                    <div class="chat-bubble">{message["content"]}</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    else:
        with st.chat_message("assistant"):
            # 출력 전에 텍스트를 가공하여 간격을 벌립니다.
            formatted_content = format_answer(message["content"])
            st.markdown(f'<div class="chat-bubble">{formatted_content}</div>', unsafe_allow_html=True)
            if "sources" in message and message["sources"]:
                with st.expander("📋 참고 자료 보기"):
                    for src in message["sources"]:
                        st.text(f"{src['item_name']} | 업체: {src['entp_name']} | 코드: {src['item_seq']}")


# --- [수정 부분 4: 채팅 입력 처리 시 format_answer 적용] ---
if user_input := st.chat_input("의약품에 대해 궁금한 점을 질문해주세요..."):
    # ❌ (삭제) st.session_state.messages.append({"role": "user", "content": user_input}) 
    # -> 이 줄을 여기서 지워야 중복 출력이 안 생깁니다.

    # 1. 사용자 질문을 화면에 즉시 렌더링 (커스텀 CSS 적용된 버전)
    st.markdown(f'''
        <div class="user-message-group">
            <div class="user-icon">👤</div>
            <div class="user-bubble-container">
                <div class="chat-bubble">{user_input}</div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # 2. 어시스턴트 답변 생성 과정
    with st.chat_message("assistant"):
        with st.spinner("정보를 검색하고 있습니다..."):
            prepared = prepare_context(user_input)
            source_drugs = prepared["source_drugs"]

        answer_placeholder = st.empty()
        full_answer = ""

        for chunk in stream_answer(prepared):
            full_answer += chunk
            display_stream = format_answer(full_answer)
            # 스트리밍 중인 임시 답변 표시
            answer_placeholder.markdown(f'<div class="chat-bubble">{display_stream}▌</div>', unsafe_allow_html=True)
        
        # 스트리밍 완료 후 최종 답변 확정 표시
        final_answer = format_answer(full_answer)
        answer_placeholder.markdown(f'<div class="chat-bubble">{final_answer}</div>', unsafe_allow_html=True)

        if prepared.get("category") and prepared.get("keyword"):
            st.caption(f"🔍 검색: {prepared['category']} → \"{prepared['keyword']}\"")

        # 소스 데이터 수집
        sources = []
        if source_drugs:
            with st.expander("📋 관련 의약품 정보"):
                for drug in source_drugs:
                    source_info = {
                        "item_name": drug.get("item_name", ""), 
                        "entp_name": drug.get("entp_name", ""), 
                        "item_seq": drug.get("item_seq", ""), 
                        "main_item_ingr": drug.get("main_item_ingr", "")
                    }
                    sources.append(source_info)
                    st.text(f"{source_info['item_name']} | 업체: {source_info['entp_name']}")

    # ---------------------------------------------------------
    # 3. ✨ [여기서부터 중요!] 모든 과정이 끝난 후 세션에 저장
    # ---------------------------------------------------------
    # (1) 사용자 질문 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # (2) 어시스턴트 답변 저장 (이미 위에서 선언된 full_answer와 sources 사용)
    st.session_state.messages.append({"role": "assistant", "content": full_answer, "sources": sources})
    
    # (3) 화면 새로고침 (이걸 해야 회색 잔상이 사라지고 상단 for문이 깔끔하게 다시 그립니다)
    st.rerun()