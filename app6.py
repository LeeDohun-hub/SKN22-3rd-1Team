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

# **서비스 이용 약관 및 법적 면책 고지**

### 본 서비스는 식품의약품안전처 공공데이터를 기반으로 정보를 제공하는 데이터 검색 보조 도구입니다.

### 사용자는 본 서비스를 이용함과 동시에 아래의 모든 사항에 **동의**한 것으로 간주됩니다.

---

## **1. 의료 행위의 부인**

### 본 시스템이 제공하는 모든 정보는 일반적인 정보 제공만을 목적으로 하며, 의학적 진단, 치료, 처방 또는 복약 지도를 대신할 수 없습니다.

> **AI가 생성한 답변을 근거로 스스로 질병을 진단하거나 약물을 선택하여 복용하지 마십시오. 이는 오남용으로 인한 심각한 부작용을 초래할 수 있습니다.**

## **2. 정보의 정확성 및 최신성 보장 불가**

### 본 서비스는 생성형 AI(RAG) 기술을 사용합니다.

### AI의 특성상 환각 현상(Hallucination)이 발생할 수 있으며, 공공데이터의 내용과 다른 부정확하거나 왜곡된 정보를 제공할 가능성이 항상 존재합니다.

> 데이터베이스 업데이트 지연으로 인해 최신 의약품 정보나 허가 취소 사항이 반영되지 않았을 수 있습니다.

> 정보의 최종 확인은 반드시 공식적인 식약처 의약품안전나라 또는 전문가를 통해 확인하시기 바랍니다.

## **3. 책임의 제한**

### 서비스 운영 주체는 본 서비스가 제공한 정보의 오류, 누락, 지연으로 인해 발생하는 어떠한 형태의 직접적·간접적·결과적 손해(신체적 부상, 질환의 악화, 경제적 손실 등)에 대해서도 법적 책임을 지지 않습니다.

> **사용자가 본 시스템의 정보를 신뢰하여 행한 모든 결정 및 행동에 대한 책임은 전적으로 사용자 본인에게 있습니다.**

## **4. 전문가 상담 필수**

### 증상이 있거나 의약품 성분에 대해 궁금한 점이 있을 경우, 반드시 전문의 또는 약사와 상담하십시오.

### 응급 상황이 발생한 경우, 본 시스템에 의존하지 말고 즉시 응급 의료 기관(119 등)에 연락하십시오.

## **5. 데이터 출처 및 오용 금지**

### 본 서비스는 식약처의 공공데이터를 인용하나, 식약처가 본 서비스의 운영이나 결과물을 보증하는 것은 아닙니다.

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

            if message.get("dur_data"):
                with st.expander("⚠️ 병용금지 주의 약물 목록", expanded=False):
                    for ingredient, contraindications in message["dur_data"].items():
                        st.markdown(f"**[{ingredient}]** 과 함께 복용하면 안 되는 성분:")
                        seen_mixtures = set()
                        for item in contraindications:
                            mixture = item.get("MIXTURE_INGR_KOR_NAME") or item.get("mixture_ingr_kor_name", "")
                            reason = item.get("PROHBT_CONTENT") or item.get("prohbt_content", "")
                            if mixture and mixture not in seen_mixtures:
                                seen_mixtures.add(mixture)
                                st.markdown(f"- {mixture}: {reason}")
                        st.divider()

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

        # 병용금지 경고 UI
        dur_data = prepared.get("dur_data", {})

        # 각 성분별 병용금지 약물 목록
        if dur_data:
            with st.expander("⚠️ 병용금지 주의 약물 목록", expanded=False):
                for ingredient, contraindications in dur_data.items():
                    st.markdown(f"**[{ingredient}]** 과 함께 복용하면 안 되는 성분:")
                    seen_mixtures = set()
                    for item in contraindications:
                        mixture = item.get("MIXTURE_INGR_KOR_NAME") or item.get("mixture_ingr_kor_name", "")
                        reason = item.get("PROHBT_CONTENT") or item.get("prohbt_content", "")
                        if mixture and mixture not in seen_mixtures:
                            seen_mixtures.add(mixture)
                            st.markdown(f"- {mixture}: {reason}")
                    st.divider()

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
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
        "dur_data": dur_data,
    })
    
    # (3) 화면 새로고침 (이걸 해야 회색 잔상이 사라지고 상단 for문이 깔끔하게 다시 그립니다)
    st.rerun()