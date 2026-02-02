import json
from typing import Generator

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.chain.prompts import ANSWER_PROMPT, CLASSIFIER_PROMPT
from src.chain.retriever import (
    check_mutual_contraindication,
    extract_ingredients,
    format_dur_results,
    format_mutual_warnings,
    format_search_results,
    search_drugs,
    search_dur_for_ingredients,
)
from src.config import CLASSIFIER_MODEL, LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY


def _get_classifier() -> ChatOpenAI:
    """분류용 LLM (gpt-4.1-mini)."""
    return ChatOpenAI(
        model=CLASSIFIER_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY,
    )


def _get_generator() -> ChatOpenAI:
    """답변 생성용 LLM (gpt-4.1)."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )


def _classify(question: str) -> dict:
    """사용자 질문을 분류하여 category와 keyword를 반환합니다."""
    llm = _get_classifier()
    result = llm.invoke(CLASSIFIER_PROMPT.format_messages(question=question))
    try:
        parsed = json.loads(result.content.strip())
    except json.JSONDecodeError:
        # JSON 파싱 실패 시 기본값: 제품명 검색
        parsed = {"category": "product_name", "keyword": question}
    # Normalize keyword to a single string (LLM may return list or other types)
    raw_keyword = parsed.get("keyword", question)
    if isinstance(raw_keyword, list):
        try:
            keyword = ", ".join(str(k).strip() for k in raw_keyword if k is not None)
        except Exception:
            keyword = str(raw_keyword)
    else:
        keyword = str(raw_keyword).strip()

    return {
        "question": question,
        "category": parsed.get("category", "product_name"),
        "keyword": keyword if keyword else question,
    }


def _search(inputs: dict) -> dict:
    """분류 결과를 바탕으로 Supabase drugs 테이블을 검색하고 DUR 정보를 수집합니다."""
    # 1. drugs 테이블 검색 (기존) — 키워드가 콤마로 구분된 다중 키워드일 수 있으므로 처리
    raw_kw = inputs.get("keyword", "")
    # build list of keywords
    if isinstance(raw_kw, list):
        keywords = [str(k).strip() for k in raw_kw if k]
    else:
        # split on comma if multiple provided
        keywords = [k.strip() for k in str(raw_kw).split(",") if k.strip()]

    all_rows = []
    seen = set()
    for kw in keywords:
        rows = search_drugs(inputs["category"], kw)
        for r in rows:
            key = r.get("item_seq") or r.get("item_name")
            if key and key not in seen:
                all_rows.append(r)
                seen.add(key)

    rows = all_rows

    # 디버그: 검색 정보 출력
    print(f"[검색] 카테고리: {inputs['category']}, 키워드: {raw_kw}")
    print(f"[검색 결과] {len(rows)}건 발견")
    
    # 검색 결과 없을 때 경고
    if not rows:
        print(f"[주의] '{inputs['keyword']}'에 대한 검색 결과가 없습니다.")
        print(f"  → 데이터베이스에 해당 약품이 없을 수 있습니다.")
        print(f"  → 식약처 의약품안전나라(https://edicavi.mfds.go.kr)에서 직접 확인하세요.")
    
    context = format_search_results(rows)

    # 2. 검색된 약품에서 성분명 추출
    ingredients = extract_ingredients(rows)

    # 3. 각 성분에 대한 DUR 병용금지 정보 검색
    dur_data = search_dur_for_ingredients(ingredients)
    dur_context = format_dur_results(dur_data)

    # 4. 검색된 약품들 간 상호 병용금지 체크
    mutual_warnings = check_mutual_contraindication(ingredients)
    mutual_context = format_mutual_warnings(mutual_warnings)

    return {
        **inputs,
        "context": context,
        "source_drugs": rows,
        "ingredients": ingredients,
        "dur_data": dur_data,
        "dur_context": dur_context,
        "mutual_warnings": mutual_warnings,
        "mutual_context": mutual_context,
    }


def _generate(inputs: dict) -> dict:
    """검색 결과를 바탕으로 최종 답변을 생성합니다."""
    llm = _get_generator()
    prompt_value = ANSWER_PROMPT.format_messages(
        question=inputs["question"],
        category=inputs["category"],
        keyword=inputs["keyword"],
        context=inputs["context"],
    )
    answer = llm.invoke(prompt_value)
    return {
        "answer": answer.content,
        "source_drugs": inputs["source_drugs"],
        "category": inputs["category"],
        "keyword": inputs["keyword"],
    }


def build_rag_chain():
    """분류 → 검색 → 생성 3단계 체인을 구성합니다."""
    return (
        RunnableLambda(_classify)
        | RunnableLambda(_search)
        | RunnableLambda(_generate)
    )


def build_rag_chain_with_sources():
    """Streamlit용 체인 — answer + source_drugs를 반환합니다."""
    return build_rag_chain()


def _get_streaming_generator() -> ChatOpenAI:
    """스트리밍 지원 답변 생성용 LLM."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
        streaming=True,
    )


def prepare_context(question: str) -> dict:
    """분류 → 검색까지 수행하고 컨텍스트를 반환합니다."""
    classified = _classify(question)
    searched = _search(classified)

    # DUR 컨텍스트 조합 (일반 병용금지 + 상호 병용금지)
    combined_dur_context = searched["dur_context"]
    if searched["mutual_context"]:
        combined_dur_context += "\n\n" + searched["mutual_context"]

    prompt_messages = ANSWER_PROMPT.format_messages(
        question=searched["question"],
        category=searched["category"],
        keyword=searched["keyword"],
        context=searched["context"],
        dur_context=combined_dur_context,
    )
    return {
        **searched,
        "prompt_messages": prompt_messages,
    }


def stream_answer(prepared: dict) -> Generator[str, None, None]:
    """준비된 컨텍스트로 답변을 스트리밍합니다."""
    llm = _get_streaming_generator()
    for chunk in llm.stream(prepared["prompt_messages"]):
        if chunk.content:
            yield chunk.content
