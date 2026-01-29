import json
from typing import Generator

from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.chain.prompts import ANSWER_PROMPT, CLASSIFIER_PROMPT
from src.chain.retriever import format_search_results, search_drugs
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
    return {
        "question": question,
        "category": parsed.get("category", "product_name"),
        "keyword": parsed.get("keyword", question),
    }


def _search(inputs: dict) -> dict:
    """분류 결과를 바탕으로 Supabase drugs 테이블을 검색합니다."""
    rows = search_drugs(inputs["category"], inputs["keyword"])
    context = format_search_results(rows)
    return {
        **inputs,
        "context": context,
        "source_drugs": rows,
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
    prompt_messages = ANSWER_PROMPT.format_messages(
        question=searched["question"],
        category=searched["category"],
        keyword=searched["keyword"],
        context=searched["context"],
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
