from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.chain.prompts import RAG_PROMPT
from src.chain.retriever import get_retriever
from src.config import LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY


def format_docs(docs: list[Document]) -> str:
    """검색된 문서들을 하나의 컨텍스트 문자열로 포맷합니다.

    메타데이터에 full_text가 있으면 전체 약품 정보를 사용하고,
    없으면 page_content를 그대로 사용합니다.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        header = f"[참고 자료 {i}] {meta.get('item_name', '정보 없음')}"
        content = meta.get("full_text", doc.page_content)
        formatted.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(formatted)


def get_llm() -> ChatOpenAI:
    """ChatOpenAI LLM을 초기화합니다."""
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )


def build_rag_chain():
    """
    LCEL 기반 RAG 체인을 구성합니다.

    파이프라인:
    1. 사용자 질문 수신
    2. Pinecone에서 관련 문서 검색
    3. 문서를 컨텍스트 문자열로 포맷
    4. 질문 + 컨텍스트를 Few-shot 프롬프트에 전달
    5. LLM으로 답변 생성
    6. 문자열로 파싱
    """
    retriever = get_retriever()
    llm = get_llm()

    retrieval_chain = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
    )

    rag_chain = retrieval_chain | RAG_PROMPT | llm | StrOutputParser()

    return rag_chain


def build_rag_chain_with_sources():
    """
    답변과 출처 문서를 함께 반환하는 RAG 체인입니다.
    Streamlit 앱에서 출처를 표시하기 위해 사용합니다.
    """
    retriever = get_retriever()
    llm = get_llm()

    def retrieve_and_format(question: str) -> dict:
        docs = retriever.invoke(question)
        context = format_docs(docs)
        return {"context": context, "question": question, "source_docs": docs}

    def generate_answer(inputs: dict) -> dict:
        prompt_value = RAG_PROMPT.invoke(
            {
                "context": inputs["context"],
                "question": inputs["question"],
            }
        )
        answer = llm.invoke(prompt_value)
        return {
            "answer": answer.content,
            "source_docs": inputs["source_docs"],
        }

    chain = RunnableLambda(retrieve_and_format) | RunnableLambda(generate_answer)

    return chain
