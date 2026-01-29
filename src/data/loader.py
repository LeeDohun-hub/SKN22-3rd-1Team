from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def create_documents(processed_items: list[dict]) -> list[Document]:
    """전처리된 데이터를 LangChain Document 객체로 변환합니다."""
    documents = []
    for item in processed_items:
        doc = Document(
            page_content=item["text"],
            metadata=item["metadata"],
        )
        documents.append(doc)
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """
    청크 크기를 초과하는 문서를 분할합니다.
    대부분의 약품 문서(~300-800자)는 분할되지 않습니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ".", " ", ""],
        length_function=len,
    )
    return text_splitter.split_documents(documents)
