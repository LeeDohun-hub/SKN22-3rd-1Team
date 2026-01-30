"""
LangSmith 추적은 src/config.py의 환경 변수로 자동 설정됩니다:
  - LANGCHAIN_TRACING_V2 = "true"
  - LANGCHAIN_API_KEY = <langsmith key>
  - LANGCHAIN_PROJECT = "drug-info-rag"

모든 LangChain 체인 호출이 자동으로 추적됩니다.
"""

from langsmith import Client


def get_langsmith_client() -> Client:
    """LangSmith 클라이언트를 반환합니다."""
    return Client()


def create_evaluation_dataset(name: str, description: str):
    """LangSmith에 평가 데이터셋을 생성합니다."""
    client = get_langsmith_client()
    dataset = client.create_dataset(
        dataset_name=name,
        description=description,
    )
    return dataset
