import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MC_DATA_API = os.getenv("MC_DATA_API")

# Drug API Configuration
DRUG_API_BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
DRUG_API_NUM_OF_ROWS = 100

# Pinecone Configuration
PINECONE_INDEX_NAME = "drug-info-index"
PINECONE_DIMENSION = 1536
PINECONE_METRIC = "cosine"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM Configuration
LLM_MODEL = "gpt-5-nano"
LLM_TEMPERATURE = 0.0

# Chunking Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Retriever Configuration
SEARCH_K = 5

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = "drug-info-rag"
