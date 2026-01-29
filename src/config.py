import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE_NAME = os.getenv("SUPABASE_TABLE_NAME", "documents")
SUPABASE_QUERY_NAME = os.getenv("SUPABASE_QUERY_NAME", "match_documents")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MC_DATA_API = os.getenv("MC_DATA_API")

# Drug API 1 Configuration (e약은요)
DRUG_API_BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
DRUG_API_NUM_OF_ROWS = 100

# Drug API 2 Configuration (허가정보)
DRUG_APPROVAL_API_BASE_URL = "http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# LLM Configuration
CLASSIFIER_MODEL = "gpt-4.1-nano"
LLM_MODEL = "gpt-4.1-mini"
LLM_TEMPERATURE = 0.0

# Search Configuration
SEARCH_LIMIT = 5

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = "drug-info-rag"
