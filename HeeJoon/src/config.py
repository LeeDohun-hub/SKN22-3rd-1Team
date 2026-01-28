import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
MC_DATA_API = os.getenv("MC_DATA_API")

# Supabase Configuration
SUPABASE_TABLE_NAME = "drug_documents"
SUPABASE_QUERY_NAME = "match_documents"

# Drug API 1 Configuration (e약은요)
DRUG_API_BASE_URL = "http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList"
DRUG_API_NUM_OF_ROWS = 100

# Drug API 2 Configuration (허가정보)
DRUG_APPROVAL_API_BASE_URL = "http://apis.data.go.kr/1471000/DrugPrdtPrmsnInfoService07/getDrugPrdtPrmsnDtlInq06"

# LLM Configuration
CLASSIFIER_MODEL = "gpt-4.1-mini"
LLM_MODEL = "gpt-4.1"
LLM_TEMPERATURE = 0.0
EMBEDDING_MODEL = "text-embedding-3-small"

# Search Configuration
SEARCH_LIMIT = 5

# Document Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# LangSmith Tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"] = "drug-info-rag"
