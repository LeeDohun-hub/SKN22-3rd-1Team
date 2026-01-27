# 의약품 정보 Q&A RAG 시스템

식품의약품안전처(MFDS)의 **e약은요** 공공 API 데이터를 기반으로, 의약품 관련 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | Streamlit |
| LLM | GPT-5-nano (OpenAI) |
| Embedding | text-embedding-3-small (OpenAI) |
| Vector DB | Pinecone (AWS, us-east-1) |
| Orchestration | LangChain |
| Tracing | LangSmith |

## 프로젝트 구조

```
DJAeun/
├── app.py                         # Streamlit 웹 애플리케이션
├── requirements.txt               # Python 의존성
├── .env                           # 환경 변수 (API 키)
│
├── src/
│   ├── config.py                  # 설정값 (모델, API, Pinecone 등)
│   ├── chain/
│   │   ├── rag_chain.py           # RAG 체인 구성
│   │   ├── retriever.py           # 키워드 + 벡터 앙상블 리트리버
│   │   └── prompts.py             # Few-shot 프롬프트
│   ├── data/
│   │   ├── collector.py           # 공공 API 데이터 수집
│   │   ├── loader.py              # Document 생성 및 분할
│   │   └── preprocessor.py        # 텍스트 전처리
│   ├── vectorstore/
│   │   ├── pinecone_store.py      # Pinecone 인덱스 관리
│   │   ├── embeddings.py          # 임베딩 모델 초기화
│   │   └── ingest.py              # 수집 → 전처리 → 업로드 파이프라인
│   └── utils/
│       └── langsmith_config.py    # LangSmith 설정
│
├── scripts/
│   ├── collect_data.py            # 데이터 수집 스크립트
│   └── ingest_to_pinecone.py      # 전체 파이프라인 스크립트
│
├── data/
│   ├── raw/                       # 원본 데이터 (JSON, CSV)
│   └── processed/                 # 전처리된 데이터
│
└── tests/                         # 테스트
```

## 시스템 아키텍처

```
사용자 질문 (Streamlit)
       │
       ▼
  Ensemble Retriever ── 효능(efcyQesitm) 필드만 검색 대상
  ├── 키워드 리트리버 (70%) ── 제품명 + 효능 텍스트 키워드 매칭
  └── 벡터 리트리버 (30%) ── 효능 텍스트 임베딩 기반 Pinecone 시맨틱 검색
       │
       ▼
  metadata["full_text"]에서 전체 약품 정보 추출 (효능, 부작용, 사용법 등)
       │
       ▼
  Few-shot 프롬프트 + 전체 약품 컨텍스트
       │
       ▼
  GPT-5-nano → 답변 + 출처 생성
       │
       ▼
  채팅 UI에 답변 표시
```

> 검색은 효능 필드만 대상으로 수행되지만, LLM 답변 생성에는 해당 약품의 전체 정보(효능, 부작용, 사용법, 주의사항 등)가 사용됩니다.

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 설정합니다.

```
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
LANGSMITH_API_KEY=...
MC_DATA_API=...          # 공공데이터포털 API 키
```

### 3. 데이터 수집 및 Pinecone 업로드 (최초 1회)

```bash
# 전체 파이프라인 실행 (수집 → 전처리 → Pinecone 업로드)
python scripts/ingest_to_pinecone.py
```

데이터 수집만 별도로 실행하려면:

```bash
python scripts/collect_data.py
```

### 4. 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 로 접속합니다.

### 질문 예시

- "타이레놀의 효능은 무엇인가요?"
- "아스피린의 부작용은?"
- "활명수는 어떻게 복용하나요?"
- "겔포스와 함께 먹으면 안 되는 약은?"

## 데이터 정보

### 의약품개요정보(e약은요) API
- **서비스 ID**: DrbEasyDrugInfoService
- **오퍼레이션**: getDrbEasyDrugList
- **Base URL**: `http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList`
- **방식**: REST (GET), JSON/XML 지원

### 데이터 현황
| 항목 | 값 |
|------|-----|
| 전체 데이터 수 | **4,740건** |
| 컬럼 수 | 14개 |

### 주요 컬럼
| 컬럼명 | 설명 |
|--------|------|
| `entpName` | 업체명 |
| `itemName` | 제품명 |
| `itemSeq` | 품목기준코드 |
| `efcyQesitm` | 효능 |
| `useMethodQesitm` | 사용법 |
| `atpnWarnQesitm` | 주의사항 경고 |
| `atpnQesitm` | 주의사항 |
| `intrcQesitm` | 상호작용 |
| `seQesitm` | 부작용 |
| `depositMethodQesitm` | 보관법 |
| `openDe` | 공개일자 |
| `updateDe` | 수정일자 |
| `itemImage` | 낱알이미지 URL |
| `bizrno` | 사업자등록번호 |

## Data Sources

- https://www.data.go.kr/data/15075057/openapi.do
- https://data.seoul.go.kr/dataList/OA-20402/S/1/datasetView.do
