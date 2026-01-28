# 의약품 정보 Q&A RAG 시스템

식품의약품안전처의 **e약은요** 및 **의약품 허가정보** 공공 API 데이터를 기반으로, 의약품 관련 질문에 답변하는 RAG(Retrieval-Augmented Generation) 챗봇 시스템입니다.

## 기술 스택

| 분류 | 기술 |
|------|------|
| UI | Streamlit |
| LLM | GPT-5-nano (OpenAI) |
| Embedding | text-embedding-3-small (OpenAI) |
| Vector DB | Supabase (pgvector) |
| Orchestration | LangChain (LCEL) |
| Tracing | LangSmith |

## 프로젝트 구조

```
DJAeun/
├── app.py                         # Streamlit 웹 애플리케이션
├── requirements.txt               # Python 의존성
├── .env                           # 환경 변수 (API 키)
│
├── src/
│   ├── config.py                  # 설정값 (모델, API, Supabase 등)
│   ├── chain/
│   │   ├── rag_chain.py           # LCEL RAG 체인 구성
│   │   ├── retriever.py           # 키워드 + 벡터 앙상블 리트리버
│   │   └── prompts.py             # Few-shot 프롬프트
│   ├── data/
│   │   ├── collector.py           # 공공 API 데이터 수집
│   │   ├── loader.py              # Document 생성 및 청크 분할
│   │   └── preprocessor.py        # 텍스트 전처리 및 API 병합
│   ├── vectorstore/
│   │   ├── supabase_store.py      # Supabase 벡터 저장소 관리
│   │   ├── embeddings.py          # 임베딩 모델 초기화
│   │   └── ingest.py              # 수집 → 전처리 → 업로드 파이프라인
│   └── utils/
│       └── langsmith_config.py    # LangSmith 설정
│
├── scripts/
│   ├── collect_data.py            # 데이터 수집 스크립트
│   └── ingest_to_supabase.py      # 전체 파이프라인 스크립트
│
└── data/
    ├── raw/                       # 원본 데이터 (JSON)
    └── processed/                 # 전처리된 데이터
```

## RAG 체인 구조

```
사용자 질문 (Streamlit)
       │
       ▼
┌─ Ensemble Retriever ──────────────────────────────┐
│                                                    │
│  ┌─ 키워드 리트리버 (가중치 70%) ───────────────┐  │
│  │  DrugNameRetriever                           │  │
│  │  - 한국어 조사 제거 후 키워드 추출           │  │
│  │  - 제품명 매칭: +10점 / 효능 텍스트: +1점   │  │
│  │  - 점수 기반 상위 k개 반환                   │  │
│  └──────────────────────────────────────────────┘  │
│                                                    │
│  ┌─ 벡터 리트리버 (가중치 30%) ───────────────┐    │
│  │  Supabase pgvector 시맨틱 검색              │    │
│  │  - text-embedding-3-small 임베딩            │    │
│  │  - 효능 텍스트 대상 유사도 검색             │    │
│  └─────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────┘
       │
       ▼
  metadata["full_text"]에서 전체 약품 정보 추출
  (효능 + 사용법 + 주의사항 + 부작용 + 상호작용 + 허가정보)
       │
       ▼
  Few-shot 프롬프트 (시스템 메시지 + 2개 예시 + 질문/컨텍스트)
       │
       ▼
  GPT-5-nano (temperature=0.0) → 답변 + 출처 생성
       │
       ▼
  Streamlit 채팅 UI (답변 표시 + 참고 자료 접기/펼치기)
```

> **핵심 설계**: 검색은 효능(efcyQesitm) 필드만 대상으로 수행하되, LLM 답변 생성에는 해당 약품의 전체 정보(효능, 부작용, 사용법, 주의사항, 허가정보 등)를 사용합니다.

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 아래 키를 설정합니다.

```
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://...
SUPABASE_KEY=...
LANGSMITH_API_KEY=...
MC_DATA_API=...          # 공공데이터포털 API 키
```

### 3. 데이터 수집 및 Supabase 업로드 (최초 1회)

```bash
# 전체 파이프라인 실행 (수집 → 전처리 → Supabase 업로드)
python scripts/ingest_to_supabase.py
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
- "타이레놀의 주성분은?"
- "아스피린의 허가일자는?"
- "아스피린의 부작용은?"
- "겔포스와 함께 먹으면 안 되는 약은?"

## 데이터 소스

### API 1: 의약품개요정보(e약은요)

| 항목 | 값 |
|------|-----|
| 서비스 ID | DrbEasyDrugInfoService |
| 데이터 수 | ~4,740건 |
| 주요 필드 | 효능, 사용법, 주의사항, 부작용, 상호작용, 보관법 |

### API 2: 의약품 허가정보

| 항목 | 값 |
|------|-----|
| 서비스 ID | DrugPrdtPrmsnInfoService07 |
| 데이터 수 | ~70,000건 (e약은요 기준 필터링 후 병합) |
| 주요 필드 | 주성분, 성상, 허가일자, 전문/일반 구분, 저장방법 |

두 API의 데이터는 `itemSeq`(품목기준코드) 기준으로 LEFT JOIN 병합됩니다.

## 주요 설정값

| 항목 | 값 |
|------|-----|
| LLM 모델 | gpt-5-nano |
| 임베딩 모델 | text-embedding-3-small |
| 청크 크기 | 1,500자 (overlap 200) |
| 검색 결과 수 (k) | 5 |
| 키워드/벡터 가중치 | 70% / 30% |

## 데이터베이스 스키마 (Supabase)

### drugs 테이블

의약품 기본 정보를 저장하는 테이블입니다. API 1(e약은요)과 API 2(허가정보)에서 수집한 데이터를 병합하여 저장합니다.

| 컬럼명 | 타입 | 설명 | 출처 |
|--------|------|------|------|
| `id` | BIGSERIAL | 자동 증가 기본키 | - |
| `item_seq` | TEXT (UNIQUE) | 품목기준코드 | API 1 |
| `item_name` | TEXT | 제품명 | API 1 |
| `entp_name` | TEXT | 업체명 | API 1 |
| `efcy_qesitm` | TEXT | 효능·효과 | API 1 |
| `use_method_qesitm` | TEXT | 사용법 | API 1 |
| `atpn_warn_qesitm` | TEXT | 주의사항 (경고) | API 1 |
| `atpn_qesitm` | TEXT | 주의사항 | API 1 |
| `intrc_qesitm` | TEXT | 상호작용 | API 1 |
| `se_qesitm` | TEXT | 부작용 | API 1 |
| `deposit_method_qesitm` | TEXT | 보관법 | API 1 |
| `open_de` | TEXT | 공개일자 | API 1 |
| `update_de` | TEXT | 수정일자 | API 1 |
| `item_image` | TEXT | 제품 이미지 URL | API 1 |
| `bizrno` | TEXT | 사업자등록번호 | API 1 |
| `item_eng_name` | TEXT | 제품명 (영문) | API 2 |
| `chart` | TEXT | 성상 (외형 설명) | API 2 |
| `main_item_ingr` | TEXT | 주성분 | API 2 |
| `ingr_name` | TEXT | 첨가제 | API 2 |
| `pack_unit` | TEXT | 포장단위 | API 2 |
| `storage_method` | TEXT | 저장방법 | API 2 |
| `valid_term` | TEXT | 유효기간 | API 2 |
| `spclty_pblc` | TEXT | 전문/일반 구분 | API 2 |
| `prduct_prmisn_no` | TEXT | 품목허가번호 | API 2 |
| `item_permit_date` | TEXT | 허가일자 | API 2 |
| `permit_kind_name` | TEXT | 허가종류 | API 2 |
| `cnsgn_manuf` | TEXT | 위탁제조업체 | API 2 |
| `rare_drug_yn` | TEXT | 희귀의약품 여부 | API 2 |
| `cancel_date` | TEXT | 취소일자 | API 2 |
| `cancel_name` | TEXT | 취소사유 | API 2 |
| `created_at` | TIMESTAMPTZ | 생성일시 | - |
| `updated_at` | TIMESTAMPTZ | 수정일시 | - |

### documents 테이블

LangChain 벡터 임베딩을 저장하는 테이블입니다. pgvector 확장을 사용합니다.

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `id` | UUID | 문서 고유 ID |
| `content` | TEXT | 문서 텍스트 (효능 정보) |
| `metadata` | JSONB | 메타데이터 (item_seq, item_name, entp_name, full_text 등) |
| `embedding` | VECTOR(1536) | 벡터 임베딩 (text-embedding-3-small) |

### 검색 함수

```sql
match_documents(query_embedding, match_count, filter)
```

벡터 유사도 검색을 위한 RPC 함수입니다. 코사인 유사도 기반으로 가장 유사한 문서를 반환합니다.
