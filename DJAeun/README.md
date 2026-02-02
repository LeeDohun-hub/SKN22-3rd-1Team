<div align="center">

# 💊 OpenFDA 의약품 정보 Q&A (DJAeun)

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-1C3C3C?style=for-the-badge)](https://langchain.com)
[![OpenFDA](https://img.shields.io/badge/OpenFDA-003366?style=for-the-badge&logo=fda&logoColor=white)](https://open.fda.gov)

<br/>

**미국 FDA 공공데이터(OpenFDA) 기반 실시간 의약품 정보 챗봇**

</div>

---

> [!CAUTION]
> **⚠️ 의료 면책 조항 (Medical Disclaimer)**
> 
> 본 시스템은 **OpenFDA 데이터**를 기반으로 정보를 제공하며, **의학적 진단이나 처방을 대신할 수 없습니다.**
> 
> - 🔴 제공된 정보는 실시간 API 호출 결과이나, AI 가공 과정에서 부정확한 내용이 포함될 수 있습니다.
> - 🔴 **모든 건강 관련 결정은 반드시 의사 또는 약사와 상담 후 진행하세요.**
> - 🔴 본 시스템 사용으로 인한 어떠한 피해에 대해서도 책임지지 않습니다.

---

## 📋 목차

- [기술 스택](#-기술-스택)
- [프로젝트 구조](#-프로젝트-구조)
- [시스템 아키텍처](#-시스템-아키텍처)
- [실행 방법](#-실행-방법)
- [질문 예시](#-질문-예시)
- [주요 설정](#-주요-설정)

---

## 🛠 기술 스택

| 분류 | 기술 | 설명 |
|:---:|:---:|:---|
| 🖥️ **UI** | Streamlit | Chat Interface 제공 |
| 🤖 **Classifier** | GPT-4.1-nano | 질문 의도 분류 (Router) |
| ✍️ **Generator** | GPT-4.1-mini | 최종 답변 생성 |
| ☁️ **Data Source** | OpenFDA API | 실시간 의약품 라벨 정보 (Labeling) |
| 🔗 **Orchestration** | LangChain | RAG 파이프라인 구성 |

---

## 📁 프로젝트 구조

```
DJAeun/
├── 🚀 app.py                    # Streamlit 메인 앱
├── 📋 requirements.txt          # 패키지 의존성
└── 📂 src/
    ├── ⚙️ config.py             # 환경 설정 (API Key 등)
    ├── 📡 api/
    │   ├── openfda_client.py    # OpenFDA API 호출 클라이언트
    │   └── formatter.py         # JSON 응답 데이터 포매팅
    └── ⛓️ chain/
        ├── rag_chain.py         # RAG 파이프라인 (분류 -> 검색 -> 생성)
        └── prompts.py           # LLM 프롬프트 템플릿
```

---

## 🔄 시스템 아키텍처

사용자의 질문을 분석하여 적절한 API 엔드포인트를 호출하고, 결과를 종합하여 답변을 생성합니다.

```mermaid
graph TD
    %% Nodes
    User([👤 사용자])
    App["🖥️ Streamlit App"]
    
    subgraph Core_Logic ["Core Logic"]
        Classifier["🤖 의도 분류 (Nano)"]
        Router{"분류 결과"}
        Generator["✍️ 답변 생성 (Mini)"]
    end
    
    subgraph Data_Source ["External API"]
        Client["📡 OpenFDA Client"]
        API[("☁️ api.fda.gov")]
    end

    %% Flow
    User -->|"질문 입력"| App
    App -->|"Context 준비"| Classifier
    Classifier --> Router
    
    Router -->|"브랜드명"| Client
    Router -->|"성분명"| Client
    Router -->|"효능/증상"| Client
    
    Client -->|"HTTP GET"| API
    API -->|"JSON 응답"| Client
    Client -->|"Formatted Text"| Generator
    
    Generator -->|"최종 답변"| App
    App -->|"화면 출력"| User

    %% Styles
    style App fill:#f9f,stroke:#333
    style Classifier fill:#ff9,stroke:#333
    style Generator fill:#9ff,stroke:#333
    style API fill:#eee,stroke:#999
```

---

## 🚀 실행 방법

### 1️⃣ 필수 패키지 설치

```bash
cd DJAeun
pip install -r requirements.txt
```

### 2️⃣ 환경 변수 설정

`.env` 파일에 아래 키를 설정해야 합니다.

```env
# OpenAI
OPENAI_API_KEY=sk-...

# OpenFDA (Optional, but recommended for higher limits)
OPENFDA_API=...

# LangSmith (Optional)
LANGSMITH_API_KEY=...
```

### 3️⃣ 애플리케이션 실행

```bash
streamlit run app.py
```

---

## 💬 질문 예시

| 카테고리 | 질문 예시 | 비고 |
|:---:|:---|:---|
| **🏷️ 브랜드명** | "Tylenol의 효능은 무엇인가요?" | `openfda.brand_name` 검색 |
| **🧪 성분명** | "Ibuprofen 복용 시 주의사항 알려줘" | `openfda.generic_name` 검색 |
| **🩹 증상/효능** | "두통(Headache)에 좋은 약 있어?" | `indications_and_usage` 검색 |

> [!TIP]
> OpenFDA 데이터 특성상 **영문 약품명**이나 **영문 증상**으로 재차 검색하면 더 정확한 결과를 얻을 수 있습니다.

---

## ⚙️ 주요 설정

`src/config.py`에서 변경 가능합니다.

- **`SEARCH_LIMIT`**: 기본 **5개**. 한 번에 가져올 API 결과 수입니다.
- **`LLM_TEMPERATURE`**: 기본 **0.0**. 사실 기반 응답을 위해 0으로 설정되어 있습니다.

---

<div align="center">
  
**SKN22-3rd-1Team / DJAeun**

</div>
