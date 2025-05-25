# Project_RAG
(Project)청년 정책 지원 도우미 : Google Gemini LLM을 이용한 RAG 시스템
RAG 파이프라인을 구성하고 모듈화를 진행했습니다.

python 포트 번호는 8000번, \n
CORS로 연결할 react 포트 번호는 3000번으로 설정되어있습니다.

(venv) 터미널에 uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
react 터미널에서 npm start 를 입력하세요.
_____________

- main.py
  FastAPI 웹 애플리케이션의 진입점(entry point)입니다.
  API 엔드포인트를 정의하고, 애플리케이션 시작 시 RAG 파이프라인을 초기화하며,
  CORS 설정 등 웹 서비스 관련 기본 구성을 담당합니다.
  사용자의 질문을 받아 RAG 파이프라인을 통해 답변을 생성하고 반환합니다.
  
- config.py
  애플리케이션 전체에서 사용될 설정값들을 중앙에서 관리합니다.
  Pydantic BaseSettings를 사용하여 .env 파일 및 환경 변수로부터
  API 키, 모델 이름, 경로, 디버그 모드 등의 설정을 로드하고 유효성을 검사합니다.
  다른 모듈들은 이 파일의 'settings' 객체를 통해 설정값에 접근합니다.
  
- rag_main_runner.py
  RAG 파이프라인의 전체적인 구성 및 실행 로직을 담당합니다.
  RAGPipeline 클래스는 데이터 로드, 텍스트 처리, 임베딩, 벡터 저장소 관리,
  LLM 연동, 체인 구성 등 RAG 시스템의 핵심 단계를 초기화하고,
  사용자 질문에 대한 답변 생성 기능을 제공합니다.
  
- rag_utils.py
  RAG 파이프라인을 구성하는 각 개별 단계를 위한 유틸리티 함수들을 제공합니다.
  텍스트 분할기 생성, 문서 분할, 임베딩 모델 로드, LLM 로드,
  벡터 저장소 생성/로드, 리트리버 생성, RAG 체인 구성 등의 기능을 포함합니다.
  rag_main_runner.py의 RAGPipeline 클래스에서 이 함수들을 호출하여 사용합니다.
_____________

- rag.py
  코랩에서 작성되었던 초기 모듈입니다.

- simple_run.py
  RAG 파이프라인의 핵심 기능을 테스트하는 모듈입니다. 터미널에서 실행하세요.  
  주요 역할:
  - `RAGPipeline` 객체를 초기화합니다.
  - 사용자로부터 질문을 입력받아, 초기화된 파이프라인을 통해 답변을 생성하고 출력합니다.
  - FastAPI 웹 서버 없이 RAG 시스템의 기본적인 질문-답변 기능을 빠르게 검증하는 데 사용됩니다.
  정상적으로 실행되지만, ChromaDB-LangChain Version Warning이 발생할 수 있습니다.

- my_data_directory/your_data_file
  샘플 데이터 디렉토리입니다. 해당 파일의 모든 데이터는 픽션입니다.
  추후에는 실제적인 데이터 파일이 필요합니다.
_____________

.env 파일을 생성하고 GOOGLE_API_KEY와 HF_TOKEN을 설정하세요.
GOOGLE_API_KEY=your_api_key
HF_TOKEN=your_token
_____________

가상환경 설정이 필요합니다.
쉬운 사용을 위해 파이참 IDE에서 실행하세요.
- fastapi
- uvicorn
- pydantic
- pydantic-settings
- python-dotenv
- langchain
- langchain-core
- langchain-community
- langchain-google-genai
- google-generativeai
- langchain-huggingface
- sentence-transformers
- torch
- konlpy
- langchain-chroma
- chromadb
- requests
- tiktoken
