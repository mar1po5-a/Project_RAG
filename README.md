## **⚙️** 설치 및 실행 방법

### 1. 저장소 복제

Sourcetree와 같은 GUI 프로그램을 사용 중이라면 해당 프로그램을 이용하여 저장소를 복제하세요.

```powershell
# 1. 원격 저장소를 로컬에 복제합니다.
git https://git-name
# 2. clone으로 생성된 프로젝트 폴더로 이동합니다.
cd git-name
```

### 2. 가상 환경 설정

RAG, LLM 시스템을 구동하기 위해 독립적인 파이썬 가상 환경을 설정합니다.

※ PyCharm으로 실행한다면 해당 과정을 단축 시킬 수 있습니다.

```powershell
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. 필요 패키지 설치

아래 requirements.txt 파일을 프로젝트 루트 디렉토리에 생성하고 내용을 붙여넣은 뒤, 다음 명령어로 한 번에 설치합니다.

※ PyCharm을 이용한다면 [ 파일-설정-Python 인터프리터 ] 기능을 통해 설치할 수 있습니다.

```
# requirements.txt

# Web Framework
fastapi
uvicorn[standard]

# RAG & LLM Core (LangChain)
langchain
langchain-community
langchain-google-genai
langchain-huggingface
langchain-text-splitters

# VectorDB
chromadb

# NLP & Utilities
konlpy
python-dotenv
pydantic-settings
filelock

# Deep Learning Framework (for HuggingFace Embeddings)
torch
```

```powershell
# 설치 명령어
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트 디렉토리에 .env 파일을 생성하고 아래와 같이 API 키를 입력합니다.

```python
GOOGLE_API_KEY="여기에_구글_API_키를_입력하세요"
HF_TOKEN="여기에_허깅페이스_토큰을_입력하세요"
```

### 5. 서버 실행

```python
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
