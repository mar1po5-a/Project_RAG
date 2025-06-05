# config.py
"""
애플리케이션 전체에서 사용될 설정값들을 중앙에서 관리합니다.
Pydantic BaseSettings를 사용하여 .env 파일 및 환경 변수로부터
API 키, 모델 이름, 경로, 디버그 모드 등의 설정을 로드하고 유효성을 검사합니다.
다른 모듈들은 이 파일의 'settings' 객체를 통해 설정값에 접근합니다.

uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
"""
# 설정 모델(Setting 클래스)의 정의와 .env 파일 값을 설정 모델에 매핑 시키기 위한 import
from pydantic_settings import BaseSettings, SettingsConfigDict
# 모델 필드 값에 대한 사용자 정의 유효성 검사 또는 전/후처리 로직을 정의하기 위한 import
from pydantic import field_validator
# 파일과 디렉토리 경로를 나타내는 객체를 생성하고 관리해줌
from pathlib import Path
# 로거
import logging

# 로거 객체 생성
config_logger = logging.getLogger(__name__)

# 현재 파일이 위치한 디렉토리의 절대 경로를 알아내어 BASE_DIR이라는 변수에 저장
# __file__ : 현재 실행 중인 스크립트 파일의 경로를 문자열로 나타냄
# Path(__file__) : __file__ 문자열 경로를 Path 객체로 변환
# .resolve() : ../, ./ 와 같은 상대 경로 지시자를 처리하여 정규화된 절대 경로로 반환
# parent : 부모 디렉토리를 나타내는 새로운 Path 객체를 반환
BASE_DIR = Path(__file__).resolve().parent

# 설정 모델 정의
class Settings(BaseSettings):
    # .env 파일에서 읽어올 환경 변수들
    GOOGLE_API_KEY: str
    HF_TOKEN: str

    # 어플리케이션 기본 설정
    # 필요하다면 추후에 .env 파일에 분리
    APP_NAME: str = "RAG FastAPI Backend"
    DEBUG_MODE: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    EMBEDDING_MODEL_NAME: str = "nlpai-lab/KURE-v1"
    LLM_MODEL_NAME: str = "gemini-2.5-flash-preview-05-20"
    CHUNK_SIZE: int = 700 # 조정 가능 수치
    CHUNK_OVERLAP: int = 70 # 조정 가능 수치
    SEARCH_K: int = 3 # 검색해올 상위 문서 수
    PROMPT_TEMPLATE: str = """당신은 대한민국 정부에서 주관하는 청년 정책의 정확한 정보를 제공하는 AI 어시스턴트입니다.
주어진 문맥(context) 정보를 바탕으로 질문에 답변해주세요. 문맥에서 답을 찾을 수 없다면, "제공된 정보만으로는 답변하기 어렵습니다."라고 솔직하게 말해주세요.

대화 기록:
{chat_history}

문맥:
{context}

질문:
{question}

답변:
"""
    DATA_PATH: Path = BASE_DIR / "my_data_directory"
    VECTORSTORE_PATH: Path = BASE_DIR / "chroma_db_rag_kure_store"
    CHAT_HISTORY_FILE: Path = BASE_DIR / "chat_history.json"
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost",
    ]

    MAX_CHAT_HISTORY_FILE_AGE_DAYS: int = 7 # 7일
    # MAX_CHAT_HISTORY_FILE_AGE_MINUTES: int = 3 # 테스트 코드, 3분

    # GOOGLE_API_KEY와 HF_TOKEN 필드에 대한 유효성 검사 및 전처리
    # .env 파일 내부에 저장된 GOOGLE_API_KEY가 모종의 이유로 문자열 앞 뒤에 ''가 포함된 상태로 할당됨
    # 이를 해결하기 위해 해당 코드로 전처리를 진행
    # mode="befor" : Pydantic의 표준 유효성 검사가 수행되기 전에 실행
    @field_validator("GOOGLE_API_KEY", "HF_TOKEN", mode="before")
    @classmethod # 인스턴스 메서드가 아닌 클래스 메서드임을 선언
    # cls : 해당 메서드가 속한 클래스 자체를 참조
    # value: any : 모든 형태의 원시 입력값 허용 (어차피 Pydantic에서 유효성 검사 및 타입 변환을 실행해줌)
    # -> str : 반환 타입 힌트, 해당 함수가 문자열 타입의 값을 반환할 것임을 명시
    def strip_quotes_from_env_vars(cls, value: any) -> str:
        if isinstance(value, str):
            # 앞뒤 공백 제거 -> 작은따옴표 제거 -> 큰따옴표 제거
            cleaned_value = value.strip().strip("'").strip('"')
            return cleaned_value
        # value가 문자열이 아니고 None이면, 빈 문자열("")을 반환
        elif value is None:
            return "" # raise ValueError()로도 처리할 수 있음
        else: # 전달된 value가 문자열도 None도 아닐 때
            raise TypeError(f"API Key는 str 또는 None이어야만 합니다. 현재 전달된 value의 타입 : {type(value)}")

    # 위에서 정의된 Settings 클래스가 해당 모듈에서 어떻게 동작할지에 대한 설정
    model_config = SettingsConfigDict(
        env_file= BASE_DIR / ".env",
        env_file_encoding='utf-8',
        # Settings 클래스에 정의되지 않은 필드가 .env 파일이나 환경 변수에 있을 경우의 처리:
        # ignore : 무시하다
        extra='ignore'
    )

# Settings 클래스의 인스턴스 생성
# 이 시점에 GOOGLE_API_KEY와 HF_TOKEN 값의 전처리를 위해 field_validator가 실행
settings = Settings()

def create_initial_directories():
    # 원본 데이터의 상위 디렉토리가 존재하지 않으면
    if not settings.DATA_PATH.exists():
        config_logger.info(f"데이터 디렉토리 '{settings.DATA_PATH}'가 없어 새로 생성합니다. 데이터를 채워주세요.")
        try:
            # settings.DATA_PATH 디렉토리 생성 시도
            # parents=True: 필요한 모든 상위 디렉토리도 함께 생성
            # exist_ok=True: 디렉토리가 이미 존재하더라도 오류를 발생시키지 않음
            # (예: 여러 요청이 동시에 디렉토리 생성을 시도하는 경쟁 상태 방지)
            settings.DATA_PATH.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            config_logger.error(f"데이터 디렉토리 '{settings.DATA_PATH}' 생성 중 오류: {e}")
    # if문 실행 시점에 이미 DATA_PATH 디렉토리가 존재했던 경우
    else:
        config_logger.info(f"데이터 디렉토리 '{settings.DATA_PATH}'가 이미 존재합니다.")

# 디버깅 로그 (settings 객체 생성 후, 전처리 된 값 확인)
# settings 객체에 GOOGLE_API_KEY 라는 속성이 실제로 존재하고, 그 속성의 값의 유효하다면
if hasattr(settings, 'GOOGLE_API_KEY') and settings.GOOGLE_API_KEY:
    config_logger.info(f"[CONFIG_PY_FINAL_DEBUG] settings.GOOGLE_API_KEY (유효성 검사 후): '{settings.GOOGLE_API_KEY[:5]}...' (길이: {len(settings.GOOGLE_API_KEY)})")
else:
    config_logger.error("[CONFIG_PY_FINAL_DEBUG] settings.GOOGLE_API_KEY가 로드되지 않았거나 비어있습니다. (유효성 검사 후)")

# settings 객체에 HF_TOKEN 이라는 속성이 실제로 존재하고, 그 속성의 값의 유효하다면
if hasattr(settings, 'HF_TOKEN') and settings.HF_TOKEN:
    config_logger.info(f"[CONFIG_PY_FINAL_DEBUG] settings.HF_TOKEN (유효성 검사 후): '{settings.HF_TOKEN[:5]}...' (길이: {len(settings.HF_TOKEN)})")
else:
    config_logger.warning("[CONFIG_PY_FINAL_DEBUG] settings.HF_TOKEN이 로드되지 않았거나 비어있습니다. (유효성 검사 후)")