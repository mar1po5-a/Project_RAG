# main.py
"""
FastAPI 웹 애플리케이션의 진입점(entry point)입니다.
API 엔드포인트를 정의하고, 애플리케이션 시작 시 RAG 파이프라인을 초기화하며,
CORS 설정 등 웹 서비스 관련 기본 구성을 담당합니다.
사용자의 질문을 받아 RAG 파이프라인을 통해 답변을 생성하고 반환합니다.

uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
"""
# FastAPI - fastapi의 핵심 클래스, 객체 생성을 위한 import
# HTTPException - 클라이언트에 오류 메시지를 JSON 형식으로 반환
from fastapi import FastAPI, HTTPException
# react와의 연결을 위해 import
from fastapi.middleware.cors import CORSMiddleware
# 데이터 타입 유효성 검사와 응답 모델 정의를 위한 import
# 제미나이 LLM이 text 데이터를 생성하더라도 클라이언트가 이해할 수 있는 형태로(JSON 등) 감싸주고,
# 해당 타입으로 응답할 것임을 fast api에 알리는 역할
from pydantic import BaseModel
# 로그
import logging
# 유저별 대화 기록 저장 및
from chat_memory import UserChatMemory

# config.py에서 설정 가져오기 - 내 파일
from config import settings, create_initial_directories


# RAG 파이프라인 클래스 가져오기 - 내 파일
from rag_main_runner import RAGPipeline

# 로거 설정(추후에 로거 단독 모듈로 분리 예정)
# 이해되지 않는 부분이 많아 모듈화 진행하면서 다시 정리하는 게 좋을 듯
logger = logging.getLogger(__name__) # __name__ : 현재 실행중인 파이썬 모듈의 이름을 나타내는 내장 변수
if not logger.handlers: # 핸들러가 이미 설정되지 않은 경우에만 기본 설정
    logging.basicConfig(level=logging.INFO if not settings.DEBUG_MODE else logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# API Key Value Error를 위한 디버깅
logger.info(f"--- settings.GOOGLE_API_KEY in main.py: {'LOADED' if settings.GOOGLE_API_KEY else 'NOT LOADED'} ---")

# FastAPI 어플리케이션 객체 생성
# 객체(인스턴스) 생성: 클래스라는 설계도를 바탕으로 실제 메모리에 할당된 실체를 만드는 일반적인 과정.
# 애플리케이션 인스턴스 생성: 특정 프레임워크(여기서는 FastAPI)에서,
# 애플리케이션 전체를 대표하고 관리하는 핵심적인 객체(인스턴스)를 생성하는 것을 의미.
# 해당 객체가 생성되지 않으면 어플리케이션이 실행될 수 없음
app = FastAPI(title=settings.APP_NAME, debug=settings.DEBUG_MODE)

# RAG 파이프라인 인스턴스를 담을 변수 선언.
# 초기값은 None이며, 애플리케이션 시작 시 (lifespan 또는 startup_event 내에서)
# 실제 RAGPipeline 객체가 생성되어 할당될 예정.
rag_pipeline_instance: RAGPipeline | None = None

# --- FastAPI 시작 시 실행될 이벤트 핸들러 ---
# FastAPI 애플리케이션의 시작 시점과 종료 시점에 특정 코드를 실행할 수 있도록 해주는 메커니즘
# lifespan 방식이 더 현대적이기에 파이참 내부에서도 on_event 대신 사용하길 권장하는 warning이 출력됨
@app.on_event("startup")
async def startup_event():
    logger.info("--- startup_event 시작 ---")  # startup_event 시작 확인
    global rag_pipeline_instance # 전역 변수를 사용하겠다고 선언
    logger.info(f"애플리케이션 '{settings.APP_NAME}' 시작...")
    logger.info(f"디버그 모드: {settings.DEBUG_MODE}")
    logger.info(f"데이터 경로: {settings.DATA_PATH}")
    logger.info(f"벡터 저장소 경로: {settings.VECTORSTORE_PATH}")

    try:
        # 초기 디렉토리 생성 (DATA_PATH 등)
        # config.py에 정의된 함수
        # 디렉토리 존재 여부를 확인하고, 없다면 새로 생성해주는 역할
        create_initial_directories()

        logger.info("RAG 파이프라인 초기화를 시작합니다...")

        # RAGPipeline 클래스의 인스턴스를 생성해서 rag_pipeline_instance 변수에 할당
        # 이는 모듈 레벨이라고 지칭할 수 있는데, 자바에서 static 변수와 유사한 개념
        rag_pipeline_instance = RAGPipeline(
            data_path=str(settings.DATA_PATH), # 원본 데이터 디렉토리 경로 전달
            vectorstore_path=str(settings.VECTORSTORE_PATH), # 벡터DB 경로 전달
            # force_create_db=True : 기존 벡터DB가 존재하더라도 항상 새로운 벡터DB를 생성
            # 개발 편의상 False로 두거나, 필요시 True로 변경하여 테스트
            force_create_db=False
        )
        logger.info("RAG 파이프라인 초기화 성공.")
    except Exception as e:
        # exc_info=True : 자바에서의 e.printStackTrace()와 유사한 역할
        logger.error(f"RAG 파이프라인 초기화 중 심각한 오류 발생: {e}", exc_info=True)
        # rag_pipeline_instance는 초기값인 None으로 유지되어 API 요청이 들어오면 오류 반환


# --- CORS 미들웨어 설정 ---
# 실질적인 react와의 연결 설정은 해당 코드에서 이루어짐
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS, # config.py에서 가져온 허용 출처
    allow_credentials=True,
    allow_methods=["*"], # 모든 HTTP 메서드 허용
    allow_headers=["*"], # 모든 HTTP 헤더 허용
)

class SearchRequest(BaseModel):
    member_id: str
    question: str

# --- 응답 모델 정의 ---
class SearchResponse(BaseModel):
    # 앞서 언급했듯이 LLM은 text 형태로 답변을 전달함.
    # 이 API의 응답에는 'answer'라는 key가 존재할 것이고 해당 키 값은 'str'이어야만 한다는 의미
    answer: str

# --- API 엔드포인트 ---
@app.post("/ask", response_model=SearchResponse)
# 비동기 함수의 정의
# Body(..., media_type="") : ...은 기본값이 없고 필수 항목이라는 의미를 가짐
async def search_rag_system(request_data: SearchRequest):
    member_id = request_data.member_id
    question_text = request_data.question

    logger.info(f"'/search' 엔드포인트 수신 - 사용자 ID: {member_id}, 질문: {question_text}")

    # startup_event에서 rag_pipeline_instance 생성에 실패했으면 (기본값이 None임)
    if rag_pipeline_instance is None:
        logger.error("RAG 파이프라인이 초기화되지 않아 요청을 처리할 수 없습니다.")
        # 503 Service Unavailable (서비스를 사용할 수 없음)
        raise HTTPException(status_code=503, detail="RAG 시스템이 현재 사용 불가능합니다. 잠시 후 다시 시도해주세요.")

    if not member_id or not member_id.strip():
        logger.warning("member_id가 비어있습니다.")
        raise HTTPException(status_code=400, detail="member_id가 필요합니다.")

    # 사용자 질문이 넘어오지 않았거나 빈 문자열을 전달 받았으면
    if not question_text or not question_text.strip():
        logger.warning("비어있는 질문이 수신되었습니다.")
        # 400 Bad Request
        raise HTTPException(status_code=400, detail="질문 내용이 필요합니다.")

    try:
    #     # 답변 생성 시도
    #     answer = rag_pipeline_instance.query(question_text)
    #     # 성공적으로 답변을 받으면 SearchResponse 모델로 감싸서 반환
    #     return SearchResponse(answer=answer[0])
    # # RAGPipeline.query 실행 중 예외가 발생하여 이 블록으로 넘어온 경우.
    # # query 메소드 내부에 자체적인 try-except가 있지만,
    # # 모든 예외를 처리하여 문자열로 반환하지 않고, 일부 예외는 그대로 발생시킬 수 있음.
    # # 또는 query 메소드 내부에서 예기치 못한 오류가 발생한 경우일 수 있음.
    # except Exception as e:
    #     logger.error(f"질문 처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
    #     raise HTTPException(status_code=500, detail="답변 생성 중 서버 내부 오류가 발생했습니다.")

        chat_memory_manager = UserChatMemory(
            member_id=member_id,
            history_file_path=str(settings.CHAT_HISTORY_FILE)
        )

        chat_history = chat_memory_manager.get_chat_messages()

        answer_str = rag_pipeline_instance.query(
            question_text,
            history=chat_history
        )

        chat_memory_manager.add_question(question_text)
        chat_memory_manager.add_answer(answer_str)

        logger.info(f"사용자 ID '{member_id}'에게 답변 생성 완료.")
        return SearchResponse(answer=answer_str)

    except Exception as e:
        logger.error(f"질문 처리 중 예기치 않은 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="답변 생성 중 서버 내부 오류가 발생했습니다.")

class ClearHistoryRequest(BaseModel):
    member_id: str

@app.post("/clear_history")
async def clear_user_history(request_data: ClearHistoryRequest):
    member_id = request_data.member_id
    logger.info(f"'/clear_history' 엔드포인트 수신 - 사용자 ID: {member_id}")

    if not member_id or not member_id.strip():
        logger.warning("member_id가 비어있습니다. (clear_history)")
        raise HTTPException(status_code=400, detail="member_id가 필요합니다.")

    try:
        chat_memory_manager = UserChatMemory(
            member_id=member_id,
            history_file_path=str(settings.CHAT_HISTORY_FILE)
        )
        chat_memory_manager.clear_history()
        logger.info(f"사용자 '{member_id}'의 대화 기록이 성공적으로 삭제되었습니다.")
        return {"message": f"사용자 '{member_id}'의 대화 기록이 삭제되었습니다."}
    except Exception as e:
        logger.error(f"사용자 '{member_id}'의 대화 기록 삭제 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="대화 기록 삭제 중 오류가 발생했습니다.")

@app.get("/")
# 비동기 함수의 정의
async def root():
    logger.info("루트 엔드포인트 '/' 요청 수신")
    # 모듈 레벨의 상태를 확인하고 알맞은 문자열을 반환
    return {"message": f"Welcome to {settings.APP_NAME}! RAG System Status: {'Ready' if rag_pipeline_instance else 'Not Ready'}"}

logger.info("--- main.py 파일 로드 완료 ---")
# --- Uvicorn으로 실행 (개발 시 터미널에서 직접 실행 권장) ---
# if __name__ == "__main__":
#     import uvicorn
#     # settings.HOST, settings.PORT, settings.DEBUG_MODE 사용
#     uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG_MODE, log_level="info")
# 위의 uvicorn.run 방식은 reload가 불안정할 수 있으므로, 터미널에서 다음 명령 사용:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info