# rag_main_runner.py
"""
RAG 파이프라인의 전체적인 구성 및 실행 로직을 담당합니다.
RAGPipeline 클래스는 데이터 로드, 텍스트 처리, 임베딩, 벡터 저장소 관리,
LLM 연동, 체인 구성 등 RAG 시스템의 핵심 단계를 초기화하고,
사용자 질문에 대한 답변 생성 기능을 제공합니다.

uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
"""
# 설정값 import
from config import settings
# RAG 유틸 함수 import
from rag_utils import (
    get_text_splitter, # 청크 분할기
    split_documents, # Document 리스트를 청크 단위 Document 리스트로 분할
    get_embedding_model, # 임베딩 모델
    get_llm, # LLM 모델
    create_or_load_vectorstore, # 벡터DB 생성 또는 기존 DB 로드
    get_retriever, # 벡터DB 문서 검색기
    create_rag_chain # RAG 체인 구성
)
# 랭체인 문서의 기본 단위인 Document 클래스 import
from langchain_core.documents import Document
import os
# 로거
import logging

# 로거 객체 생성
logger = logging.getLogger(__name__)

# 데이터 로드 함수
# -> list[Document] : 반환 타입 힌트
def load_data_from_directory(data_directory_path: str) -> list[Document]:
    """
    지정된 디렉토리에서 텍스트 파일(.txt)을 읽어 Document 객체 리스트로 반환.
    """
    loaded_documents = []
    logger.info(f"데이터 로드 시도: '{data_directory_path}'")

    # 1. 경로 존재 여부 확인
    if not os.path.exists(data_directory_path):
        logger.error(f"오류: 데이터 디렉토리 '{data_directory_path}'를 찾을 수 없습니다.")
        return [] # 빈 리스트 반환
        # 데이터 로드 실패 시, 예외를 직접 발생시키는 대신 빈 리스트를 반환하여
        # 호출자(RAGPipeline)가 상황을 인지하고 후속 처리를 결정하도록 함.
        # (함수의 반환 타입 일관성 유지 및 유연한 오류 처리 가능)

    # 2. 해당 경로가 디렉토리인지 확인
    if not os.path.isdir(data_directory_path):
        logger.error(f"오류: '{data_directory_path}'는 유효한 디렉토리가 아닙니다.")
        return []

    logger.info(f"'{data_directory_path}' 디렉토리에서 파일 스캔 중...")
    total_files_in_dir = 0 # 디렉토리 내부에 존재하는 총 파일 수에 대한 변수 선언과 초기값 세팅
    loaded_txt_files = 0 # 불러온 text 파일 수에 대한 변수 선언과 초기값 세팅
    try:
        for filename in os.listdir(data_directory_path):
            total_files_in_dir += 1
            # os.path.join()은 운영체제에 올바른 경로 구분자를 사용하여 결과를 반환
            # (data_directory_path, '/ 또는 \', filename) 식으로 작성하지 않아도 됨
            file_path = os.path.join(data_directory_path, filename)

            # 파일인지 확인 (하위 디렉토리 또는 .text 확장자가 아니면 무시)
            if os.path.isfile(file_path):
                # 현재는 .txt 파일만 처리
                if filename.endswith(".txt"):
                    logger.info(f"텍스트 파일 읽기 시도: '{file_path}'")
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # .strip() : 문자열의 양쪽 공백을 제거
                            # content가 원래부터 비어있었거나, 공백으로만 이루어져 있었다면, 빈 문자열이 반환됨
                            # 빈 문자열, ""은 파이썬에서 False로 평가됨
                            if not content.strip(): # 내용이 비어있는지 확인 (공백만 있는 경우 포함)
                                logger.warning(f"경고: '{filename}' 파일 내용이 비어있거나 공백만 있습니다. 문서는 생성되지만 내용이 존재하지 않습니다.")
                            # 앞서 생성한 loaded_documents 리스트에 Document 타입으로 변환된 content와 metadata를 저장
                            loaded_documents.append(Document(page_content=content, metadata={"source": filename}))
                            logger.info(f"성공: '{filename}' 파일 로드 완료. (내용 길이: {len(content)})")
                            loaded_txt_files += 1
                    except Exception as e:
                        logger.error(f"오류: '{file_path}' 파일 읽기 중 예외 발생: {e}", exc_info=True)
                else:
                    logger.debug(f"무시: '{filename}' 파일은 .txt 파일이 아닙니다.")
            else:
                logger.debug(f"무시: '{filename}'은 파일이 아닌 디렉토리입니다.")

        logger.info(f"'{data_directory_path}' 디렉토리 스캔 완료. 총 파일 수: {total_files_in_dir}, 로드된 .txt 파일 수: {loaded_txt_files}")

    except Exception as e:
        logger.error(f"오류: '{data_directory_path}' 폴더 접근 또는 내용 읽기 중 예외가 발생: {e}", exc_info=True)
        return [] # 오류 발생 시 빈 리스트 반환

    if not loaded_documents:
        logger.warning(f"경고: '{data_directory_path}' 디렉토리에서 로드할 수 있는 .txt 파일이 없거나, 모든 파일 읽기에 실패했습니다.")

    return loaded_documents

class RAGPipeline:
    # 생성자 정의
    def __init__(self,
                 data_path: str = str(settings.DATA_PATH),
                 vectorstore_path: str = str(settings.VECTORSTORE_PATH),
                 # force_create_db = False : 벡터DB가 이미 존재한다면 새로 생성하지 않고 기존 것을 사용
                 force_create_db: bool = False):
        self.data_path = data_path
        self.vectorstore_path = vectorstore_path
        self.force_create_db = force_create_db
        self.rag_chain = None
        self._initialize_pipeline()

    # initialize : 초기화
    def _initialize_pipeline(self):
        logger.info("RAG 파이프라인 초기화 시작...")

        # 세팅된 data_path를 load_data_from_directory 함수에 전달해서 실행하고
        # 결과값인 Documents 타입의 리스트를 actual_docs 변수를 선언함과 동시에 저장
        actual_docs = load_data_from_directory(self.data_path)

        if not actual_docs and not self.force_create_db and not os.path.exists(self.vectorstore_path):
            logger.error("처리할 데이터가 없고, 기존 벡터 저장소도 없어 파이프라인을 초기화할 수 없습니다.")
            raise ValueError("데이터 또는 벡터 저장소 없음. RAG 파이프라인 초기화 실패.")

        # creation : 창조
        split_docs_for_creation = None
        if actual_docs:
            # 텍스트를 청크 단위로 분할해주는 분할기 생성
            text_splitter = get_text_splitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
            # actual_docs(로드된 Documents 타입의 리스트)를 더 작은 청크 단위의 Document 객체 리스트로 분할해서 저장
            split_docs_for_creation = split_documents(text_splitter, actual_docs)

        try:
            embeddings = get_embedding_model(model_name=settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}", exc_info=True)
            raise

        should_create_new_db = self.force_create_db
        # DB 생성 없이 기존 DB를 사용하려고 했는데, 처리해야 할 새로운 문서가 존재할 때
        if not should_create_new_db and split_docs_for_creation:
            if not os.path.exists(self.vectorstore_path):
                logger.info(f"'{self.vectorstore_path}' 경로에 벡터 저장소가 없어 새로 생성합니다.")
                # False로 세팅해뒀던 force_create_db 값을 True로 변경
                should_create_new_db = True
            else:
                try:
                    temp_vectorstore = create_or_load_vectorstore(embeddings=embeddings,
                                                                  persist_directory=self.vectorstore_path,
                                                                  force_create=False)
                    if temp_vectorstore._collection.count() == 0:
                        logger.info(f"'{self.vectorstore_path}'의 기존 벡터 저장소가 비어있어 새로 생성합니다.")
                        should_create_new_db = True
                except Exception as e:
                    logger.warning(f"기존 벡터 저장소 '{self.vectorstore_path}' 로드 실패({e}), 새로 생성합니다.")
                    should_create_new_db = True

        # 새로운 DB를 생성해서 사용하려고 했는데, 처리해야 할 새로운 문서가 존재하지 않을 때
        if should_create_new_db and not split_docs_for_creation:
            logger.warning("DB를 새로 생성하도록 설정되었으나, 제공된 문서가 없습니다. 빈 벡터 저장소가 생성될 수 있습니다.")

        try:
            vectorstore = create_or_load_vectorstore(
                # should_create_new_db가 True이면 split_docs_for_creation을 사용 (비어있을 수도 있음),
                # False이면 None을 사용 (기존 DB 로드 시 새 문서 불필요).
                split_docs=split_docs_for_creation if should_create_new_db else None,
                embeddings=embeddings,
                persist_directory=self.vectorstore_path,
                # 상단의 if문들을 거쳐 세팅된 should_create_new_db의 참/거짓 값대로 force_create가 할당됨
                force_create=should_create_new_db
            )
        except Exception as e:
            logger.error(f"벡터 저장소 처리 중 오류: {e}", exc_info=True)
            raise

        try:
            logger.info(f"[RAG_PIPELINE_DEBUG] LLM 호출 시도. 세팅된 GOOGLE_API_KEY: {'********' if settings.GOOGLE_API_KEY else 'None'}")
            llm = get_llm(
                model_name=settings.LLM_MODEL_NAME
            )
        except Exception as e:
            logger.error(f"LLM 로드 실패: {e}", exc_info=True)
            raise

        # 문서 검색기 생성
        retriever = get_retriever(vectorstore, k=settings.SEARCH_K)
        # RAG 파이프라인 생성
        self.rag_chain = create_rag_chain(retriever, llm, prompt_template_str=settings.PROMPT_TEMPLATE)
        logger.info("RAG 파이프라인 초기화 완료.")

    # 사용자 질문을 전달해 LLM 답변을 반환
    def query(self, question: str) -> str:
        if not self.rag_chain:
            logger.error("RAG 파이프라인이 초기화되지 않았습니다.")
            return "오류: RAG 시스템이 준비되지 않았습니다."
        try:
            logger.info(f"RAG 파이프라인으로 질문 처리 중: {question}")
            answer = self.rag_chain.invoke(question)
            logger.info(f"RAG 파이프라인 답변 생성 완료.")
            return answer
        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {e}", exc_info=True)
            return "답변 생성 중 오류가 발생했습니다."