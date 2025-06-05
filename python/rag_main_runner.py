# rag_main_runner.py
"""
RAG 파이프라인의 전체적인 구성 및 실행 로직을 담당합니다.
RAGPipeline 클래스는 데이터 로드, 텍스트 처리, 임베딩, 벡터 저장소 관리,
LLM 연동, 체인 구성 등 RAG 시스템의 핵심 단계를 초기화하고,
사용자 질문에 대한 답변 생성 기능을 제공합니다.

uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
"""
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage

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
from data_manager import (
    scan_data_directory,
    load_metadata,
    save_metadata,
    get_changed_files,
    update_metadata_after_processing,
    remove_metadata_for_deleted_files
)
# 파일 경로를 객체로 다루기 위한 import
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
# 디렉토리 및 파일 트리 삭제 등 고수준 파일/디렉토리 작업을 위한 import
import shutil

# 로거 객체 생성
logger = logging.getLogger(__name__)

# 데이터 로드 함수
# -> list[Document] : 반환 타입 힌트
def load_docs_from_paths(base_path: Path, relative_paths: List[str]) -> list[Document]:
    """
    지정된 디렉토리에서 텍스트 파일(.txt)을 읽어 Document 객체 리스트로 반환.
    """
    loaded_docs = []
    if not relative_paths:
        return loaded_docs
    logger.info(f"지정된 경로에서 문서 로드 시도 (기준: '{base_path}'): {len(relative_paths)}개 파일")

    for rel_path_str in relative_paths:
        abs_file_path  = base_path / rel_path_str
        file_name = abs_file_path.name

        if not abs_file_path.is_file():
            logger.warning(f"파일을 찾을 수 없거나 파일이 아님 (건너뜀): '{abs_file_path}'")
            continue

        logger.debug(f"파일 읽기 시도: '{abs_file_path}'")
        try:
            with open(abs_file_path, "r", encoding='utf-8') as f:
                content = f.read()
                # .strip() : 문자열의 양쪽 공백을 제거
                # content가 원래부터 비어있었거나, 공백으로만 이루어져 있었다면, 빈 문자열이 반환됨
                # 빈 문자열, ""은 파이썬에서 False로 평가됨
            if not content.strip(): # 내용이 비어있는지 확인 (공백만 있는 경우 포함)
                logger.warning(f"'{file_name}' 파일 내용이 비어있거나 공백만 있음.")

            doc_metadata = {
                "source": file_name,
                "relative_path": rel_path_str
            }
            # 앞서 생성한 loaded_documents 리스트에 Document 타입으로 변환된 content와 metadata(파일명, 상대 경로)를 저장
            loaded_docs.append(Document(page_content=content, doc_metadata=doc_metadata))
            logger.debug(f"성공: '{file_name}' 로드 완료 (내용 길이: {len(content)})")
        except Exception as e:
            logger.error(f"오류: '{abs_file_path}' 파일 읽기 중 예외 발생: {e}", exc_info=True)

        if not loaded_docs and relative_paths:
            logger.warning("요청된 파일 목록에서 유효한 문서를 로드하지 못했습니다.")
        elif loaded_docs:
            logger.info(f"총 {len(loaded_docs)}개의 문서 로드 완료.")

    return loaded_docs


class RAGPipeline:
    # 생성자 정의
    def __init__(self,
                 data_path: str = str(settings.DATA_PATH),
                 vectorstore_path: str = str(settings.VECTORSTORE_PATH),
                 # force_create_db = False : 벡터DB가 이미 존재한다면 새로 생성하지 않고 기존 것을 사용
                 force_create_db: bool = False,
                 # 모든 파일을 강제로 재임베딩할지 여부
                 force_reprocess_all_files: bool = False
                 ):
        self.data_path = Path(data_path) # 문자열로 들어온 data_path를 Path 객체로 변환
        self.vectorstore_path = vectorstore_path
        self.force_create_db = force_create_db
        self.force_reprocess_all_files = force_reprocess_all_files
        self.rag_chain = None
        # 기존의 지역 변수였던 embeddings와 vectorstore를 클래스 변수로 관리
        # 다른 함수들에서 해당 변수들에 접근할 수 있게 만들기 위함
        # 기존에는 _initialize_pipeline 함수가 끝나면 사라졌었음
        self.embeddings = None
        self.vectorstore: Chroma | None = None
        self._initialize_pipeline()

    # 삭제할 문서 리스트의 상대 경로를 전달받아 문서를 삭제하는 함수
    def _delete_docs_by_relative_paths(self, relative_paths: List[str]):
        if not self.vectorstore or not relative_paths:
            logger.debug("벡터 저장소가 없거나 삭제할 경로 목록이 비어있어 삭제를 건너뜁니다.")
            return

        logger.info(f"벡터 DB에서 다음 상대 경로의 문서 삭제 시도 (ID 기반): {relative_paths}")

        # ids : 벡터DB에 저장된 문서들의 ID
        all_ids_to_delete: List[str] = []

        for rel_path_str in relative_paths:
            try:
                # ChromaDB의 get. where문 사용
                retrieved_docs_info = self.vectorstore.get(
                    where={"relative_path": rel_path_str}
                )

                if retrieved_docs_info and retrieved_docs_info.get("ids"):
                    ids_for_path = retrieved_docs_info["ids"]
                    all_ids_to_delete.extend(ids_for_path)
                    logger.debug(f"경로 '{rel_path_str}'에 대해 삭제할 ID 발견: {ids_for_path}")
                else:
                    logger.debug(f"경로 '{rel_path_str}'에 대해 DB에서 해당 문서를 찾지 못함 (이미 삭제되었거나 없음).")
            except Exception as e:
                logger.error(f"경로 '{rel_path_str}'의 ID 조회 중 오류 발생: {e}", exc_info=True)
                continue

            if all_ids_to_delete:
                unique_ids_to_delete = list(set(all_ids_to_delete))
                logger.info(f"삭제할 고유 벡터 ID 목록 ({len(unique_ids_to_delete)}개): {unique_ids_to_delete}")
                try:
                    self.vectorstore.delete(ids=unique_ids_to_delete)
                    self.vectorstore.persist()
                except Exception as e:
                    logger.error(f"벡터 삭제 API 호출 중 오류 발생 (ID: {unique_ids_to_delete}): {e}", exc_info=True)
                else:
                    logger.info("삭제할 벡터 ID가 없습니다.")


    # initialize : 초기화
    def _initialize_pipeline(self):
        logger.info("RAG 파이프라인 초기화 시작...")

        try:
            self.embeddings = get_embedding_model(model_name=settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}", exc_info=True)
            raise

        # 현재 존재하는 모든 데이터 파일을 읽어오고 {상대 경로: 현재 해시값}으로 저장
        current_files_hashes = scan_data_directory(self.data_path)
        # 현재 메타데이터
        previous_metadata = load_metadata()
        # 현재 파일 해시와 이전 메타데이터 비교, 변경된 파일(신규/수정/삭제) 목록 선언
        new_file_paths, modified_file_paths, deleted_file_paths = get_changed_files(
            current_files_hashes, previous_metadata
        )
        # Path 객체인 vectorstor_path를 문자열로 변환해서 저장
        db_path_str = str(self.vectorstore_path)
        # 벡터DB 초기화
        self.vectorstore = None

        db_action: str = "load_or_create"
        files_to_load_for_db: List[str] = []
        paths_to_delete_from_db: List[str] = []
        # 이전 메타데이터 복사해서 저장
        base_metadata_for_update: Dict[str, Any] = previous_metadata.copy()

        if self.force_create_db:
            logger.info(f"DB 강제 재생성 요청: 기존 벡터 저장소 '{db_path_str}' 삭제 시도.")
            # 기존 벡터DB가 존재하면 삭제 먼저 진행
            if os.path.exists(self.vectorstore_path):
                shutil.rmtree(self.vectorstore_path)
            db_action = "create_new"
            # 현재 해시값을 가지고 있는 모든 파일
            files_to_load_for_db = list(current_files_hashes.keys())
            # 기존 벡터DB를 삭제했기에 저장해둔 이전 메타데이터도 삭제
            base_metadata_for_update = {}
        elif self.force_reprocess_all_files:
            logger.info("모든 파일 강제 재처리 요청.")
            db_action = "load_and_reprocess_all"
            if previous_metadata:
                paths_to_delete_from_db.extend(list(previous_metadata.keys()))
            files_to_load_for_db = list(current_files_hashes.keys())
            base_metadata_for_update = {}
        else:
            if deleted_file_paths:
                paths_to_delete_from_db.extend(list(deleted_file_paths))
            if modified_file_paths:
                paths_to_delete_from_db.extend(modified_file_paths)

            files_to_load_for_db.extend(new_file_paths)
            files_to_load_for_db.extend(modified_file_paths)

            base_metadata_for_update = remove_metadata_for_deleted_files(
                deleted_file_paths, base_metadata_for_update
            )

        if db_action != "create new" and os.path.exists(self.vectorstore_path) and \
            any(Path(self.vectorstore_path).iterdir()):
                try:
                    self.vectorstore = Chroma(persist_directory=db_path_str, embedding_function=self.embeddings)
                    logger.info(f"기존 벡터 저장소 로드 완료 (액션: {db_action}).")
                except Exception as e:
                    logger.warning(f"기존 DB 로드 실패 ({e}). DB를 새로 생성합니다.", exc_info=True)

                    if os.path.exists(self.vectorstore_path):
                        shutil.rmtree(self.vectorstore_path)
                        self.vectorstore_path = None
                        db_action = "create_new"

                if paths_to_delete_from_db and self.vectorstore:
                    self._delete_docs_by_relative_paths(paths_to_delete_from_db)
                elif paths_to_delete_from_db and not self.vectorstore:
                    logger.warning(f"삭제할 벡터 경로({paths_to_delete_from_db})가 있으나, DB 객체가 로드되지 않아 삭제를 건너뜁니다 (DB가 새로 생성될 예정).")

        split_docs_for_db: list[Document] = []
        if files_to_load_for_db:
            docs_to_process = load_docs_from_paths(
                self.data_path,
                files_to_load_for_db
            )
            if docs_to_process:
                # 텍스트를 청크 단위로 분할해주는 분할기 생성
                text_splitter = get_text_splitter(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
                # actual_docs(로드된 Documents 타입의 리스트)를 더 작은 청크 단위의 Document 객체 리스트로 분할해서 저장
                split_docs_for_db = split_documents(text_splitter, docs_to_process)
                logger.info(f"{len(split_docs_for_db)}개의 문서 청크를 DB에 반영할 예정입니다.")

            if db_action == "create_new" and self.vectorstore is None:
                if not split_docs_for_db and db_action == "create_new":
                    logger.warning("새 DB 생성 요청되었으나 처리할 문서가 없습니다. 빈 DB가 생성될 수 있습니다.")

                self.vectorstore = Chroma.from_documents(
                    documents=split_docs_for_db if split_docs_for_db else [],
                    embedding=self.embeddings,
                    persist_directory=db_path_str
                )
                logger.info(f"새 벡터 저장소 생성 완료. 저장된 청크 수: {len(split_docs_for_db)}")

            elif split_docs_for_db:
                logger.info(f"{len(split_docs_for_db)}개의 청크를 기존 벡터 저장소에 추가합니다.")
                self.vectorstore.add_documents(documents=split_docs_for_db)
                logger.info("문서 추가 완료.")

                if self.vectorstore is None:
                    logger.error("벡터 저장소를 초기화할 수 없습니다.")
                    raise ValueError("벡터 저장소 초기화 실패.")
                self.vectorstore.persist()

        if files_to_load_for_db or deleted_file_paths:
            final_metadata_to_save = update_metadata_after_processing(
                files_to_load_for_db,
                current_files_hashes,
                base_metadata_for_update
            )
            save_metadata(final_metadata_to_save)

        # try:
        #     self.rag_chain = create_rag_chain(
        #         llm = get_llm(model_name=settings.LLM_MODEL_NAME),
        #         retriever=get_retriever(self.vectorstore, settings.SEARCH_K),
        #         prompt_template_str=settings.PROMPT_TEMPLATE
        #     )
        # except Exception as e:
        #     logger.error(f"LLM, Retriever 또는 RAG 체인 설정 실패: {e}", exc_info=True)
        #     raise

        if self.vectorstore is None: # 방어 코드: vectorstore가 초기화되지 않았다면
            logger.error("Vectorstore is not initialized. Cannot create retriever, prompt, llm.")
            raise ValueError("Vectorstore initialization failed.")

        try:
            self.retriever = get_retriever(self.vectorstore, settings.SEARCH_K)
            self.llm = get_llm(model_name=settings.LLM_MODEL_NAME)
            self.prompt = ChatPromptTemplate.from_template(settings.PROMPT_TEMPLATE)
            self.output_parser = StrOutputParser()
            logger.info("Retriever, LLM, Prompt, OutputParser 초기화 완료.")
        except Exception as e:
            logger.error(f"RAG 구성 요소 (Retriever, LLM, Prompt, Parser) 초기화 실패: {e}", exc_info=True)
            raise

        logger.info("RAG 파이프라인 초기화 완료.")


    # 사용자 질문을 전달해 LLM 답변을 반환
    def query(self, question: str, history: Optional[List[BaseMessage]] = None) -> str:

        if not all([self.retriever, self.prompt, self.llm, self.output_parser]):
            logger.error("RAG 파이프라인의 일부 구성요소가 초기화되지 않았습니다.")
            return "오류: RAG 시스템이 준비되지 않았습니다."

        try:
            logger.info(f"RAG 파이프라인으로 질문 처리 중: {question}")

            retrieved_docs: List[Document] = self.retriever.invoke(question)

            logger.debug(f"검색된 문서 개수: {len(retrieved_docs)}")
            for i, doc in enumerate(retrieved_docs):
                logger.debug(f"문서 {i + 1} 소스: {doc.metadata.get('source', 'N/A')}, 내용 일부: {doc.page_content[:100]}...")

            context_parts = []
            for doc in retrieved_docs:
                context_parts.append(doc.page_content)

            context_str = "\n\n".join(context_parts)

            chat_history_str = ""
            if history:
                history_lines = []
                for msg in history:
                    if msg.type == "human":
                        history_lines.append(f"사용자: {msg.content}")
                    elif msg.type == "ai":
                        history_lines.append(f"AI: {msg.content}")
                chat_history_str = "\n".join(history_lines)
            else:
                chat_history_str = "이전 대화 기록이 존재하지 않습니다."

            chain = (
                    self.prompt
                    | self.llm
                    | self.output_parser
            )

            answer = chain.invoke({
                "chat_history": chat_history_str,
                "context": context_str,
                "question": question
            })

            logger.info(f"RAG 파이프라인 답변 생성 완료.")
            return answer  # 답변과 소스 문서 내용 리스트 반환

        except Exception as e:
            logger.error(f"질문 처리 중 오류 발생: {e}", exc_info=True)
            return "답변 생성 중 오류가 발생했습니다."


    # def query(self, question: str) -> str:
        # if not self.rag_chain:
        #     logger.error("RAG 파이프라인이 초기화되지 않았습니다.")
        #     return "오류: RAG 시스템이 준비되지 않았습니다."
        # try:
        #     logger.info(f"RAG 파이프라인으로 질문 처리 중: {question}")
        #     answer = self.rag_chain.invoke(question)
        #     logger.info(f"RAG 파이프라인 답변 생성 완료.")
        #     return answer
        # except Exception as e:
        #     logger.error(f"질문 처리 중 오류 발생: {e}", exc_info=True)
        #     return "답변 생성 중 오류가 발생했습니다."