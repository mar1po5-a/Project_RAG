# rag_utils.py
"""
RAG 파이프라인을 구성하는 각 개별 단계를 위한 유틸리티 함수들을 제공합니다.
텍스트 분할기 생성, 문서 분할, 임베딩 모델 로드, LLM 로드,
벡터 저장소 생성/로드, 리트리버 생성, RAG 체인 구성 등의 기능을 포함합니다.
rag_main_runner.py의 RAGPipeline 클래스에서 이 함수들을 호출하여 사용합니다.

uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info
"""
from config import settings
from dotenv import load_dotenv
import os
# 제미나이 LLM import
from langchain_google_genai import ChatGoogleGenerativeAI
# Chroma 벡터 DB import
from langchain_community.vectorstores import Chroma
# 랭체인 문서의 기본 단위인 Document 클래스 import
from langchain_core.documents import Document
# 한국어 형태소 분할기 import
from langchain_text_splitters import KonlpyTextSplitter
# 프롬프트 템플릿 import
from langchain_core.prompts import ChatPromptTemplate
# 사용자의 질문을 어떤 변환이나 처리 과정을 거치지 않고 RAG 체인에 그대로 전달하기 위해 import
from langchain_core.runnables import RunnablePassthrough
# 기본적으로 LLM은 여러가지 데이터가 담긴 구조화 객체를 반환함
# 그 중에서 LLM의 답변 부분만 추출해 문자열로 변환시켜주는 역할
from langchain_core.output_parsers import StrOutputParser
# 허깅 페이스 임베딩 모델
import torch
from langchain_huggingface import HuggingFaceEmbeddings
# 로거
import logging

# 로거 생성
logger = logging.getLogger(__name__)

# 텍스트 분할기
def get_text_splitter(chunk_size: int = settings.CHUNK_SIZE,
                      chunk_overlap: int = settings.CHUNK_OVERLAP):
    logger.info(f"텍스트 분할기 초기화 (chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap})")
    return KonlpyTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )

# Documents 분할기
def split_documents(text_splitter, docs: list[Document]):
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"문서 분할 완료: 원본 {len(docs)}개 -> 청크 {len(split_docs)}개")
    return split_docs

# 임베딩 모델 호출
def get_embedding_model(model_name: str = settings.EMBEDDING_MODEL_NAME):
    try:
        # CUDA를 지원하는 NVIDIA GPU가 사용 가능하면 'cuda' (GPU 사용), 그렇지 않으면 'cpu' (CPU 사용)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            # 임베딩 벡터의 정규화
            # 임베딩 모델이 만든 숫자 벡터들의 길이를 모두 1로 통일시켜서, 의미적인 방향 비교를 더 쉽고 정확하게 하자
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"임베딩 모델 '{model_name}' 로드 완료 (device: {device})")

        return embeddings
    
    except Exception as e:
        logger.error(f"HuggingFaceEmbeddings ({model_name}) 초기화 중 오류: {e}", exc_info=True)
        raise

# LLM 모델 호출
def get_llm(model_name: str = settings.LLM_MODEL_NAME,
            temperature: float = 0.3):

    api_key_to_use = settings.GOOGLE_API_KEY

    if not api_key_to_use: # 인자로 받은 google_api_key가 None이거나 비어있으면 오류 발생
        logger.error("LLM 초기화 오류: Google API 키가 설정되지 않았습니다 (settings.GOOGLE_API_KEY 확인).")
        raise ValueError("Google API 키가 LLM 초기화에 필요합니다 (settings.GOOGLE_API_KEY 확인).")

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key_to_use,
            model_kwargs={"include_thoughts": False}
        )
        logger.info(f"LLM '{model_name}' 로드 완료 (창의성: {temperature})")
        return llm
    except Exception as e:
        logger.error(f"ChatGoogleGenerativeAI ({model_name}) 초기화 중 오류: {e}", exc_info=True)
        raise

# 벡터DB 생성 또는 기존DB 불러오기
def create_or_load_vectorstore(
        split_docs: list[Document] | None = None,
        embeddings=None,
        persist_directory: str = str(settings.VECTORSTORE_PATH),
        force_create: bool = False
):
    # DB를 강제로 생성해야 하거나, 디렉토리가 없거나, 기존 디렉토리가 존재하나 비어있을 때
    if force_create or not os.path.exists(persist_directory) or \
            (os.path.exists(persist_directory) and not os.listdir(persist_directory)):
        if not split_docs or not embeddings:
            raise ValueError("새 벡터 저장소 생성 시 split_docs와 embeddings가 필요합니다.")
        logger.info(f"'{persist_directory}'에 새 벡터 저장소 생성 중...")
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        logger.info(f"벡터 저장소 생성 완료. 총 {vectorstore._collection.count()}개의 벡터 저장됨.")
  
    else:
        if not embeddings:
            raise ValueError("기존 벡터 저장소 로드 시 embeddings가 필요합니다.")
        logger.info(f"'{persist_directory}'에서 기존 벡터 저장소 로드 중...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        logger.info(f"벡터 저장소 로드 완료. 총 {vectorstore._collection.count()}개의 벡터 확인됨.")
    return vectorstore

# 문서 검색기
def get_retriever(vectorstore, k: int = settings.SEARCH_K):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    logger.info(f"Retriever 생성 완료 (검색 결과 수: {k})")
    return retriever

# RAG 파이프라인 생성
def create_rag_chain(retriever, llm, prompt_template_str: str = settings.PROMPT_TEMPLATE):
    prompt = ChatPromptTemplate.from_template(prompt_template_str)
    rag_chain = (
        # 검색된 상위 문서 3개는 동일한 원본 데이터에서 분할된 데이터이거나 아주 유사한 데이터일 가능성이 높음
        # 따라서 LLM의 context 이해를 돕기 위해 하나의 통합된 텍스트 블록(문자열)으로 만들어서 보내는 것임
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
         "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("RAG 체인 구성 완료.")
    return rag_chain