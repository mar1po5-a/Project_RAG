# simple_run.py
"""
[테스트용 스크립트]
RAG 파이프라인의 핵심 기능을 테스트하는 모듈입니다. 터미널에서 실행하세요.

주요 역할:
- `RAGPipeline` 객체를 초기화합니다.
- 사용자로부터 질문을 입력받아, 초기화된 파이프라인을 통해 답변을 생성하고 출력합니다.
- FastAPI 웹 서버 없이 RAG 시스템의 기본적인 질문-답변 기능을 빠르게 검증하는 데 사용됩니다.

정상적으로 실행되지만, ChromaDB-LangChain Version Warning이 발생할 수 있습니다.
"""

from rag_main_runner import RAGPipeline

def simple_run_rag_app():
    data_folder = "./my_data_directory"

    try:
        print("RAG 파이프라인을 설정합니다...")

        rag_pipeline = RAGPipeline(data_path=data_folder, force_create_db=False)

    except Exception as e:
        print(f"RAG 파이프라인 초기화에 실패했습니다: {e}")
        return

    print("\nRAG 애플리케이션이 준비되었습니다. 질문을 입력하세요.")
    print("종료하려면 'exit' 또는 'quit'를 입력하세요.\n")

    while True:
        try:
            question = input("질문: ")
            if question.lower() in ["exit", "quit"]:
                print("애플리케이션을 종료합니다.")
                break
            if not question.strip():
                continue

            answer = rag_pipeline.query(question)
            print(f"답변: {answer}\n")

        except Exception as e:
            print(f"질문 처리 중 오류 발생: {e}")

if __name__ == "__main__":
    simple_run_rag_app()