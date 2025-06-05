#----------------------------테스트용-------------------------
from chat_memory import UserChatMemory
from typing import Dict, List

user_memories: Dict[str, UserChatMemory] = {}

class MockRAGPipeline:
    def query(self, question:str, user_id:str) -> str:
        logger.info(f"MockRAGPipeline.query called for user '{user_id}' with question: '{question}'")

        return f"Answer to '{question}' for user '{user_id}' (history not used in mock)"

mock_rag_pipeline = MockRAGPipeline()

class MemoryTestSearchResponse(BaseModel):
    answer: str

class MemoryTestHistoryResponse(BaseModel):
    history: List[Dict[str, str]]

def get_or_create_user_memory(user_id: str) -> UserChatMemory:
    if user_id not in user_memories:
        logger.info(f"UserChatMemory for '{user_id}' not found. Creating new instance.")
        user_memories[user_id] = UserChatMemory(user_id=user_id)
    return user_memories[user_id]

@app.post("/memory-test/search", response_model=MemoryTestSearchResponse)
async def memory_test_search_system(
        payload: dict = Body(...)
):

    question = payload.get("question")
    user_id_from_request = payload.get("user_id") # 요청에서 받은 user_id 사용

    logger.info(f"'/search' endpoint received: user_id='{user_id_from_request}', question='{question}'")

    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question text is required.")
    if not user_id_from_request or not user_id_from_request.strip():
        raise HTTPException(status_code=400, detail="User ID is required.")

    user_memory = get_or_create_user_memory(user_id_from_request)
    answer = mock_rag_pipeline.query(question, user_id_from_request)

    user_memory.add_question(question)
    user_memory.add_answer(answer)
    return MemoryTestSearchResponse(answer=answer)

@app.get("/chat-history/{user_id_path}", response_model=MemoryTestHistoryResponse)
async def memory_test_get_chat_history(user_id_path: str = FastApiPath(..., title="User ID")):
    logger.info(f"Fetching chat history for user_id: {user_id_path}")
    user_memory = get_or_create_user_memory(user_id_path)
    history_data = user_memory.get_history_for_api()
    return MemoryTestHistoryResponse(history=history_data)

@app.post("/chat-history/{user_id_path}/clear")
async def clear_user_chat_history(user_id_path: str = FastApiPath(..., title="User ID")):
    logger.info(f"Clearing chat history for user_id: {user_id_path}")
    user_memory = get_or_create_user_memory(user_id_path)
    user_memory.clear_history()
    return {"message": f"Chat history cleared for user_id: {user_id_path}"}

#----------------------------테스트용-------------------------