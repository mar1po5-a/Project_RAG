# chat_memory.py
import json
import os
from typing import List, Dict, Any

from filelock import FileLock
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, message_to_dict, messages_from_dict
import logging
from config import settings

from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# langchain의 BaseChatMessageHistory 클래스를 상속
class CustomChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, member_id: str, file_path: str = "chat_history.json"):
        if not member_id:
            raise ValueError("member_id는 필수 항목입니다.")
        self.member_id = member_id
        self.file_path = file_path
        # 파일 동시 접근 제어
        self.lock_file_path = f"{self.file_path}.lock"  # 잠금 파일 경로 (실제 데이터 파일과 같은 이름의 .lock 파일)
        self.lock = FileLock(self.lock_file_path, timeout=10)  # 10초 타임아웃 설정
        self._ensure_file_exists()
        self._check_and_manage_file()

    def _check_and_manage_file(self):
        with self.lock:
            if not os.path.exists(self.file_path):
                logger.debug(f"파일({self.file_path})이 존재하지 않아 파일 관리를 건너뜁니다.")
                return

            file_deleted = False

            try:
                file_mod_timestamp = os.path.getmtime(self.file_path)
                file_mod_datetime = datetime.fromtimestamp(file_mod_timestamp)
                current_datetime = datetime.now()
                # datetime과 float의 연산으로 type miss match warning이 뜨지만
                # 연산을 진행하는 과정 중에 datetime으로 변환되기에 문제되지 않음
                file_age = current_datetime - file_mod_datetime

                # ----------- 테스트용 -----------
                # file_age_minutes = file_age.total_seconds() / 60
                # if file_age_minutes > settings.MAX_CHAT_HISTORY_FILE_AGE_MINUTES:
                # ----------- 테스트용 -----------

                if file_age.days > settings.MAX_CHAT_HISTORY_FILE_AGE_DAYS:
                    logger.warning(
                        f"채팅 기록 파일({self.file_path}) 유효 기간 초과: {file_age.days} 일 (최대 {settings.MAX_CHAT_HISTORY_FILE_AGE_DAYS} 일). 파일을 삭제합니다.")
                    os.remove(self.file_path)
                    file_deleted = True
                else:
                    logger.debug(f"파일({self.file_path}) 유효 기간 ({file_age.days} 일)이 아직 유효합니다.")

            except OSError as e:
                logger.error(f"파일 유효 기간 확인 중 오류 발생: {self.file_path}, 오류: {e}")
            except Exception as e:
                logger.error(f"파일 유효 기간 초과 검사 중 예상치 못한 오류: {e}", exc_info=settings.DEBUG_MODE)

                # 파일이 존재하지 않으면 재생성
                # _ensure_file_exists는 자체적으로 락을 획득하므로,
                # 현재 _check_and_manage_file이 이미 락을 가진 상태에서 호출하면
                # 락 중복 획득 시도 또는 불필요한 복잡성을 피하기 위함
                # 현재 락 상태를 유지하며 직접 빈 파일로 초기화하는 것이 더 안전하고 명확함
            if file_deleted:
                logger.info(f"파일({self.file_path})이 삭제되었으므로 다시 생성합니다.")
                try:
                    with open(self.file_path, "w", encoding='utf-8') as f:
                        json.dump({}, f)
                    logger.info(f"채팅 기록 파일 재생성: {self.file_path}")
                except IOError as e:
                    logger.error(f"채팅 기록 파일 재생성 실패: {self.file_path}, 오류: {e}")
                    raise

    # 디렉토리 및 파일 존재 여부 확인
    def _ensure_file_exists(self):
        directory = os.path.dirname(self.file_path)

        # 상위 디렉토리가 존재하지 않으면
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"디렉토리 생성: {directory}")
            except OSError as e:
                logger.error(f"디렉토리 생성 실패: {directory}, 오류: {e}")
                raise

        with self.lock:
            # 해당 경로에 파일이 존재하지 않으면
            if not os.path.exists(self.file_path):
                try:
                # 쓰기 모드(w)로 파일을 열고 빈 딕셔너리를 작성
                    with open(self.file_path, "w", encoding='utf-8') as f:
                        # json.dumps() : 파이썬 객체를 문자열로 반환
                        # json.dump() : 파이썬 객체를 json으로 파일에 저장
                        json.dump({}, f)
                    logger.info(f"채팅 기록 파일 생성: {self.file_path}")
                except IOError as e:
                    logger.error(f"채팅 기록 파일 생성 실패: {self.file_path}, 오류: {e}")
                    raise

    # 대화 기록 불러오기
    # -> dict : 반환 타입 힌트
    def _load_all_users_history(self) -> Dict[str, List[Dict[str, Any]]]:
        # 동시 접근이 제어된 상태
        with self.lock:
            try:
                with open(self.file_path, "r", encoding='utf-8') as f:
                    file_content = f.read()
                    if not file_content.strip():
                        logger.info(f"채팅 기록 파일({self.file_path})이 비어있습니다.")
                        return {}

                    data = json.loads(file_content)

                    if not isinstance(data, dict):
                        logger.error(f"채팅 기록 파일({self.file_path})의 최상위 JSON 구조가 딕셔너리가 아닙니다. 파일을 확인해주세요. (타입: {type(data)})")
                        return {}
                    return data
            except json.JSONDecodeError:
                logger.error(f"채팅 기록 파일({self.file_path}) JSON 디코딩 오류. 파일이 손상되었을 수 있습니다. 빈 기록으로 대체합니다.")
                return {}
            except IOError as e:
                logger.error(f"채팅 기록 파일 읽기 오류: {self.file_path}, 오류: {e}")
                return {}
            except Exception as e:
                logger.error(f"채팅 기록 로드 중 예상치 못한 오류: {e}", exc_info=settings.DEBUG_MODE)
                return {}

    # 메서드를 클래스의 속성처럼 사용할 수 있게 해주는 데코레이터
    # 호출 없이 일반 변수처럼 사용 가능
    # @property - getter | @(propertyname).setter - setter
    # @(propertyname).deleter - deleter
    @property
    def messages(self) -> List[BaseMessage]:
        all_users_data = self._load_all_users_history()
        user_messages_json = all_users_data.get(self.member_id, [])
        return messages_from_dict(user_messages_json)

    @messages.setter
    def messages(self, messages: List[BaseMessage]) -> None:
        all_users_data = self._load_all_users_history()
        # BaseMessage 객체 리스트를 JSON 직렬화 가능한 딕셔너리 리스트로 변환
        all_users_data[self.member_id] = [message_to_dict(msg) for msg in messages]
        self._save_all_users_history(all_users_data)

    def _save_all_users_history(self, all_users_history: Dict[str, List[Dict[str, Any]]]):
        self._check_and_manage_file()

        with self.lock:
            try:
                with open(self.file_path, "w", encoding='utf-8') as f:
                    json.dump(all_users_history, f, ensure_ascii=False, indent=2)
            except IOError as e:
                logger.error(f"채팅 기록 저장 실패: {self.file_path}, 오류: {e}")
            except Exception as e:
                logger.error(f"채팅 기록 저장 중 예상치 못한 오류: {e}", exc_info=settings.DEBUG_MODE)

    def add_message(self, message: BaseMessage) -> None:
        current_user_message: List[BaseMessage] = self.messages

        current_user_message.append(message)

        try:
            self.messages = current_user_message
        except Exception as e:
            logger.error(f"사용자 '{self.member_id}'의 메시지 추가 후 저장 중 오류: {e}", exc_info=settings.DEBUG_MODE)

    def clear(self) -> None:
        with self.lock:
            all_users_data = self._load_all_users_history()

            if self.member_id in all_users_data:
                all_users_data[self.member_id] = []
                try:
                    self._save_all_users_history(all_users_data)
                    logger.info(f"사용자 '{self.member_id}'의 대화 기록이 삭제되었습니다.")
                except Exception as e:
                    logger.error(f"사용자 '{self.member_id}'의 대화 기록 삭제 후 저장 중 오류: {e}", exc_info=settings.DEBUG_MODE)
                    raise
            else:
                logger.info(f"사용자 '{self.member_id}'의 삭제할 대화 기록이 존재하지 않습니다.")


class UserChatMemory:
    def __init__(self, member_id: str, history_file_path: str = "chat_history.json"):
        if not member_id:
            raise ValueError("UserChatMemory 생성 시 member_id는 필수 항목입니다.")
        self.member_id: str = member_id
        self.file_path: str = history_file_path

        self.chat_message_history: BaseChatMessageHistory = \
        CustomChatMessageHistory(
            member_id=self.member_id,
            file_path=self.file_path
        )

    def add_question(self, question: str) -> None:
        if not isinstance(question, str):
            logger.warning(f"add_question에 문자열이 아닌 입력이 들어왔습니다: {type(question)}")

            try:
                question = str(question)
            except Exception:
                logger.error(f"add_question 입력({question})을 문자열로 변환할 수 없습니다.")
                return

        self.chat_message_history.add_message(HumanMessage(content=question))

    def add_answer(self, answer: str) -> None:
        if not isinstance(answer, str):
            logger.warning(f"add_answer에 문자열이 아닌 입력이 들어왔습니다: {type(answer)}")

            try:
                answer = str(answer)
            except Exception:
                logger.error(f"add_answer 입력({answer})을 문자열로 변환할 수 없습니다.")
                return

        self.chat_message_history.add_message(AIMessage(content=answer))

    def get_chat_messages(self) -> List[BaseMessage]:
        return self.chat_message_history.messages

    def get_history_for_api(self) -> List[Dict[str, str]]:
        messages = self.get_chat_messages()
        history_list = []
        for msg in messages:
            role = ""
            if isinstance(msg, HumanMessage):
                role = "human"
            elif isinstance(msg, AIMessage):
                role = "ai"
            else:
                role = "unknown"
                logger.warning(
                    f"get_history_for_api: 사용자 '{self.member_id}'의 대화 기록에 알 수 없는 메시지 타입({type(msg).__name__})이 포함되어 있습니다. "
                    f"Content: '{str(msg.content)[:50]}...'"  # 내용 일부 로깅
                )
            history_list.append({
                "role" : role,
                "content" : msg.content
            })
        return history_list

    def clear_history(self) -> None:
        self.chat_message_history.clear()