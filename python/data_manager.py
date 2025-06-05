# data_manager.py
"""
{ 파일 상대 경로 : 파일 해시값 }에 대한 목록을 JSON으로 저장하고 관리합니다.

해당 모듈은 새로운 데이터 파일을 추가할 때,
기존에 존재하던 벡터 DB를 수동 삭제해야하는
번거로움을 해결하기 위해 제작되었습니다.

- .txt 파일 스캔 후 현재 파일별 해시값 목록 생성
- 이전 처리 정보(JSON)와 현재 파일 목록/해시값 비교
- 변경 파일(신규/수정/삭제) 목록 반환
- 처리 후 최신 파일 정보를 JSON에 업데이트

* 일반적인 해시코드: 객체 식별용 정수값. 내용/ID가 같으면 해시코드도 동일.
* 파일 해시: 파일 내용 기반 고유 식별값 (16진수 문자열). 내용 변경 시 해시값도 변경됨.
"""
import os
# 다양한 해시 함수를 제공하는 모듈
import hashlib
import json
from json import JSONDecodeError
# 파일 경로를 객체로 다루기 위한 import
from pathlib import Path
# 로거
import logging
from typing import List, Dict, Tuple, Set, Any

# config.py setting load
import config

logger = logging.getLogger(__name__)

# 메타데이터를 저장할 파일의 경로
METADATA_FILE_PATH = config.BASE_DIR / "data_files_metadata.json"
ALLOWED_EXTENSIONS = {".txt"} # 허용된 확장자

# 파일 내용의 SHA256 해시값을 계산하여 문자열로 반환. 오류 시 빈 문자열.
def calculate_file_hash(file_path: Path) -> str:
    # sha256() : 256비트의 고정된 해시값을 출력, 현재는 초기 상태
    sha256_hash = hashlib.sha256()
    try:
        # rb : byte 단위로 읽기 모드
        with open(file_path, "rb") as f:
            # iter() 함수는 두 개의 인자를 받을 때 특별한 방식으로 동작
            # 첫번째 인자 : 호출 가능한 객체
            # 두번째 인자 : 종료 값
            # f.read(4096) : 최대 4096byte 만큼 데이터를 읽어옴, 일반적 관례로 4KB를 사용
            # b"" : 빈 바이트 문자열, f.read()가 파일의 끝에 도달하면 b""을 반환
            for byte_block in iter(lambda: f.read(4096), b""):
                # 읽어온 데이터 덩어리(byte_block)를 가져와서 sha256_hash 객체의 내부 상태를 변경
                sha256_hash.update(byte_block)
        # 최종적으로 계산된 256비트 해시값을 16진수 문자열로 출력
        return sha256_hash.hexdigest()
    # Input/Output Error, 파일 입출력 작업 중 발생한 오류에 대한 예외 처리
    except IOError:
        logger.error(f"파일 해시 생성하는 과정 중 오류 발생, 경로: {file_path}", exc_info=True)
        # 개별 파일의 해시 계산 실패가 전체 스캔 작업을 중단시키지 않도록 하기 위해 빈 문자열을 반환
        return ""

# JSON 파일에서 이전 파일 처리 메타데이터를 로드하여 딕셔너리로 반환.
def load_metadata() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(METADATA_FILE_PATH):
        return {} # 빈 딕셔너리 반환
    try:
        with open(METADATA_FILE_PATH, "r", encoding='utf-8') as f:
            # 파일을 json 형태로 읽어옴
            return json.load(f)
    except (JSONDecodeError, IOError):
        logger.error(f"메타 데이터 불러오는 과정에서 오류 발생, 경로: {METADATA_FILE_PATH}", exc_info=True)
        return {}

# 파일 처리 메타데이터 딕셔너리를 JSON 파일에 저장.
def save_metadata(metadata: Dict[str, Dict[str, Any]]):
    try:
        with open(METADATA_FILE_PATH, "w", encoding='utf-8') as f:
            # 전달받은 metadata를 f에 저장
            # indent=4 : 4칸 들여쓰기 허용
            json.dump(metadata, f, indent=4, ensure_ascii=False)
        logger.info(f"메타데이터 저장 완료, 경로: {METADATA_FILE_PATH}")
    except IOError:
        logger.error(f"메타데이터 저장 중 오류 발생, 경로: {METADATA_FILE_PATH}", exc_info=True)

# 데이터 디렉토리를 스캔, 파일들의 (상대 경로: 현재 해시값) 딕셔너리 반환.
def scan_data_directory(data_path: Path) -> Dict[str, str]:
    # 현재 모든 파일의 정보
    current_files_hashes: Dict[str, str] = {}
    if not data_path.is_dir():
        logger.warning(f"해당 경로에 디렉토리가 존재하지 않거나, 디렉토리가 아닙니다. 입력된 경로: {data_path}")
        # 초기값을 그대로 반환
        return current_files_hashes

    # os.work() : 입력 받은 경로의 하위 디렉토리를 전부 방문, 각 디렉토리의 튜플을 반환
    # 해당 튜플의 구조는 dirpath(str), dirnames(list), filenames(list)로 구성됨
    # root : dirpath | files : filenames
    # _(언더 스코어)는, os.walk()가 항상 세 개의 값을 가진 튜플을 반환하기에,
    # 문법적인 구조를 맞춰줌으로써 ValueError를 피하기 위해 사용됨
    for root, _, files in os.walk(data_path):
        for filename in files:
            # suffix : 파일 경로에서 마지막 점(.) 이후의 문자열 부분, 즉 확장자를 반환
            # 파일이 허용된 확장자를 사용하고 있는지 확인
            if Path(filename).suffix.lower() in ALLOWED_EXTENSIONS:
                # 현재 디렉토리(root)와 파일명(filename)을 결합하여 파일의 전체 경로 객체 생성
                file_path_obj = Path(root) / filename
                try:
                    # data_path를 기준으로 현재 파일(file_path_obj)의 상대 경로를 계산하여 문자열로 저장
                    relative_file_path_str = str(file_path_obj.relative_to(data_path))
                    # 현재 파일의 해시를 생성해서 저장
                    file_hash = calculate_file_hash(file_path_obj)
                    if file_hash:
                        # 스캔된 파일 목록(current_files_hashes)에 현재 파일의 상대 경로와 해시값 키:값 쌍으로 매핑
                        current_files_hashes[relative_file_path_str] = file_hash
                except ValueError:
                    logger.warning(f"{data_path}를 기준으로 {file_path_obj}에 대한 상대 경로를 확인할 수 없습니다. ")
                except Exception:
                    logger.error(f"{file_path_obj} 파일을 스캔하는 과정에서 오류 발생", exc_info=True)

        logger.debug(f"{data_path}에서 {len(current_files_hashes)} 파일을 스캔했습니다.")
        # 업데이트 된 current_files_hashes 반환
        return current_files_hashes

# 현재 파일 해시와 이전 메타데이터 비교, 변경된 파일(신규/수정/삭제) 목록 반환.
def get_changed_files(
    current_file_hashes: Dict[str, str],
    previous_metadata: Dict[str, Dict[str, Any]] # 이전 메타데이터
) -> Tuple[List[str], List[str], Set[str]]:
    new_files: List[str] = []
    modified_files: List[str] = [] # 수정된 파일

    # 이전 메타데이터의 key 값을 set으로 묶어서 이전 파일 경로에 저장
    previous_file_paths: Set[str] = set(previous_metadata.keys())
    # 현재 메타데이터의 key 값을 set으로 묶어서 현재 파일 경로에 저장
    current_file_paths: Set[str] = set(current_file_hashes.keys())

    # 현재 파일 상대 경로와 파일 해시값을 꺼내옴
    for rel_path_str, current_hash in current_file_hashes.items():
        # 상대 파일 경로가 이전 메타데이터에 존재하지 않는다면(즉, 새로운 파일이라면)
        if rel_path_str not in previous_metadata:
            # new_files(새로운 파일 리스트)에 추가
            new_files.append(rel_path_str)
        # 이전 메타데이터에 저장된 상대 파일 경로의 값인 hash가 현재 파일 해시값과 다르다면
        elif previous_metadata[rel_path_str].get("hash") != current_hash:
            # modified_files(수정된 파일 리스트)에 추가
            modified_files.append(rel_path_str)

    # 이전 파일 경로에서 현재 파일 경로를 제외하고 남은 경로들을 deleted_files_paths에 저장
    # 차집합 연산: 집합 A에는 속하지만 집합 B에는 속하지 않는 모든 원소들로 이루어진 새로운 집합을 반환
    deleted_files_paths: Set[str] = previous_file_paths - current_file_paths

    if new_files:
        logger.info(f"감지된 새 파일: {len(new_files)}")
    if modified_files:
        logger.info(f"감지된 수정 파일: {len(modified_files)}")
    if deleted_files_paths:
        logger.info(f"감지된 삭제 파일: {len(deleted_files_paths)}")

    return new_files, modified_files, deleted_files_paths

# 벡터 DB에 성공적으로 반영된 파일들의 메타데이터를 최신 해시값으로 업데이트
def update_metadata_after_processing(
    processed_relative_paths: List[str], # 처리된 상대 경로들
    current_files_hashes: Dict[str, str], # 현재 모든 파일의 정보
    existing_metadata: Dict[str, Dict[str, Any]] # 기존 메타데이터(업데이트 전)
) -> Dict[str, Dict[str, Any]]:

    updated_metadata = existing_metadata.copy()
    # 처리된 각 파일 경로에 대해 반복
    for rel_path_str in processed_relative_paths:
        # 이 파일이 현재 파일 시스템에 실제로 존재하는지 확인
        if rel_path_str in current_files_hashes:
            """
            현재 처리 중인 파일 경로(rel_path_str)를 key로
            '현재' 파일 시스템에서 가져온 이 파일의 가장
            최신 해시값으로 해당 파일의 정보를 업데이트
            """
            updated_metadata[rel_path_str] = {
                "hash": current_files_hashes[rel_path_str]
            }
    return updated_metadata

# 삭제된 파일들의 경로를 받아, 기존 메타데이터에서 해당 항목들을 제거한 새 딕셔너리 반환.
def remove_metadata_for_deleted_files(
    deleted_relative_paths: Set[str], # 삭제된 파일의 상대 경로
    existing_metadata: Dict[str, Dict[str, Any]] # 기존 메타데이터
) -> Dict[str, Dict[str, Any]]:
    updated_metadata = existing_metadata.copy()
    # 기존 메타데이터에서 삭제된 파일 경로 삭제
    for rel_path_str in deleted_relative_paths:
        del updated_metadata[rel_path_str]
    return updated_metadata