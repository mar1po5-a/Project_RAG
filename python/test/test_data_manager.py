# test_data_manager.py
import unittest
import os
import shutil
import json
from pathlib import Path
from data_manager import ( # data_manager.py가 같은 디렉토리에 있거나 PYTHONPATH에 있어야 함
    calculate_file_hash,
    load_metadata,
    save_metadata,
    scan_data_directory,
    get_changed_files,
    update_metadata_after_processing,
    remove_metadata_for_deleted_files,
    METADATA_FILE_PATH as ACTUAL_METADATA_FILE_PATH # 실제 메타데이터 파일 경로 백업
)

TEST_BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_BASE_DIR / "temp_test_data"
TEST_METADATA_FILE = TEST_BASE_DIR / "temp_test_manifest.json"

import data_manager
data_manager.METADATA_FILE_PATH = TEST_METADATA_FILE


class TestDataManagerSimplified(unittest.TestCase):

    def setUp(self):
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
        if TEST_METADATA_FILE.exists():
            TEST_METADATA_FILE.unlink()

    def tearDown(self):
        shutil.rmtree(TEST_DATA_DIR)
        if TEST_METADATA_FILE.exists():
            TEST_METADATA_FILE.unlink()

    def _create_file(self, relative_path_str: str, content: str):
        full_path = TEST_DATA_DIR / relative_path_str
        full_path.parent.mkdir(parents=True, exist_ok=True)  # 하위 디렉토리 생성
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return full_path

    def _assert_metadata_contains(self, metadata, expected_files_with_hashes):
        self.assertEqual(len(metadata), len(expected_files_with_hashes))
        for rel_path, expected_hash in expected_files_with_hashes.items():
            self.assertIn(rel_path, metadata)
            self.assertEqual(metadata[rel_path].get("hash"), expected_hash)

    def test_scan_single_file(self):
        """단일 파일 스캔 및 해시값 정확성 테스트"""
        print("\n--- test_scan_single_file ---")
        self._create_file("file1.txt", "Hello")
        current_hashes = scan_data_directory(TEST_DATA_DIR)

        self.assertEqual(len(current_hashes), 1)
        self.assertIn("file1.txt", current_hashes)
        self.assertEqual(current_hashes["file1.txt"], calculate_file_hash(TEST_DATA_DIR / "file1.txt"))

    def test_initial_run_creates_metadata(self):
        """첫 실행 시 모든 파일을 새 파일로 감지하고 메타데이터 생성 테스트"""
        print("\n--- test_initial_run_creates_metadata ---")
        self._create_file("new1.txt", "content1")
        self._create_file("new2.txt", "content2")

        current_hashes = scan_data_directory(TEST_DATA_DIR)
        prev_meta = load_metadata()  # 비어있어야 함
        new, mod, deleted = get_changed_files(current_hashes, prev_meta)

        self.assertCountEqual(new, ["new1.txt", "new2.txt"])
        self.assertEqual(len(mod), 0)
        self.assertEqual(len(deleted), 0)

        # 메타데이터 업데이트 및 저장
        updated_meta = update_metadata_after_processing(new, current_hashes, prev_meta)
        save_metadata(updated_meta)

        reloaded_meta = load_metadata()
        self._assert_metadata_contains(reloaded_meta, {
            "new1.txt": current_hashes["new1.txt"],
            "new2.txt": current_hashes["new2.txt"]
        })

    def test_add_new_file_to_existing_metadata(self):
        """기존 메타데이터에 새 파일 추가 시나리오 테스트"""
        print("\n--- test_add_new_file_to_existing_metadata ---")
        # 초기 상태: original.txt만 존재
        self._create_file("original.txt", "original content")
        initial_hashes = scan_data_directory(TEST_DATA_DIR)
        initial_meta = update_metadata_after_processing(list(initial_hashes.keys()), initial_hashes, {})
        save_metadata(initial_meta)

        # 변경: new_added.txt 추가
        self._create_file("new_added.txt", "newly added content")

        current_hashes_after_add = scan_data_directory(TEST_DATA_DIR)
        prev_meta_loaded = load_metadata()
        new, mod, deleted = get_changed_files(current_hashes_after_add, prev_meta_loaded)

        self.assertCountEqual(new, ["new_added.txt"])
        self.assertEqual(len(mod), 0)
        self.assertEqual(len(deleted), 0)

        # 메타데이터 업데이트
        meta_after_add = update_metadata_after_processing(new, current_hashes_after_add, prev_meta_loaded)
        save_metadata(meta_after_add)

        reloaded_meta = load_metadata()
        self._assert_metadata_contains(reloaded_meta, {
            "original.txt": initial_hashes["original.txt"],
            "new_added.txt": current_hashes_after_add["new_added.txt"]
        })

    def test_modify_existing_file(self):
        """기존 파일 수정 시나리오 테스트"""
        print("\n--- test_modify_existing_file ---")
        # 초기 상태
        self._create_file("to_modify.txt", "initial version")
        initial_hashes = scan_data_directory(TEST_DATA_DIR)
        initial_meta = update_metadata_after_processing(list(initial_hashes.keys()), initial_hashes, {})
        save_metadata(initial_meta)

        # 변경: to_modify.txt 내용 수정
        self._create_file("to_modify.txt", "MODIFIED version")  # 내용 변경

        current_hashes_after_modify = scan_data_directory(TEST_DATA_DIR)
        prev_meta_loaded = load_metadata()
        new, mod, deleted = get_changed_files(current_hashes_after_modify, prev_meta_loaded)

        self.assertEqual(len(new), 0)
        self.assertCountEqual(mod, ["to_modify.txt"])
        self.assertEqual(len(deleted), 0)

        # 메타데이터 업데이트
        meta_after_modify = update_metadata_after_processing(mod, current_hashes_after_modify, prev_meta_loaded)
        save_metadata(meta_after_modify)

        reloaded_meta = load_metadata()
        self._assert_metadata_contains(reloaded_meta, {
            "to_modify.txt": current_hashes_after_modify["to_modify.txt"]  # 수정된 해시로 업데이트되었는지 확인
        })
        self.assertNotEqual(initial_hashes["to_modify.txt"],
                            current_hashes_after_modify["to_modify.txt"])  # 해시가 바뀌었는지도 확인

    def test_delete_existing_file(self):
        """기존 파일 삭제 시나리오 테스트"""
        print("\n--- test_delete_existing_file ---")
        # 초기 상태
        self._create_file("file_A.txt", "content A")
        path_to_delete = self._create_file("file_to_delete.txt", "this will be deleted")
        initial_hashes = scan_data_directory(TEST_DATA_DIR)
        initial_meta = update_metadata_after_processing(list(initial_hashes.keys()), initial_hashes, {})
        save_metadata(initial_meta)

        # 변경: file_to_delete.txt 삭제
        path_to_delete.unlink()

        current_hashes_after_delete = scan_data_directory(TEST_DATA_DIR)  # 삭제된 파일은 스캔 안됨
        prev_meta_loaded = load_metadata()
        new, mod, deleted = get_changed_files(current_hashes_after_delete, prev_meta_loaded)

        self.assertEqual(len(new), 0)
        self.assertEqual(len(mod), 0)
        self.assertCountEqual(deleted, {"file_to_delete.txt"})

        # 메타데이터 업데이트 (삭제된 파일 정보 제거)
        meta_after_delete = remove_metadata_for_deleted_files(deleted, prev_meta_loaded)
        # 중요: update_metadata_after_processing은 새/수정된 파일용이므로 여기서는 호출 안 함
        save_metadata(meta_after_delete)

        reloaded_meta = load_metadata()
        self._assert_metadata_contains(reloaded_meta, {
            "file_A.txt": initial_hashes["file_A.txt"]
        })
        self.assertNotIn("file_to_delete.txt", reloaded_meta)

# ... (if __name__ == "__main__": 부분은 기존과 유사하게 사용하되, TestDataManagerSimplified 클래스를 사용) ...
# 예: unittest.main(verbosity=2, defaultTest='TestDataManagerSimplified')
# 또는 그냥 unittest.main() 호출 시 해당 파일 내 모든 Test* 클래스 실행