import os
import hashlib
import time


class FileRegistry:
    def __init__(self):
        self.files = {}

    def get_hash(self, file_path):
        if not os.path.exists(file_path):
            return None

        # Retry mechanism for Windows file lock
        for _ in range(5):
            try:
                hasher = hashlib.sha256()
                with open(file_path, "rb") as f:
                    hasher.update(f.read())
                return hasher.hexdigest()
            except PermissionError:
                time.sleep(0.5)

        return None

    def needs_update(self, file_path):
        file_hash = self.get_hash(file_path)

        if file_hash is None:
            return False

        if file_path not in self.files:
            self.files[file_path] = file_hash
            return True

        if self.files[file_path] != file_hash:
            self.files[file_path] = file_hash
            return True

        return False
