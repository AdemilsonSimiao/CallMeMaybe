"""Checks for src/loader.py. Run: uv run python -m tests.test_loader"""

import os
import tempfile

from src.loader import LoadError, load_json


def _write(directory: str, name: str, content: bytes) -> str:
    path = os.path.join(directory, name)
    with open(path, "wb") as file:
        file.write(content)
    return path


def _expect_load_error(path: str, part: str) -> None:
    try:
        load_json(path)
    except LoadError as error:
        assert part in str(error), f"message was: {error}"
        return
    raise AssertionError(f"no LoadError for {path}")


def test_valid_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "ok.json", b'[{"a": 1}]')
        assert load_json(path) == [{"a": 1}]


def test_missing_file() -> None:
    _expect_load_error("/tmp/does_not_exist_xyz.json", "not found")


def test_invalid_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _expect_load_error(_write(tmp, "bad.json", b"{bad"), "Invalid JSON")


def test_empty_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _expect_load_error(_write(tmp, "empty.json", b""), "Invalid JSON")


def test_invalid_utf8() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "bin.json", bytes([255, 254, 0]))
        _expect_load_error(path, "UTF-8")


def test_directory_instead_of_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _expect_load_error(tmp, "Cannot read")


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")