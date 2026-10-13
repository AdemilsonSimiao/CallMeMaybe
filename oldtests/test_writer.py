"""Checks for src/writer.py. Run: uv run python -m tests.test_writer"""

import json
import os
import tempfile

from src.models import FunctionCall
from src.writer import WriteError, write_output


def _call() -> FunctionCall:
    return FunctionCall(
        prompt="What is the sum of 2 and 3?",
        name="fn_add_numbers",
        parameters={"a": 2.0, "b": 3.0},
    )


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def _expect_write_error(path: str, calls: list[FunctionCall]) -> None:
    try:
        write_output(path, calls)
    except WriteError:
        return
    raise AssertionError(f"no WriteError for {path}")


def test_writes_expected_structure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        write_output(path, [_call()])
        assert json.loads(_read(path)) == [{
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2.0, "b": 3.0},
        }]


def test_keeps_float_format() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        write_output(path, [_call()])
        assert '"a": 2.0' in _read(path)


def test_creates_missing_folders() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data", "output", "out.json")
        write_output(path, [_call()])
        assert os.path.isfile(path)


def test_empty_list() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        write_output(path, [])
        assert json.loads(_read(path)) == []


def test_overwrites_existing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        write_output(path, [_call(), _call()])
        write_output(path, [_call()])
        assert len(json.loads(_read(path))) == 1


def test_keeps_accents() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        call = FunctionCall(
            prompt="Diga olá", name="fn_greet", parameters={"name": "José"}
        )
        write_output(path, [call])
        assert "José" in _read(path)
        assert json.loads(_read(path))[0]["parameters"]["name"] == "José"


def test_nan_is_rejected_and_nothing_is_written() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "out.json")
        call = FunctionCall(
            prompt="p", name="fn", parameters={"a": float("nan")}
        )
        _expect_write_error(path, [call])
        assert not os.path.exists(path)


def test_directory_instead_of_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _expect_write_error(tmp, [_call()])


def test_parent_is_a_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        blocker = os.path.join(tmp, "blocker")
        with open(blocker, "w", encoding="utf-8") as file:
            file.write("x")
        _expect_write_error(os.path.join(blocker, "out.json"), [_call()])


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()