"""Checks for src/__main__.py. Run: uv run python -m tests.test_main"""

import contextlib
import io
import json
import os
import tempfile

from src.__main__ import generate_all, parse_args
from src.__main__ import main as run_program
from src.models import FunctionDef, PromptItem


class FakeRow:
    def tolist(self) -> list[int]:
        return [0]


class FakeModel:
    def __init__(self, script: list[int], vocab_path: str) -> None:
        self.script = script
        self.vocab_path = vocab_path
        self.calls = 0

    def encode(self, text: str) -> list[FakeRow]:
        return [FakeRow()]

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * 20
        if self.calls < len(self.script):
            logits[self.script[self.calls]] = 10.0
        self.calls += 1
        return logits

    def get_path_to_vocab_file(self) -> str:
        return self.vocab_path


def _tokens() -> dict[str, int]:
    return {
        "Ġfn": 0, "_": 1, "add": 2, "greet": 3, "flag": 4,
        "0": 5, "1": 6, "2": 7, "3": 8, ".": 9, "-": 10,
        "\"": 11, "Ċ": 12, "true": 13, "false": 14,
        "hello": 15, "world": 16, "5": 17, "s\"": 18, "_more": 19,
    }


def _write_json(directory: str, name: str, content: object) -> str:
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file)
    return path


def _function(name: str, kind: str) -> FunctionDef:
    return FunctionDef.model_validate({
        "name": name,
        "description": "d",
        "parameters": {"p": {"type": kind}},
        "returns": {"type": "string"},
    })


def _run_main(argv: list[str]) -> tuple[int, str]:
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        code = run_program(argv)
    return code, errors.getvalue()


def test_default_paths() -> None:
    args = parse_args([])
    assert args.functions_definition == "data/input/functions_definition.json"
    assert args.input == "data/input/function_calling_tests.json"
    assert args.output == "data/output/function_calls.json"


def test_custom_paths() -> None:
    args = parse_args(
        ["--functions_definition", "f.json", "--input", "i.json",
         "--output", "o.json"]
    )
    assert (args.functions_definition, args.input, args.output) == (
        "f.json", "i.json", "o.json"
    )


def test_main_reports_missing_input() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        code, errors = _run_main(["--input", os.path.join(tmp, "nope.json")])
        assert code == 1
        assert "Error: File not found" in errors


def test_main_reports_invalid_functions_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        bad = _write_json(tmp, "functions.json", {"not": "a list"})
        code, errors = _run_main(["--functions_definition", bad])
        assert code == 1
        assert "Error: " in errors


def test_generate_all_handles_every_prompt() -> None:
    script = [0, 1, 3, 15, 11, 0, 1, 3, 16, 11]
    functions = [_function("fn_greet", "string")]
    prompts = [PromptItem(prompt="one"), PromptItem(prompt="two")]
    with tempfile.TemporaryDirectory() as tmp:
        vocab_path = _write_json(tmp, "vocab.json", _tokens())
        calls = generate_all(FakeModel(script, vocab_path), functions, prompts)
        assert [call.prompt for call in calls] == ["one", "two"]
        assert [call.parameters for call in calls] == [
            {"p": "hello"}, {"p": "world"}
        ]


def test_generate_all_skips_failing_prompts() -> None:
    functions = [_function("fn_add", "array")]
    prompts = [PromptItem(prompt="one"), PromptItem(prompt="two")]
    errors = io.StringIO()
    with tempfile.TemporaryDirectory() as tmp:
        vocab_path = _write_json(tmp, "vocab.json", _tokens())
        model = FakeModel([0, 1, 2, 0, 1, 2], vocab_path)
        with contextlib.redirect_stderr(errors):
            calls = generate_all(model, functions, prompts)
    assert calls == []
    assert errors.getvalue().count("Skipped prompt") == 2


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()