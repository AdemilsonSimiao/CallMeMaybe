"""Checks for src/parser.py. Run: uv run python -m tests.test_parser"""

from typing import Any

from src.loader import load_json
from src.parser import ParseError, parse_functions, parse_prompts

GOOD = {
    "name": "fn_add",
    "description": "Add two numbers",
    "parameters": {"a": {"type": "number"}},
    "returns": {"type": "number"},
}


def _expect_error(func: Any, raw: Any, part: str) -> None:
    try:
        func(raw)
    except ParseError as error:
        assert part in str(error), f"message was: {error}"
        return
    raise AssertionError(f"no ParseError for {raw!r}")


def test_real_input_files() -> None:
    functions = parse_functions(
        load_json("data/input/functions_definition.json")
    )
    prompts = parse_prompts(load_json("data/input/function_calling_tests.json"))
    assert len(functions) > 0
    assert len(prompts) > 0


def test_functions_valid() -> None:
    assert parse_functions([GOOD])[0].name == "fn_add"


def test_functions_not_a_list() -> None:
    _expect_error(parse_functions, {}, "JSON list")


def test_functions_empty_list() -> None:
    _expect_error(parse_functions, [], "empty")


def test_functions_item_not_an_object() -> None:
    _expect_error(parse_functions, ["x"], "#1")


def test_functions_missing_returns() -> None:
    bad = {k: v for k, v in GOOD.items() if k != "returns"}
    _expect_error(parse_functions, [bad], "returns")


def test_functions_wrong_parameter_type() -> None:
    bad = dict(GOOD, parameters={"a": {"type": 5}})
    _expect_error(parse_functions, [bad], "parameters.a.type")


def test_functions_duplicate_name() -> None:
    _expect_error(parse_functions, [GOOD, GOOD], "Duplicate")


def test_prompts_valid() -> None:
    assert parse_prompts([{"prompt": "hi"}])[0].prompt == "hi"


def test_prompts_empty_list_is_ok() -> None:
    assert parse_prompts([]) == []


def test_prompts_not_a_list() -> None:
    _expect_error(parse_prompts, {}, "JSON list")


def test_prompts_missing_field() -> None:
    _expect_error(parse_prompts, [{"x": 1}], "prompt")


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")