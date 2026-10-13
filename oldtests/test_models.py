"""Checks for src/models.py. Run: uv run python -m tests.test_models"""

from pydantic import ValidationError

from src.models import FunctionCall, FunctionDef, PromptItem


def test_function_def_valid() -> None:
    function = FunctionDef.model_validate({
        "name": "fn_add",
        "description": "Add two numbers",
        "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
        "returns": {"type": "number"},
    })
    assert function.name == "fn_add"
    assert function.parameters["a"].type == "number"
    assert function.returns.type == "number"


def test_function_def_missing_returns() -> None:
    try:
        FunctionDef.model_validate({
            "name": "fn_add", "description": "d", "parameters": {},
        })
    except ValidationError:
        return
    raise AssertionError("missing 'returns' was accepted")


def test_function_def_wrong_type() -> None:
    try:
        FunctionDef.model_validate({
            "name": 5, "description": "d", "parameters": {},
            "returns": {"type": "string"},
        })
    except ValidationError:
        return
    raise AssertionError("name=5 was accepted")


def test_prompt_item_valid() -> None:
    assert PromptItem.model_validate({"prompt": "hi"}).prompt == "hi"


def test_prompt_item_missing_field() -> None:
    try:
        PromptItem.model_validate({})
    except ValidationError:
        return
    raise AssertionError("empty prompt object was accepted")


def test_function_call_dump() -> None:
    call = FunctionCall(
        prompt="sum 2 and 3", name="fn_add", parameters={"a": 2.0, "b": 3.0}
    )
    assert call.model_dump() == {
        "prompt": "sum 2 and 3",
        "name": "fn_add",
        "parameters": {"a": 2.0, "b": 3.0},
    }


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")