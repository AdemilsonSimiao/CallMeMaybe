from typing import Any
from pydantic import ValidationError
from src.models import FunctionDef, PromptItem
from src.errors import ParseError


def _first_error(error: ValidationError) -> str:
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"])
    if field:
        return f"{field}: {first['msg']}"
    return str(first["msg"])


def parse_functions(raw: Any) -> list[FunctionDef]:
    """Validate the function definitions."""
    if not isinstance(raw, list):
        raise ParseError("Function definitions must be a JSON list")
    if not raw:
        raise ParseError("Function definitions list is empty")
    functions: list[FunctionDef] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        try:
            function = FunctionDef.model_validate(item)
        except ValidationError as error:
            raise ParseError(
                f"Invalid function definition #{index + 1}: "
                f"{_first_error(error)}"
            ) from None
        if function.name in seen:
            raise ParseError(f"Duplicate function name: {function.name}")
        seen.add(function.name)
        functions.append(function)
    return functions


def parse_prompts(raw: Any) -> list[PromptItem]:
    """Validate the prompts to processed"""
    if not isinstance(raw, list):
        raise ParseError("Prompts must be a JSON list")
    prompts: list[PromptItem] = []
    for index, item in enumerate(raw):
        try:
            prompts.append(PromptItem.model_validate(item))
        except ValidationError as error:
            raise ParseError(
                f"Invalid prompt #{index + 1}: {_first_error(error)}"
            ) from None
    return prompts
