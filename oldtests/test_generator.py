"""Checks for src/generator.py. Run: uv run python -m tests.test_generator"""

import json
import os
import tempfile

from src.decoder import Decoder
from src.errors import DecodeError
from src.generator import Generator, build_prompt
from src.models import FunctionDef
from src.vocab import Vocab


class FakeModel:
    """Prefers the next token of a script; real choices stay constrained."""

    def __init__(self, script: list[int]) -> None:
        self.script = script
        self.calls = 0

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        logits = [0.0] * 20
        if self.calls < len(self.script):
            logits[self.script[self.calls]] = 10.0
        self.calls += 1
        return logits


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]


def _tokens() -> dict[str, int]:
    return {
        "Ġfn": 0, "_": 1, "add": 2, "greet": 3, "flag": 4,
        "0": 5, "1": 6, "2": 7, "3": 8, ".": 9, "-": 10,
        "\"": 11, "Ċ": 12, "true": 13, "false": 14,
        "hello": 15, "world": 16, "5": 17, "s\"": 18, "_more": 19,
    }


def _function(name: str, params: dict[str, str]) -> FunctionDef:
    return FunctionDef.model_validate({
        "name": name,
        "description": f"Does {name}",
        "parameters": {key: {"type": kind} for key, kind in params.items()},
        "returns": {"type": "string"},
    })


def _generator(
    tmp: str, functions: list[FunctionDef], script: list[int], limit: int = 8
) -> Generator:
    path = os.path.join(tmp, "vocab.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_tokens(), file)
    vocab = Vocab(path)
    decoder = Decoder(FakeModel(script), vocab)
    return Generator(decoder, vocab, FakeTokenizer(), functions, limit)


def test_string_argument() -> None:
    functions = [_function("fn_greet", {"name": "string"})]
    with tempfile.TemporaryDirectory() as tmp:
        call = _generator(tmp, functions, [0, 1, 3, 15, 11]).generate_call("hi")
        assert call.name == "fn_greet"
        assert call.parameters == {"name": "hello"}
        assert call.prompt == "hi"


def test_string_cut_inside_a_token() -> None:
    functions = [_function("fn_greet", {"name": "string"})]
    with tempfile.TemporaryDirectory() as tmp:
        call = _generator(tmp, functions, [0, 1, 3, 15, 18]).generate_call("hi")
        assert call.parameters == {"name": "hellos"}


def test_two_number_arguments() -> None:
    functions = [_function("fn_add", {"a": "number", "b": "number"})]
    script = [0, 1, 2, 7, 12, 8, 12]
    with tempfile.TemporaryDirectory() as tmp:
        call = _generator(tmp, functions, script).generate_call("2 and 3")
        assert call.parameters == {"a": 2.0, "b": 3.0}
        assert isinstance(call.parameters["a"], float)


def test_negative_decimal_number() -> None:
    functions = [_function("fn_add", {"a": "number"})]
    script = [0, 1, 2, 10, 6, 9, 17, 12]
    with tempfile.TemporaryDirectory() as tmp:
        call = _generator(tmp, functions, script).generate_call("x")
        assert call.parameters == {"a": -1.5}


def test_integer_argument() -> None:
    functions = [_function("fn_add", {"a": "integer"})]
    script = [0, 1, 2, 7, 6, 12]
    with tempfile.TemporaryDirectory() as tmp:
        call = _generator(tmp, functions, script).generate_call("x")
        assert call.parameters == {"a": 21}
        assert type(call.parameters["a"]) is int


def test_boolean_arguments() -> None:
    functions = [_function("fn_flag", {"on": "boolean"})]
    with tempfile.TemporaryDirectory() as tmp:
        yes = _generator(tmp, functions, [0, 1, 4, 13]).generate_call("x")
        no = _generator(tmp, functions, [0, 1, 4, 14]).generate_call("x")
        assert yes.parameters == {"on": True}
        assert no.parameters == {"on": False}


def test_name_that_is_prefix_of_another() -> None:
    functions = [_function("fn_add", {}), _function("fn_add_more", {})]
    with tempfile.TemporaryDirectory() as tmp:
        short = _generator(tmp, functions, [0, 1, 2, 12]).generate_call("x")
        long = _generator(tmp, functions, [0, 1, 2, 19]).generate_call("x")
        assert short.name == "fn_add"
        assert long.name == "fn_add_more"


def test_invalid_choice_is_overruled() -> None:
    functions = [_function("fn_add", {"a": "number"})]
    with tempfile.TemporaryDirectory() as tmp:
        generator = _generator(tmp, functions, [0, 1, 2, 15], limit=4)
        call = generator.generate_call("x")
        assert call.parameters == {"a": 0.0}


def test_unsupported_type() -> None:
    functions = [_function("fn_add", {"a": "array"})]
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _generator(tmp, functions, [0, 1, 2]).generate_call("x")
        except DecodeError:
            return
        raise AssertionError("accepted an unsupported type")


def test_build_prompt() -> None:
    functions = [_function("fn_add", {"a": "number", "b": "number"})]
    prompt = build_prompt(functions, "sum 2 and 3")
    assert "fn_add(a: number, b: number): Does fn_add" in prompt
    assert "Request: sum 2 and 3" in prompt
    assert prompt.endswith("Function:")
    assert prompt.count("Function:") == 2


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()