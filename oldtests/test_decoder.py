"""Checks for src/decoder.py. Run: uv run python -m tests.test_decoder"""

import json
import os
import tempfile

from src.decoder import Decoder
from src.errors import DecodeError
from src.vocab import Vocab


class FakeModel:
    def __init__(self, logits: list[float]) -> None:
        self.logits = logits
        self.received: list[int] = []

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        self.received = list(input_ids)
        return self.logits


def _tokens() -> dict[str, int]:
    return {
        "f": 0, "fn": 1, "_": 2, "add": 3, "_add": 4, "greet": 5,
        "_greet": 6, "0": 7, "1": 8, ".": 9, "-": 10, "a": 11,
        "\"": 12, "\\": 13, "Ċ": 14, "Ã": 15, "Ġ": 16,
    }


def _decoder(logits: list[float], tmp: str) -> tuple[Decoder, FakeModel]:
    path = os.path.join(tmp, "vocab.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_tokens(), file)
    model = FakeModel(logits)
    return Decoder(model, Vocab(path)), model


def _flat(size: int = 17) -> list[float]:
    return [0.0] * size


def test_picks_best_allowed_token() -> None:
    logits = _flat()
    logits[5], logits[1], logits[2] = 9.0, 3.0, 2.0
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(logits, tmp)
        assert decoder.next_token([1, 2], {1, 2}) == 1


def test_best_token_overall_is_ignored_when_not_allowed() -> None:
    logits = _flat()
    logits[5] = 100.0
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(logits, tmp)
        assert decoder.next_token([1], {7, 8}) in {7, 8}


def test_negative_logits_still_work() -> None:
    logits = [-10.0] * 17
    logits[8] = -1.0
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(logits, tmp)
        assert decoder.next_token([1], {7, 8, 9}) == 8


def test_model_receives_the_ids() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, model = _decoder(_flat(), tmp)
        decoder.next_token([4, 5, 6], {1})
        assert model.received == [4, 5, 6]


def test_out_of_range_ids_are_ignored() -> None:
    logits = _flat()
    logits[3] = 1.0
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(logits, tmp)
        assert decoder.next_token([1], {3, 5000, -1}) == 3


def test_no_allowed_token_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        for allowed in (set(), {5000}):
            try:
                decoder.next_token([1], allowed)
            except DecodeError:
                continue
            raise AssertionError(f"accepted allowed={allowed!r}")


def test_prefix_ids_single_text() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert decoder.prefix_ids(["fn_add"]) == {0, 1}


def test_prefix_ids_several_texts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert decoder.prefix_ids(["_add", "_greet"]) == {2, 4, 6}


def test_prefix_ids_no_match() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert decoder.prefix_ids(["xyz"]) == frozenset()
        assert decoder.prefix_ids([]) == frozenset()


def test_charset_ids_numbers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert decoder.charset_ids("0123456789.-") == {7, 8, 9, 10}
        assert decoder.charset_ids("0123456789.-") == {7, 8, 9, 10}

def test_text_ids_excludes_unsafe_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert decoder.text_ids() == set(range(17)) - {14, 15}


def test_text_ids_allows_backslash() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        decoder, _ = _decoder(_flat(), tmp)
        assert 13 in decoder.text_ids()


def main() -> None:
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")


if __name__ == "__main__":
    main()