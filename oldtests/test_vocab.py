"""Checks for src/vocab.py. Run: uv run python -m tests.test_vocab"""
import json
import os
import tempfile

from src.loader import LoadError
from src.vocab import Vocab

TOKENS = {
    "the": 1,
    "Ġthe": 2,
    "Ċ": 3,
    "5": 4,
    "Ã©": 5,
    "Ã": 6,
    "Ġ": 7,
}


def _write_vocab(directory: str, content: object) -> str:
    path = os.path.join(directory, "vocab.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file)
    return path


def test_plain_token() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Vocab(_write_vocab(tmp, TOKENS))
        assert vocab.text_of(1) == "the"
        assert vocab.text_of(4) == "5"


def test_space_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Vocab(_write_vocab(tmp, TOKENS))
        assert vocab.text_of(2) == " the"
        assert vocab.text_of(7) == " "


def test_newline_marker() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert Vocab(_write_vocab(tmp, TOKENS)).text_of(3) == "\n"


def test_multibyte_character() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert Vocab(_write_vocab(tmp, TOKENS)).text_of(5) == "é"


def test_partial_utf8_is_marked() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        text = Vocab(_write_vocab(tmp, TOKENS)).text_of(6)
        assert text is not None and "\ufffd" in text


def test_id_of() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Vocab(_write_vocab(tmp, TOKENS))
        assert vocab.id_of(" the") == 2
        assert vocab.id_of("the") == 1
        assert vocab.id_of("nope") is None


def test_unknown_id() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        assert Vocab(_write_vocab(tmp, TOKENS)).text_of(999) is None


def test_len_and_items() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        vocab = Vocab(_write_vocab(tmp, TOKENS))
        assert len(vocab) == len(TOKENS)
        assert (2, " the") in vocab.items()


def test_invalid_format() -> None:
    for bad in ([1, 2], {"a": "x"}):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                Vocab(_write_vocab(tmp, bad))
            except LoadError:
                continue
            raise AssertionError(f"accepted bad vocabulary {bad!r}")


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"PASS {name}")