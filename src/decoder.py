"""Constrained decoding: choose the next token among the allowed ones."""
import numpy as np
from collections.abc import Collection, Iterable
from typing import Protocol
from src.errors import DecodeError
from src.vocab import Vocab


class LogitsModel(Protocol):
    """Anything that scores the next token, such as Small_LLM_Model."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the raw logits for the token that follows `input_ids`."""


class Decoder:
    """Picks tokens from the model, never leaving the allowed set."""

    def __init__(self, model: LogitsModel, vocab: Vocab) -> None:
        """Prepare the decoder."""
        self._model = model
        self._vocab = vocab
        self._charsets: dict[str, frozenset[int]] = {}
        self._text_ids: frozenset[int] | None = None

    def next_token(self, ids: list[int], allowed: Collection[int]) -> int:
        """Choose the most likely token among the allowed ones."""
        logits = np.asarray(
            self._model.get_logits_from_input_ids(ids), dtype=np.float64
        )
        valid = [token_id for token_id in allowed
                 if 0 <= token_id < logits.size]
        if not valid:
            raise DecodeError("No allowed token to choose from")
        mask = np.full(logits.shape, -np.inf)
        mask[valid] = 0.0
        return int(np.argmax(logits + mask))

    def prefix_ids(self, texts: Iterable[str]) -> frozenset[int]:
        """Find the tokens that can start any of the given texts."""
        ids: set[int] = set()
        for text in texts:
            for end in range(1, len(text) + 1):
                token_id = self._vocab.id_of(text[:end])
                if token_id is not None:
                    ids.add(token_id)
        return frozenset(ids)

    def charset_ids(self, chars: str) -> frozenset[int]:
        """Find the tokens made only of the given characters."""
        if chars not in self._charsets:
            self._charsets[chars] = frozenset(
                token_id
                for token_id, text in self._vocab.items()
                if text and all(char in chars for char in text)
            )
        return self._charsets[chars]

    def text_ids(self) -> frozenset[int]:
        """Find the tokens that can appear inside a text value."""
        if self._text_ids is None:
            self._text_ids = frozenset(
                token_id
                for token_id, text in self._vocab.items()
                if text and all(
                    char >= " " and char not in "\ufffd" for char in text
                )
            )
        return self._text_ids
