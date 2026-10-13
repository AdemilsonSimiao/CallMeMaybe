from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol
import numpy as np
from .prompts import build_chat_prompt
from .session import Message


class TextModel(Protocol):
    """The parts of the model that the chat uses."""

    def encode(self, text: str) -> Any:
        """Return the token ids of `text` as a 2-D tensor."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the raw logits for the next token."""

    def decode(self, ids: list[int]) -> str:
        """Turn token ids back into text."""


class Reply(NamedTuple):
    """What the model wrote and how it stopped."""

    text: str
    tokens: int
    finished: bool


class TextGenerator:
    """Generate conversational replies, always taking the likeliest token."""

    def __init__(self, model: TextModel, max_tokens: int = 64) -> None:
        """Prepare the generator."""
        self._model = model
        self._max_tokens = max_tokens
        self._stop_ids = self._find_ids(["<|im_end|>", "<|endoftext|>"])

    def generate_response(self, messages: list[Message]) -> Reply:
        """Write the assistant's reply to the last message."""
        prompt = build_chat_prompt(messages)
        ids = [int(i) for i in self._model.encode(prompt)[0].tolist()]
        answer: list[int] = []
        finished = False
        for _ in range(self._max_tokens):
            logits = self._model.get_logits_from_input_ids(ids)
            token_id = int(np.argmax(logits))
            if token_id in self._stop_ids:
                finished = True
                break
            answer.append(token_id)
            ids.append(token_id)
        text = self._model.decode(answer).strip()
        return Reply(text, len(answer), finished)

    def _find_ids(self, names: Iterable[str]) -> set[int]:
        """Find the ids of special tokens such as the end of a message."""
        ids: set[int] = set()
        for name in names:
            encoded = self._model.encode(name)[0].tolist()
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
        return ids
