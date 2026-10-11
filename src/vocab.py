"""Map token ids to readable text and back, using the vocabulary file."""
from collections.abc import ItemsView
from src.loader import LoadError, load_json


def _build_char_to_byte() -> dict[str, int]:
    """Build the table that undoes the byte-level BPE character mapping."""
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[str, int] = {}
    extra = 0
    for byte in range(256):
        if byte in keep:
            table[chr(byte)] = byte
        else:
            table[chr(256 + extra)] = byte
            extra += 1
    return table


def _to_text(token: str, char_to_byte: dict[str, int]) -> str:
    """Convert a raw vocabulary token into the text it stands for."""
    try:
        data = bytes(char_to_byte[char] for char in token)
    except KeyError:
        return token
    return data.decode("utf-8", errors="replace")


class Vocab:
    """Token vocabulary: maps ids to readable text and back."""

    def __init__(self, path: str) -> None:
        """Load the vocabulary file."""
        raw = load_json(path)
        if not isinstance(raw, dict) or not all(
            isinstance(token, str) and isinstance(token_id, int)
            for token, token_id in raw.items()
        ):
            raise LoadError(f"Invalid vocabulary format: {path}")
        char_to_byte = _build_char_to_byte()
        self._id_to_text: dict[int, str] = {
            token_id: _to_text(token, char_to_byte)
            for token, token_id in raw.items()
        }
        self._text_to_id: dict[str, int] = {}
        for token_id in sorted(self._id_to_text):
            self._text_to_id.setdefault(self._id_to_text[token_id], token_id)

    def __len__(self) -> int:
        """Return how many tokens the vocabulary knows."""
        return len(self._id_to_text)

    def text_of(self, token_id: int) -> str | None:
        """Return the text of a token."""
        return self._id_to_text.get(token_id)

    def id_of(self, text: str) -> int | None:
        """Return the id of the token whose text is exactly `text`."""
        return self._text_to_id.get(text)

    def items(self) -> ItemsView[int, str]:
        """Return every known (id, text) pair."""
        return self._id_to_text.items()
