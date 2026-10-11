"""Turn a natural-language request into a function call."""
import re
from typing import Protocol
from src.decoder import Decoder
from src.errors import DecodeError
from src.models import FunctionCall, FunctionDef
from src.vocab import Vocab


class Tokenizer(Protocol):
    """Anything that turns text into token ids."""

    def encode(self, text: str) -> list[int]:
        """Return the token ids of `text`."""


def build_prompt(functions: list[FunctionDef], request: str) -> str:
    """Write the text that asks the model to pick a function."""
    lines = [
        "Choose the function that fulfills the request and write its "
        "arguments.",
        "",
        "Write each value exactly as it must be used, with no extra spaces.",
        "",
        "Example (this function is not available):",
        "Request: Replace every digit in 'room 42' with the symbol #",
        "Function: fn_example",
        "Arguments:",
        'text: "room 42"',
        'pattern: "\\d"',
        'replacement: "#"',
        "",
        "Functions:",
    ]
    for function in functions:
        params = ", ".join(
            f"{name}: {param.type}"
            for name, param in function.parameters.items()
        )
        lines.append(f"- {function.name}({params}): {function.description}")
    lines += ["", f"Request: {request}", "Function:"]
    return "\n".join(lines)


def _is_number(text: str, integer: bool, complete: bool) -> bool:
    """Tell if `text` is a number, or the start of one when not complete."""
    if integer:
        pattern = r"-?\d+" if complete else r"-?\d*"
    else:
        pattern = r"-?\d+(\.\d+)?" if complete else r"-?\d*(\.\d*)?"
    return re.fullmatch(pattern, text) is not None


class Generator:
    """Args:
            decoder: Chooses each token under the current constraint.
            vocab: Vocabulary used to read the chosen tokens.
            tokenizer: Turns the prompt text into token ids.
            functions: The functions the model may choose from.
            max_tokens: Most tokens spent on a single argument.
        """

    def __init__(
        self,
        decoder: Decoder,
        vocab: Vocab,
        tokenizer: Tokenizer,
        functions: list[FunctionDef],
        max_tokens: int = 64,
    ) -> None:

        self._decoder = decoder
        self._vocab = vocab
        self._tokenizer = tokenizer
        self._functions = functions
        self._by_name = {function.name: function for function in functions}
        self._max_tokens = max_tokens
        self._newline = vocab.id_of("\n")

    def generate_call(self, prompt: str) -> FunctionCall:
        """Translate one request into a function call."""
        ids = self._tokenizer.encode(build_prompt(self._functions, prompt))
        options = [" " + function.name for function in self._functions]
        name = self._choose(ids, options).lstrip()
        function = self._by_name[name]
        ids.extend(self._tokenizer.encode("\nArguments:\n"))
        parameters: dict[str, int | float | bool | str] = {}
        for param_name, param in function.parameters.items():
            is_text = param.type == "string"
            label = f'{param_name}: "' if is_text else f"{param_name}: "
            ids.extend(self._tokenizer.encode(label))
            parameters[param_name] = self._generate_value(ids, param.type)
            ids.extend(self._tokenizer.encode('"\n' if is_text else "\n"))
        return FunctionCall(prompt=prompt, name=name, parameters=parameters)

    def _text(self, token_id: int) -> str:
        """Return the text of a token, or nothing if it is unknown."""
        return self._vocab.text_of(token_id) or ""

    def _choose(self, ids: list[int], options: list[str]) -> str:
        """Let the model pick exactly one of the options."""
        text = ""
        while True:
            remaining = [
                option[len(text):]
                for option in options
                if option.startswith(text)
            ]
            allowed = set(self._decoder.prefix_ids(r for r in remaining if r))
            stop_id: int | None = None
            if "" in remaining:
                if len(remaining) == 1:
                    return text
                stop_id = self._newline
                if stop_id is not None:
                    allowed.add(stop_id)
            token_id = self._decoder.next_token(ids, allowed)
            if token_id == stop_id:
                return text
            text += self._text(token_id)
            ids.append(token_id)

    def _generate_value(
        self, ids: list[int], type_name: str
    ) -> int | float | bool | str:
        """Write one argument according to its type."""
        if type_name == "number":
            return self._generate_number(ids, False)
        if type_name == "integer":
            return self._generate_number(ids, True)
        if type_name == "boolean":
            return self._choose(ids, ["true", "false"]) == "true"
        if type_name == "string":
            return self._generate_string(ids)
        raise DecodeError(f"Unsupported parameter type: {type_name}")

    def _generate_number(self, ids: list[int], integer: bool) -> int | float:
        """Write a number; only tokens that keep it a number are allowed."""
        text = ""
        charset = self._decoder.charset_ids("0123456789.-")
        for _ in range(self._max_tokens):
            allowed = {
                token_id
                for token_id in charset
                if _is_number(text + self._text(token_id), integer, False)
            }
            stop_id: int | None = None
            if _is_number(text, integer, True) and self._newline is not None:
                stop_id = self._newline
                allowed.add(stop_id)
            token_id = self._decoder.next_token(ids, allowed)
            if token_id == stop_id:
                break
            text += self._text(token_id)
            ids.append(token_id)
        if not _is_number(text, integer, True):
            raise DecodeError(f"Could not read a number, got {text!r}")
        return int(text) if integer else float(text)

    def _generate_string(self, ids: list[int]) -> str:
        """Write a text up to the closing quote."""
        text = ""
        allowed = self._decoder.text_ids()
        for _ in range(self._max_tokens):
            token_id = self._decoder.next_token(ids, allowed)
            piece = self._text(token_id)
            if '"' in piece:
                return text + piece.split('"', 1)[0]
            text += piece
            ids.append(token_id)
        return text
