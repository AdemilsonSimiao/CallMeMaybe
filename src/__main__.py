"""Command line entry point: run with `python -m src`."""
import argparse
import sys
from typing import Any, Protocol
from src.decoder import Decoder
from src.errors import CallMeMaybeError, DecodeError
from src.generator import Generator
from src.loader import load_json
from src.models import FunctionCall, FunctionDef, PromptItem
from src.parser import parse_functions, parse_prompts
from src.vocab import Vocab
from src.writer import write_output


class Model(Protocol):
    """The parts of the language model that this program uses."""

    def encode(self, text: str) -> Any:
        """Return the token ids of `text` as a 2-D tensor."""

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return the raw logits for the token that follows `input_ids`."""

    def get_path_to_vocab_file(self) -> str:
        """Return the path of the vocabulary file."""


class ModelTokenizer:
    """Gives the generator a plain list of ids from the model's encode."""

    def __init__(self, model: Model) -> None:
        """Wrap the model."""
        self._model = model

    def encode(self, text: str) -> list[int]:
        """Return the token ids of `text`."""
        rows = self._model.encode(text)
        return [int(token_id) for token_id in rows[0].tolist()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Read the command line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural-language prompts into function calls.",
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="JSON file with the available functions",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="JSON file with the prompts to translate",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="JSON file where the function calls are written",
    )
    return parser.parse_args(argv)


def generate_all(
    model: Model, functions: list[FunctionDef], prompts: list[PromptItem]
) -> list[FunctionCall]:
    """Translate every prompt; prompts that fail are reported and skipped."""
    vocab = Vocab(model.get_path_to_vocab_file())
    decoder = Decoder(model, vocab)
    generator = Generator(decoder, vocab, ModelTokenizer(model), functions)
    calls: list[FunctionCall] = []
    for item in prompts:
        try:
            calls.append(generator.generate_call(item.prompt))
        except DecodeError as error:
            print(f"Skipped prompt {item.prompt!r}: {error}", file=sys.stderr)
    return calls


def main(argv: list[str] | None = None) -> int:
    """Run the whole program."""
    args = parse_args(argv)
    try:
        functions = parse_functions(load_json(args.functions_definition))
        prompts = parse_prompts(load_json(args.input))
        from llm_sdk import Small_LLM_Model
        calls = generate_all(Small_LLM_Model(), functions, prompts)
        write_output(args.output, calls)
    except CallMeMaybeError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(calls)} function calls to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
