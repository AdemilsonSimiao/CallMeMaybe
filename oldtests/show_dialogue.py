"""Shows the whole dialogue between the program and the model.

Run all prompts: uv run python -m tests.show_dialogue
Run one prompt:  uv run python -m tests.show_dialogue 9
"""

import heapq
import sys
from collections.abc import Collection

from src.__main__ import Model, ModelTokenizer, parse_args
from src.decoder import Decoder
from src.generator import Generator
from src.loader import load_json
from src.models import FunctionDef, PromptItem
from src.parser import parse_functions, parse_prompts
from src.vocab import Vocab


class LoggingTokenizer(ModelTokenizer):
    """Prints every text the program writes to the model."""

    def encode(self, text: str) -> list[int]:
        shown = text if len(text) > 80 else repr(text)
        print(f"[program] {shown}")
        return super().encode(text)


class LoggingDecoder(Decoder):
    """Prints every token the model chooses, with its best rivals."""

    def next_token(self, ids: list[int], allowed: Collection[int]) -> int:
        logits = self._model.get_logits_from_input_ids(ids)
        valid = [i for i in allowed if 0 <= i < len(logits)]
        best = heapq.nlargest(3, valid, key=logits.__getitem__)
        top = "  ".join(
            f"{self._vocab.text_of(i)!r}={logits[i]:.1f}" for i in best
        )
        token_id = super().next_token(ids, allowed)
        text = self._vocab.text_of(token_id)
        print(f"[model]   {text!r}  top: {top}")
        return token_id


def show(
    model: Model,
    functions: list[FunctionDef],
    prompts: list[PromptItem],
    number: int | None,
) -> None:
    vocab = Vocab(model.get_path_to_vocab_file())
    decoder = LoggingDecoder(model, vocab)
    generator = Generator(decoder, vocab, LoggingTokenizer(model), functions)
    for index, item in enumerate(prompts, start=1):
        if number is not None and index != number:
            continue
        print(f"\n===== prompt {index}: {item.prompt}")
        call = generator.generate_call(item.prompt)
        print(f"===== result: {call.name} {call.parameters}")


def main() -> None:
    from llm_sdk import Small_LLM_Model
    args = parse_args([])
    functions = parse_functions(load_json(args.functions_definition))
    prompts = parse_prompts(load_json(args.input))
    number = int(sys.argv[1]) if len(sys.argv) > 1 else None
    show(Small_LLM_Model(), functions, prompts, number)


if __name__ == "__main__":
    main()