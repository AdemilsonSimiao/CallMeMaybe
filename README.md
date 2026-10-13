# CallMeMaybe

Translate natural-language requests into structured function calls with a
small local language model and **constrained decoding**.

## Description

A traditional LLM answers "What is the sum of 40 and 2?" with text such as
"The sum is 42". Function calling asks for something different: the name of a
function and the arguments to call it with.

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": { "a": 40.0, "b": 2.0 }
}
```

A 0.6 billion parameter model (`Qwen/Qwen3-0.6B`) cannot be trusted to write
valid JSON on its own. This project does not ask it to. The program writes
the structure itself and lets the model decide only two things: **which
function to call** and **the value of each argument**. Every token the model
picks is restricted by a mask over its logits, so the model can only choose
tokens that keep the answer valid.

Everything runs locally. No external service is used after the model has been
downloaded once.

## Instructions

### Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- About 3 GB of free memory. The model runs on the CPU in `float32` unless a
  GPU is available.
- Internet access on the first run: the model (about 1.2 GB) is downloaded
  from Hugging Face and cached.

### Install

```bash
make install
```

This runs `uv sync`, which creates `.venv` and installs the locked
dependencies (`numpy`, `pydantic`, `torch`, `transformers`, and the
development tools `flake8` and `mypy`).

### Run

```bash
make run
```

which is the same as:

```bash
uv run python -m src
```

By default it reads `data/input/functions_definition.json` and
`data/input/function_calling_tests.json`, and writes
`data/output/function_calls.json` (the folder is created if it does not
exist).

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--functions_definition` | `data/input/functions_definition.json` | The available functions |
| `--input` | `data/input/function_calling_tests.json` | The prompts to translate |
| `--output` | `data/output/function_calls.json` | Where the result is written |

```bash
uv run python -m src \
  --functions_definition my_functions.json \
  --input my_prompts.json \
  --output results/out.json
```

### Makefile targets

| Target | What it does |
| --- | --- |
| `make install` | Installs the dependencies with `uv sync` |
| `make run` | Runs the program with the default paths |
| `make debug` | Runs the program under `pdb` |
| `make clean` | Removes `__pycache__` and `.mypy_cache` |
| `make lint` | Runs `flake8` and `mypy` with the required flags |
| `make lint-strict` | Runs `flake8` and `mypy --strict` |

### Input and output formats

`functions_definition.json` is a list of function definitions:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": { "a": { "type": "number" }, "b": { "type": "number" } },
    "returns": { "type": "number" }
  }
]
```

`function_calling_tests.json` is a list of prompts:

```json
[ { "prompt": "What is the sum of 2 and 3?" } ]
```

The output is a list with one object per prompt, with exactly three keys
(`prompt`, `name`, `parameters`) and no extra text.

Supported parameter types are `number` (written as a float), `integer`,
`boolean` and `string`.

## Project structure

```text
CallMeMaybe/
├── data/
│   └── input/
│       ├── functions_definition.json
│       └── function_calling_tests.json
├── llm_sdk/                  Provided model wrapper (not modified)
├── src/
│   ├── __init__.py
│   ├── __main__.py           Command line entry point
│   ├── decoder.py            Logit masking and token selection
│   ├── errors.py             Every exception the program raises on purpose
│   ├── generator.py          Prompt, function choice and argument values
│   ├── loader.py             Reads JSON files
│   ├── models.py             Pydantic models
│   ├── parser.py             Validates the input data
│   ├── vocab.py              Token ids to text and back
│   └── writer.py             Writes the output JSON
├── Makefile
├── pyproject.toml
├── uv.lock
└── README.md
```

Each file has one responsibility, and the exceptions live in one place
(`errors.py`) just like the data classes live in `models.py`.

## Algorithm explanation

### 1. Load and validate

`loader.py` reads the two input files and `parser.py` validates them with
Pydantic: the top level must be a list, every function needs a name,
description, parameters and return type, and function names must be unique.
Problems are reported as a one-line message (for example
`Invalid function definition #2: returns: Field required`) and the program
exits with code 1. It never crashes with a traceback.

### 2. Build the prompt

For each request, `generator.build_prompt` writes a plain text prompt with a
short worked example (using a function that is not available), the list of
functions with their parameters and descriptions, and the request. The prompt
ends with `Function:`, where the model must write a function name.

### 3. Constrained decoding

At every step the program asks the model for the logits of the next token (one
number per vocabulary entry), sets the logit of every token that is not
allowed to `-inf`, and takes the best one that remains:

```text
logits  +  mask(-inf where not allowed)  →  argmax  →  next token
```

The model cannot pick a masked token no matter how much it prefers it. This is
done in `Decoder.next_token` with `numpy`. What changes between steps is the
set of allowed tokens, which `Decoder` builds in three ways:

- `prefix_ids`: tokens whose text is a prefix of what still has to be written.
  It looks the prefixes up directly in the vocabulary instead of scanning the
  whole vocabulary, which keeps each step fast.
- `charset_ids`: tokens made only of some characters (for numbers).
- `text_ids`: tokens that can appear inside a text value.

### 4. Choose the function

The allowed tokens are the ones that continue the name of at least one
function. When one name is a prefix of another (`fn_add` and `fn_add_more`)
a newline is also allowed as a way to stop, so the shorter name can still be
chosen.

### 5. Write each argument by its type

After the name, the program writes `Arguments:` and then one line per
parameter, in the order of the definition:

| Type | Constraint |
| --- | --- |
| `number` | Only tokens that keep the text a valid number (`-1.5` is accepted, `1.2.3` is not). A newline is accepted only when the number is complete. |
| `integer` | The same, without a decimal part. |
| `boolean` | The model chooses between `true` and `false`. |
| `string` | Any token without control or broken characters, until the model writes the closing quote. |

A parameter with another type (for example `array`) cannot be generated: the
prompt is skipped and reported on `stderr`.

### 6. Write the result

`writer.py` turns the results into JSON with `json.dumps(..., allow_nan=False)`
and only then opens the file, so a failure never leaves a half-written file.

### The vocabulary

The vocabulary of Qwen is byte-level BPE: the file maps token strings to ids,
but spaces and line breaks are stored as special characters (`Ġ` is a space,
`Ċ` is a line break) and some characters are split across several tokens.
`vocab.py` undoes that mapping so the decoder can reason about the real text
of each token. Tokens that are only part of a character are marked and never
allowed inside a text.

## Design decisions

- **The program writes the structure, the model fills in the values.** The
  JSON is valid by construction, because the model never writes quotes,
  braces or commas. The model only decides names and values.
- **Constraints on tokens, not on text afterwards.** The model is never asked
  to retry. Invalid output is impossible rather than detected.
- **Strict validation of the inputs** with Pydantic, so malformed files are
  rejected with a clear message before the (slow) model is loaded.
- **A prompt that skips a failure.** If one prompt cannot be generated, it is
  reported and skipped, and the other prompts are still written.
- **Dependencies pointing inwards.** `models.py`, `errors.py`, `loader.py`
  and `vocab.py` depend on almost nothing, and `__main__.py` is the only file
  that knows all the others. The model is only imported after the input files
  are known to be valid.
- **No global state.** There are no module-level variables or constants: the
  default paths are argument defaults and lookup tables are built inside the
  objects that use them.

## Error handling

| Situation | Behaviour |
| --- | --- |
| Input file missing, unreadable, or invalid JSON | `Error: File not found: ...`, exit code 1 |
| Input file with the wrong structure | `Error: Invalid ...` with the field name, exit code 1 |
| Duplicate function names, empty function list | `Error: ...`, exit code 1 |
| Output folder missing | Created automatically |
| Output cannot be written (permissions, a file in the way) | `Error: Cannot write ...`, exit code 1 |
| A prompt that cannot be generated | `Skipped prompt ...` on `stderr`, the rest is written |
| Empty prompt list | An empty list is written |

## Performance analysis

Measured on Ubuntu 22.04 under WSL2, CPU only, `float32`, with the 11
prompts of `data/input/function_calling_tests.json`:

| Measure | Result |
| --- | --- |
| Function selection | 11 of 11 |
| Argument extraction | 10 of 11 (91%) |
| Total time | about 3 minutes 8 seconds (about 17 seconds per prompt) |

The one imperfect argument is a regex replacement: for "Replace all vowels
in 'Programming is fun' with asterisks" the model writes the replacement as
`" * "` (with spaces around the symbol) instead of `"*"`.

Almost all of the time is model inference. The provided SDK recomputes the
whole sequence every time it is asked for the logits of one token, so a
generation of about 30 tokens runs the full prompt about 30 times, and the
cost of each step grows with the length of the prompt, so every extra line
in the prompt makes every step slower. The fixed parts of the answer (the
labels, the quotes and the structure of the JSON) are written by the program
and never generated by the model, so they cost no model calls.

## Limitations

- The quality of the values is the quality of a 0.6B model. Plain requests
  (sums, greetings, string reversal, square roots) work reliably. Requests
  that need judgement, such as writing a regular expression or deciding what
  an "asterisk" replacement should look like, are the weak spot.
- Only the types `number`, `integer`, `boolean` and `string` are generated.
- The model is not told about parameters beyond their name, type and the
  function description, so good descriptions in the definitions file matter.

## Challenges

- **Prefix-sharing names.** Names such as `fn_add` and `fn_add_more` need an
  explicit way to stop, otherwise the shorter one could never be chosen.
- **The vocabulary.** Reading the BPE vocabulary as plain text gives wrong
  results (`Ġthe` is " the"); converting it correctly is what makes any
  character-level constraint possible.
- **Backslashes.** An early version forbade backslashes inside text values
  for safety, which made it impossible to write a regex such as `\d+`. It was
  unnecessary because the JSON writer escapes them, and removing it fixed the
  regex prompts.
- **Speed without a cache.** The SDK offers no way to reuse earlier
  computation, so the number of model calls and the size of the prompt are
  the only levers.

## Example usage

```bash
$ make run
Wrote 11 function calls to data/output/function_calls.json
```

Input `"Greet shrek"`:

```json
{
  "prompt": "Greet shrek",
  "name": "fn_greet",
  "parameters": { "name": "shrek" }
}
```

Input `"What is the sum of 265 and 345?"`:

```json
{
  "prompt": "What is the sum of 265 and 345?",
  "name": "fn_add_numbers",
  "parameters": { "a": 265.0, "b": 345.0 }
}
```

Input `"Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS"`:

```json
{
  "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "(\\d+)",
    "replacement": "NUMBERS"
  }
}
```

(In the JSON file the regex is written `(\\d+)`; the actual pattern is `(\d+)`.)

## Resources

### References

- Qwen3 model card, `Qwen/Qwen3-0.6B`, on Hugging Face
- Hugging Face Transformers documentation
- Pydantic documentation
- Radford et al., *Language Models are Unsupervised Multitask Learners*
  (2019), where the byte-level BPE vocabulary used by this model family comes
  from
- Willard and Louf, *Efficient Guided Generation for Large Language Models*
  (2023), on constraining generation with token masks

### Use of AI

An AIs assistants was used throughout this project: to discuss the
architecture, to review the source code and the local test scripts,
to diagnose problems found while running the model (for example the
backslash restriction and the prompt wording), and to draft this README. The
code was run and checked against the provided inputs, and the results above
come from real executions.
