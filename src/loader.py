import json
from typing import Any
from src.errors import LoadError


def load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise LoadError(f"File not found: {path}") from None
    except json.JSONDecodeError as error:
        raise LoadError(f"Invalid JSON in {path}: {error}") from error
    except UnicodeDecodeError:
        raise LoadError(f"File is not valid UTF-8 text: {path}") from None
    except OSError as error:
        raise LoadError(f"Cannot read {path}: {error.strerror}") from error
