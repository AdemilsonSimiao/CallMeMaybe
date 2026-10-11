import json
import os
from src.models import FunctionCall
from src.errors import WriteError


def write_output(path: str, calls: list[FunctionCall]) -> None:
    """Write the function calls to a JSON  file."""
    data = [call.model_dump() for call in calls]
    try:
        text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        message = f"Results cannot be written as JSON: {error}"
        raise WriteError(message) from error
    try:
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
            with open(path, "w", encoding="utf-8") as file:
                file.write(text + "\n")
    except OSError as error:
        raise WriteError(f"Cannot write {path}: {error.strerror}") from error
