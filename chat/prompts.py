from .session import Message


def get_system_prompt() -> str:
    """Return the base instruction for the assistant."""
    return (
        "You are a helpful assistant. "
        "Reply in the same language as the user. "
        "Keep answers short and natural."
    )


def build_chat_prompt(messages: list[Message], limit: int = 6) -> str:
    """Build the prompt from the instruction and the recent messages."""
    recent = messages[-limit:]
    if recent and recent[0]["role"] == "assistant":
        recent = recent[1:]
    parts = [f"<|im_start|>system\n{get_system_prompt()}<|im_end|>\n"]
    for message in recent:
        role, content = message["role"], message["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)