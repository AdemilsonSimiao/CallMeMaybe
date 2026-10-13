from typing import Literal, TypedDict


class Message(TypedDict):
    """One message of the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str


class Session:
    """Stores the messages exchanged during a chat."""
    def __init__(self) -> None:
        """Start an empty conversation."""
        self.messages: list[Message] = []

    def add_user_message(self, text: str) -> None:
        """Store a message written by the user."""
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        """Store a reply written by the assistant."""
        self.messages.append({"role": "assistant", "content": text})

    def clear(self) -> None:
        """Forget the whole conversation."""
        self.messages.clear()