"""Interactive terminal chat."""

from .session import Session
from .text_generator import Reply, TextGenerator


class ChatSession:
    """Terminal chat where the model answers freely, with no constraints."""

    def __init__(self, generator: TextGenerator, session: Session) -> None:
        """Connect the generator to the conversation."""
        self._generator = generator
        self._session = session

    def run(self) -> None:
        """Read user messages until the user exits."""
        print("CallMeMaybe Chat (free generation, no constraints)")
        print("Type 'exit' to quit or '/clear' to forget the conversation.")
        while True:
            try:
                message = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nChat stopped.")
                break
            if message.lower() in {"exit", "quit"}:
                break
            if message == "/clear":
                self._session.clear()
                print("Conversation cleared.")
                continue
            if not message:
                continue
            reply = self.handle_message(message)
            how = "ended on its own" if reply.finished else "cut at the limit"
            print(f"\nAssistant: {reply.text}")
            print(f"  [{reply.tokens} tokens, {how}]")

    def handle_message(self, message: str) -> Reply:
        """Generate the reply to one user message and remember both."""
        self._session.add_user_message(message)
        reply = self._generator.generate_response(self._session.messages)
        self._session.add_assistant_message(reply.text)
        return reply
