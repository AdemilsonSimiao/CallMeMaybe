from src.vocab import Vocab
from llm_sdk.__init__ import Small_LLM_Model
from .chat import ChatSession
from .session import Session
from .text_generator import TextGenerator


def build_chat() -> ChatSession:
    "Create and configure the interactive chat."""
    generator = TextGenerator(model=Small_LLM_Model())
    return ChatSession(generator=generator, session=Session())