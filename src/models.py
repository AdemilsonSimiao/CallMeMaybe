"""Pydantic models used across the project."""
from typing import Any
from pydantic import BaseModel


class ParameterDef(BaseModel):
    """Type of one function paramenter, such as"""
    type: str


class ReturnDef(BaseModel):
    """Type of the value a function returns."""
    type: str


class FunctionDef(BaseModel):
    """One function avaliable to the LLM, from definitions file."""
    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ReturnDef


class PromptItem(BaseModel):
    """One natural-language request from the test file"""
    prompt: str


class FunctionCall(BaseModel):
    """Chosen function and its typed arguments for one prompt."""
    prompt: str
    name: str
    parameters: dict[str, Any]
