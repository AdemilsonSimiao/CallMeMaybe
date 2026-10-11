class CallMeMaybeError(Exception):
    """Base class for every error the program raises on purpose."""


class LoadError(CallMeMaybeError):
    """Read a JSON file and return its raw content."""


class ParseError(CallMeMaybeError):
    """Raised when input data does not match the expected structure."""


class WriteError(CallMeMaybeError):
    """Raised when the output file cannot be produced."""


class DecodeError(CallMeMaybeError):
    """Raised when no token can be chosen under the current constraints."""
