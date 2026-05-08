from __future__ import annotations


class ToolError(Exception):
    def __init__(self, code: str, message: str, *, recoverable: bool = True):
        super().__init__(message)
        self.code = code
        self.message = message
        self.recoverable = recoverable


class InvalidInputError(ToolError):
    def __init__(self, message: str, *, code: str = "INVALID_INPUT"):
        super().__init__(code, message, recoverable=True)

