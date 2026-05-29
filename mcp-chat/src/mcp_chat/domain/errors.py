"""Domain exceptions."""


class ConversationError(Exception):
    """Conversation domain error."""

    def __init__(self, message: str) -> None:
        """Initialize ConversationError.

        Args:
            message: Error message.
        """
        super().__init__(message)
        self.message = message
