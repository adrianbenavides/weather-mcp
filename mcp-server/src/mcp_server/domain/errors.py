"""Domain exceptions."""


class WeatherError(Exception):
    """Weather domain error."""

    def __init__(self, message: str, location: str | None = None) -> None:
        """Initialize WeatherError.

        Args:
            message: Error message.
            location: Optional location context.
        """
        super().__init__(message)
        self.message = message
        self.location = location
