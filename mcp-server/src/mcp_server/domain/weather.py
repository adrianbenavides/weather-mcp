"""Weather domain models."""

from pydantic import BaseModel, ConfigDict


class Coordinates(BaseModel):
    """Geographic coordinates (latitude, longitude).

    Frozen: immutable value object.
    """

    latitude: float
    longitude: float

    model_config = ConfigDict(frozen=True)


class WeatherData(BaseModel):
    """Current weather data at a location.

    Frozen: immutable value object.
    """

    temperature_c: float
    wind_speed_kmh: float
    conditions: str
    location_name: str

    model_config = ConfigDict(frozen=True)
