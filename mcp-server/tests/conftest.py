import pytest

pytest_plugins = ()


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as an integration test hitting real services")


@pytest.fixture
def asyncio_mode() -> str:
    return "auto"
