from __future__ import annotations

from tests.helpers import test_query_embedder


def pytest_configure() -> None:
    import app

    app.index.set_query_embedder(test_query_embedder)
    app.engine.index.set_query_embedder(test_query_embedder)
