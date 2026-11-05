"""Shared pytest fixtures.

Coming from JUnit: a fixture is roughly @Before, but instead of mutating shared
state you *request* it by naming it as a test function argument. `autouse=True`
makes one apply to every test without being requested.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _no_real_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip provider credentials from the environment for every test.

    This is a safety net, not a nicety. Without it, one forgotten mock means a
    test suite that calls a real API: slow, flaky, and billed. Any test that
    genuinely needs a key should set it explicitly via monkeypatch.
    """
    for key in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "LANGSMITH_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def fake_llm() -> MagicMock:
    """A stand-in for a chat model that returns canned text.

    Deliberately provider-agnostic. Once you have met LangChain's own test
    doubles (`langchain_core.language_models.fake_chat_models.FakeListChatModel`
    and `GenericFakeChatModel`), prefer those for chain tests — they implement
    the real Runnable interface, so they work inside a piped chain. Use this one
    for plain unit tests where you only care that something was called.
    """
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="canned response")
    return llm
