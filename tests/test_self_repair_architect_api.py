import os
import unittest
from unittest.mock import patch

from core.self_repair import (
    _call_architect,
    _extract_responses_text,
    _extract_supported_max_tokens,
)


class _DummyMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _DummyChoice:
    def __init__(self, content: str) -> None:
        self.message = _DummyMessage(content)


class _DummyChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_DummyChoice(content)]


class _DummyResponsesPayload:
    def __init__(self, output_text: str | None = None, output=None) -> None:
        self.output_text = output_text
        self.output = output


class _DummyChatCompletions:
    def __init__(self, parent) -> None:
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._handle("chat", kwargs)


class _DummyChat:
    def __init__(self, parent) -> None:
        self.completions = _DummyChatCompletions(parent)


class _DummyResponses:
    def __init__(self, parent) -> None:
        self._parent = parent

    def create(self, **kwargs):
        return self._parent._handle("responses", kwargs)


class _DummyClient:
    def __init__(self, handler, include_responses: bool = True) -> None:
        self._handler = handler
        self.chat = _DummyChat(self)
        if include_responses:
            self.responses = _DummyResponses(self)
        self.calls = []

    def _handle(self, endpoint: str, kwargs):
        self.calls.append((endpoint, kwargs))
        return self._handler(endpoint, kwargs)


class ArchitectApiCompatibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_env = {
            "ARCHITECT_MAX_RETRIES": "1",
            "ARCHITECT_API_TIMEOUT_SECONDS": "5",
            "ARCHITECT_TOTAL_TIMEOUT_SECONDS": "5",
        }

    def test_auto_codex_prefers_responses_only(self) -> None:
        def handler(endpoint, _kwargs):
            if endpoint == "responses":
                return _DummyResponsesPayload(output_text="responses-ok")
            raise AssertionError("chat endpoint should not be called for codex auto mode")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            output = _call_architect(client, "gpt-5.1-codex", "hello")

        self.assertEqual(output, "responses-ok")
        self.assertEqual([name for name, _ in client.calls], ["responses"])

    def test_auto_non_codex_falls_back_when_chat_incompatible(self) -> None:
        def handler(endpoint, _kwargs):
            if endpoint == "chat":
                raise RuntimeError("The chatCompletion operation does not work with the specified model")
            if endpoint == "responses":
                return _DummyResponsesPayload(output_text="fallback-ok")
            raise AssertionError(f"unexpected endpoint {endpoint}")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            output = _call_architect(client, "gpt-5.1", "hello")

        self.assertEqual(output, "fallback-ok")
        self.assertEqual([name for name, _ in client.calls], ["chat", "responses"])

    def test_non_codex_uses_chat_only_when_chat_succeeds(self) -> None:
        def handler(endpoint, _kwargs):
            if endpoint == "chat":
                return _DummyChatResponse("chat-ok")
            raise AssertionError("responses endpoint should not be called when chat succeeds")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            output = _call_architect(client, "gpt-5.1", "hello")

        self.assertEqual(output, "chat-ok")
        self.assertEqual([name for name, _ in client.calls], ["chat"])

    def test_auto_codex_surfaces_responses_error(self) -> None:
        def handler(endpoint, _kwargs):
            if endpoint == "responses":
                raise RuntimeError("responses endpoint blocked")
            raise AssertionError("chat endpoint should not be called for codex auto mode")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            with self.assertRaises(RuntimeError) as ctx:
                _call_architect(client, "gpt-5.1-codex", "hello")

        self.assertIn("responses endpoint blocked", str(ctx.exception))
        self.assertEqual([name for name, _ in client.calls], ["responses"])

    def test_temperature_override_is_forwarded(self) -> None:
        captured = {}

        def handler(endpoint, kwargs):
            if endpoint == "chat":
                captured.update(kwargs)
                return _DummyChatResponse("ok")
            raise AssertionError("unexpected endpoint")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            _call_architect(client, "gpt-5.1", "hello", temperature_override=0.37)

        self.assertAlmostEqual(captured["temperature"], 0.37, places=6)

    def test_codex_omits_temperature_by_default(self) -> None:
        captured = {}

        def handler(endpoint, kwargs):
            if endpoint == "responses":
                captured.update(kwargs)
                return _DummyResponsesPayload(output_text="ok")
            raise AssertionError("unexpected endpoint")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            _call_architect(client, "gpt-5.1-codex", "hello", temperature_override=0.37)

        self.assertNotIn("temperature", captured)

    def test_responses_retries_without_temperature_when_not_supported(self) -> None:
        def handler(endpoint, kwargs):
            if endpoint == "chat":
                raise RuntimeError("The chatCompletion operation does not work with the specified model")
            if endpoint != "responses":
                raise AssertionError(f"unexpected endpoint {endpoint}")
            if "temperature" in kwargs:
                raise RuntimeError("Unsupported parameter: 'temperature' is not supported with this model.")
            return _DummyResponsesPayload(output_text="ok-no-temp")

        client = _DummyClient(handler)
        with patch.dict(os.environ, self.base_env, clear=False):
            output = _call_architect(client, "gpt-5.1", "hello", temperature_override=0.25)

        self.assertEqual(output, "ok-no-temp")
        self.assertEqual([name for name, _ in client.calls], ["chat", "responses", "responses"])
        self.assertIn("temperature", client.calls[1][1])
        self.assertNotIn("temperature", client.calls[2][1])

    def test_extract_responses_text_from_nested_output(self) -> None:
        payload = _DummyResponsesPayload(
            output_text=None,
            output=[
                {
                    "content": [
                        {"text": "line-1"},
                        {"text": "line-2"},
                    ]
                }
            ],
        )
        self.assertEqual(_extract_responses_text(payload), "line-1\nline-2")

    def test_extract_supported_max_tokens_patterns(self) -> None:
        self.assertEqual(_extract_supported_max_tokens("supports at most 2048 completion tokens"), 2048)
        self.assertEqual(_extract_supported_max_tokens("supports at most 1024 output tokens"), 1024)
        self.assertEqual(_extract_supported_max_tokens("maximum output tokens is 4096"), 4096)


if __name__ == "__main__":
    unittest.main()
