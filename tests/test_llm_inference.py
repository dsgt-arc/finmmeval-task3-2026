"""Tests for the LLM inference helpers."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "decision_making"))

from llm.inference import LLMConfig, get_model
from llm.provider import ModelConfig


class _StubChatModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_model_uses_loaded_model_class(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(ModelConfig, "load_model_class", lambda self: _StubChatModel)

    model = get_model(LLMConfig(provider="OpenAI", model="gpt-4o-mini", temperature=0.2))

    assert isinstance(model, _StubChatModel)
    assert model.kwargs["model"] == "gpt-4o-mini"
    assert model.kwargs["api_key"] == "test-key"
    assert model.kwargs["temperature"] == 0.2
