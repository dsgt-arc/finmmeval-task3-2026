from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass
class ModelConfig:
    """Configuration for a model provider"""

    model_class_path: str
    env_key: str | None = None
    base_url: str | None = None
    requires_api_key: bool = True

    def load_model_class(self) -> type[BaseChatModel]:
        module_name, class_name = self.model_class_path.rsplit(".", 1)
        module = import_module(module_name)
        return getattr(module, class_name)


class Provider(StrEnum):
    """Supported LLM providers"""

    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    DEEPSEEK = "DeepSeek"
    ALIBABA = "Alibaba"
    ZHIPU = "ZhiPu"
    OLLAMA = "Ollama"
    FIREWORKS = "Fireworks"
    YIZHAN = "YiZhan"
    AIHUBMIX = "AiHubMix"
    GOOGLE = "Google"

    @property
    def config(self) -> ModelConfig:
        """Get the configuration for this provider"""
        PROVIDER_CONFIGS = {
            Provider.OPENAI: ModelConfig(
                model_class_path="langchain_openai.ChatOpenAI",
                env_key="OPENAI_API_KEY",
            ),
            Provider.ANTHROPIC: ModelConfig(
                model_class_path="langchain_anthropic.ChatAnthropic",
                env_key="ANTHROPIC_API_KEY",
            ),
            Provider.DEEPSEEK: ModelConfig(
                model_class_path="langchain_deepseek.ChatDeepSeek",
                env_key="DEEPSEEK_API_KEY",
            ),
            Provider.ALIBABA: ModelConfig(
                model_class_path="langchain_openai.ChatOpenAI",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                env_key="QWEN_API_KEY",
            ),
            Provider.ZHIPU: ModelConfig(
                model_class_path="langchain_openai.ChatOpenAI",
                base_url="https://open.bigmodel.cn/api/paas/v4",
                env_key="ZHIPU_API_KEY",
            ),
            Provider.OLLAMA: ModelConfig(
                model_class_path="langchain_ollama.ChatOllama",
                requires_api_key=False,
            ),
            Provider.FIREWORKS: ModelConfig(
                model_class_path="langchain_fireworks.ChatFireworks",
                env_key="FIREWORKS_API_KEY",
            ),
            Provider.YIZHAN: ModelConfig(
                model_class_path="langchain_openai.ChatOpenAI",
                env_key="YIZHAN_API_KEY",
                base_url="https://vip.yi-zhan.top/v1",
            ),
            Provider.AIHUBMIX: ModelConfig(
                model_class_path="langchain_openai.ChatOpenAI",
                env_key="AIHUBMIX_API_KEY",
                base_url="https://api.aihubmix.com/v1",
            ),
            Provider.GOOGLE: ModelConfig(
                model_class_path="langchain_google_genai.ChatGoogleGenerativeAI",
                env_key="GOOGLE_API_KEY",
            ),
        }
        return PROVIDER_CONFIGS[self]
