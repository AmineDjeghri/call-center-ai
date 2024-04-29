from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from deepeval.models.base_model import DeepEvalBaseLLM
from fastapi import BackgroundTasks
from helpers.config import CONFIG
from helpers.logging import build_logger
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI
from models.call import CallStateModel, CallInitiateModel
from typing import Any
import hashlib
import pytest
import random
import string


_logger = build_logger(__name__)


@pytest.fixture
def random_text() -> str:
    text = "".join(random.choice(string.printable) for _ in range(100))
    return text


@pytest.fixture
def call_mock() -> CallStateModel:
    call = CallStateModel(
        initiate=CallInitiateModel(
            **CONFIG.workflow.default_initiate.model_dump(),
            phone_number="+33612345678",  # type: ignore
        )
    )
    return call


@pytest.fixture
def deepeval_model(cache: pytest.Cache) -> DeepEvalBaseLLM:
    model = DeepEvalAzureOpenAI(cache)
    return model


class DeepEvalAzureOpenAI(DeepEvalBaseLLM):
    """
    LangChain OpenAI LLM integration for DeepEval SDK.

    All calls are cached at best effort with Pytest implementation. Thread safe, can be used across processes.
    """

    _langchain_kwargs: dict[str, Any]
    _cache: pytest.Cache
    _model: BaseChatModel

    def __init__(self, cache: pytest.Cache):
        platform = CONFIG.llm.backup.azure_openai
        assert platform

        self._cache = cache
        self._langchain_kwargs = {
            # Repeatability
            "model_kwargs": {
                "seed": 42,
            },
            "temperature": 0,
            # Reliability
            "max_retries": 3,
            "timeout": 60,
            # Azure deployment
            "api_version": "2023-12-01-preview",
            "azure_deployment": platform.deployment,
            "azure_endpoint": platform.endpoint,
            "model": platform.model,
            # Authentication, either RBAC or API
            "api_key": platform.api_key.get_secret_value() if platform.api_key else None,  # type: ignore
            "azure_ad_token_provider": (
                get_bearer_token_provider(
                    DefaultAzureCredential(),
                    "https://cognitiveservices.azure.com/.default",
                )
                if not platform.api_key
                else None
            ),
        }
        self._model = AzureChatOpenAI(**self._langchain_kwargs)

    def load_model(self) -> BaseChatModel:
        return self._model

    def generate(self, prompt: str) -> str:
        cache_key = self._cache_key(prompt)

        # Try cache
        content: str = self._cache.get(cache_key, None)
        if content:
            return content

        # Try live
        model = self.load_model()
        res = model.invoke(prompt)
        content = res.content  # type: ignore

        # Update cache
        self._cache.set(cache_key, content)

        return content

    async def a_generate(self, prompt: str) -> str:
        cache_key = self._cache_key(prompt)

        # Try cache
        content: str = self._cache.get(cache_key, None)
        if content:
            return content

        # Try live
        model = self.load_model()
        res = await model.ainvoke(prompt)
        content = res.content  # type: ignore

        # Update cache
        self._cache.set(cache_key, content)

        return content

    def get_model_name(self) -> str:
        return "Azure OpenAI"

    def _cache_key(self, prompt: str) -> str:
        langchain_config = self._model._get_llm_string(**self._langchain_kwargs)
        suffix = hashlib.sha256(
            f"{langchain_config}-{prompt}".encode(),
            usedforsecurity=False,
        ).digest()  # Arguments contain secrets, so hash them
        return f"call-center-ai/{suffix}"


class BackgroundTasksMock(BackgroundTasks):
    def add_task(self, *args, **kwargs) -> None:
        _logger.info("add_task, ignoring")
