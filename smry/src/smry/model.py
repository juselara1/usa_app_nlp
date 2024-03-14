from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, TypeVar, Generic
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

class OpenAILLMConfig(BaseModel):
    model: str
    api_key: str
    temperature: float
    frequency_penalty: float

class OpenAIEmbeddingsConfig(BaseModel):
    model: str
    api_key: str

M = TypeVar('M', BaseChatModel, Embeddings)
C = TypeVar('C', OpenAILLMConfig, OpenAIEmbeddingsConfig)

class AbstractModelBuilder(ABC, Generic[M, C]):
    model: M
    config: C

    def set_config(self, config: C) -> "AbstractModelBuilder[M, C]":
        self.config = config
        return self

    def get_model(self) -> M:
        return self.model

    @abstractmethod
    def build(self) -> "AbstractModelBuilder[M, C]":
        ...

class OpenAILLMBuilder(AbstractModelBuilder[BaseChatModel, OpenAILLMConfig]):
    def build(self) -> "LLMBuilder":
        self.model = ChatOpenAI(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                model_kwargs={
                    "frequency_penalty": self.config.frequency_penalty,
                    },
                )
        return self

class OpenAIEmbeddingsBuilder(AbstractModelBuilder[Embeddings, OpenAIEmbeddingsConfig]):
    def build(self) -> "OpenAIEmbeddingsBuilder":
        self.model = OpenAIEmbeddings(
                model=self.config.model,
                api_key=self.config.api_key
                )
        return self
