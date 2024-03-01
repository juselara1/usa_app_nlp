from abc import ABC, abstractmethod
from enum import Enum
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        MessagesPlaceholder,
        ChatPromptTemplate
        )
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain.chains.base import Chain
from pydantic import BaseModel

class AbstractChainBuilder(ABC):
    model: BaseChatModel
    prompt: BasePromptTemplate
    memory: BaseMemory
    chain: Chain

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        ...

    @abstractmethod
    def get_prompt(self) -> BasePromptTemplate:
        ...

    @abstractmethod
    def get_memory(self) -> BaseMemory:
        ...

    def get_chain(self) -> Chain:
        return self.chain

    def build(self) -> "AbstractChainBuilder":
        model = self.get_model()
        prompt = self.get_prompt()
        memory = self.get_memory()
        self.chain = LLMChain(
                llm=model,
                prompt=prompt,
                memory=memory
                )
        return self

class ModelEnum(Enum):
    GPT35 = "gpt-3.5-turbo-0125"
    GPT4 = "gpt-4-turbo-preview"

class OpenAIConfig(BaseModel):
    api_key: str
    model: ModelEnum
    temperature: float
    system_prompt: str

class OpenAIChainBuilder(AbstractChainBuilder):
    def __init__(self, config: OpenAIConfig):
        self.config = config

    def get_model(self) -> BaseChatModel:
        return ChatOpenAI(
                model_name=self.config.model.value,
                openai_api_key=self.config.api_key,
                temperature=self.config.temperature
                )

    def get_prompt(self) -> BasePromptTemplate:
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.config.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{message}")
            ])

    def get_memory(self) -> BaseMemory:
        return ConversationBufferMemory(memory_key="history", return_messages=True)
