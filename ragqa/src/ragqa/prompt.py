from typing import TypeVar, Generic, Self
from abc import ABC, abstractmethod
from pydantic import BaseModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain.prompts import (
        ChatPromptTemplate, SystemMessagePromptTemplate,
        HumanMessagePromptTemplate
        )

T = TypeVar('T')
class AbstractPromptBuilder(ABC, Generic[T]):
    config: T
    prompt: BasePromptTemplate

    def set_config(self, config: T) -> Self:
        self.config = config
        return self

    def get_prompt(self) -> BasePromptTemplate:
        return self.prompt

    @abstractmethod
    def build(self) -> Self:
        ...

class RagPromptConfig(BaseModel):
    system_prompt: str
    template: str

class RagPromptBuilder(AbstractPromptBuilder[RagPromptConfig]):
    def build(self) -> Self:
        self.prompt = ChatPromptTemplate.from_messages(
                messages=[
                    SystemMessagePromptTemplate.from_template(self.config.system_prompt),
                    HumanMessagePromptTemplate.from_template(self.config.template)
                    ]
                )
        return self
