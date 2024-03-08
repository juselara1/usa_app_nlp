from abc import ABC, abstractmethod
from operator import itemgetter
from typing import Self
from pydantic import PositiveInt
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_community.vectorstores.chroma import Chroma

class AbstractChainBuilder(ABC):
    llm: BaseChatModel
    chain: Runnable

    def set_llm(self, llm: BaseChatModel) -> Self:
        self.llm = llm
        return self

    def get_chain(self) -> Runnable:
        return self.chain

    @abstractmethod
    def build(self) -> Self:
        ...

class RagChainBuilder(AbstractChainBuilder):
    prompt: BasePromptTemplate
    chroma: Chroma

    def __init__(self, k: PositiveInt):
        self.k = k

    def set_elements(self, prompt: BasePromptTemplate, chroma: Chroma) -> Self:
        self.prompt = prompt
        self.chroma = chroma
        return self

    def build(self) -> Self:
        self.chain = (
                {
                    "context": itemgetter("question") | self.chroma.as_retriever(search_kwargs={"k": self.k}),
                    "question": itemgetter("question")
                    }
                | self.prompt
                | self.llm
                | StrOutputParser()
                )
        return self
