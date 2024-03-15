from abc import ABC, abstractmethod
from enum import StrEnum, auto
from typing import Self
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain.chains.summarize import load_summarize_chain

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


class DefaultSummaryEnum(StrEnum):
    STUFF = auto()
    MAP_REDUCE = auto()
    REFINE = auto()

class DefaultSummaryChainBuilder(AbstractChainBuilder):
    def __init__(self, chain_type: DefaultSummaryEnum):
        self.chain_type = chain_type

    def build(self) -> Self:
        self.chain = load_summarize_chain(
                llm=self.llm, chain_type=self.chain_type.name.lower(),
                input_key="context", output_key="output_text"
                )
        return self
