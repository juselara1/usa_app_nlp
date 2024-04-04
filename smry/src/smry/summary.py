import numpy as np
from enum import StrEnum, auto
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from typing import List, Self, Literal, Generic, TypeVar
from pydantic import BaseModel, PositiveInt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_community.vectorstores.chroma import Chroma


class SummaryEnum(StrEnum):
    EXTRACTIVE = auto()
    RAG = auto()
    STUFF = auto()
    MAP_REDUCE = auto()
    REFINE = auto()

ZeroPositive = PositiveInt | Literal[0]
class BaseSummaryConfig(BaseModel):
    chunk_size: PositiveInt
    chunk_overlap: ZeroPositive

T = TypeVar('T', bound=BaseSummaryConfig)

class AbstractSummarizer(ABC, Generic[T]):
    config: T

    def set_config(self, config: T) -> Self:
        self.config = config
        return self

    def get_chunks(self, document: Document) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        documents = splitter.split_documents([document])
        return documents

    @abstractmethod
    def summarize(self, document: Document) -> str:
        ...


class ExtractiveSummaryConfig(BaseSummaryConfig):
    n_clusters: PositiveInt
    seed: ZeroPositive

class ExtractiveSummarizer(AbstractSummarizer[ExtractiveSummaryConfig]):
    chroma: Chroma

    def set_chroma(self, chroma: Chroma) -> Self:
        self.chroma = chroma
        return self

    def summarize(self, document: Document) -> str:
        chunks = self.get_chunks(document)
        self.chroma.add_documents(chunks)
        embeddings_response = self.chroma.get(include=["embeddings"])
        embeddings = np.array(embeddings_response["embeddings"])

        centroids = (
                KMeans(
                    n_clusters=self.config.n_clusters,
                    random_state=self.config.seed,
                    )
                .fit(embeddings)
                .cluster_centers_
                )
        relevant_docs = []
        for i in range(centroids.shape[0]):
            relevant_docs.append(
                    self.chroma.similarity_search_by_vector(
                        embedding=centroids[i].tolist(),
                        k=1
                        )[0]
                    )
        return "\n- ".join(doc.page_content for doc in relevant_docs)

class DefaultSummarizer(AbstractSummarizer[BaseSummaryConfig], ABC):
    chain: Runnable

    def set_chain(self, chain: Runnable) -> Self:
        self.chain = chain
        return self

    def summarize(self, document: Document) -> str:
        chunks = self.get_chunks(document)
        response = self.chain.invoke({"context": chunks})["output_text"]
        return response

class RagSummarizer(ExtractiveSummarizer):
    chain: Runnable

    def set_chain(self, chain: Runnable) -> Self:
        self.chain = chain
        return self

    def summarize(self, document: Document) -> str:
        relevant_chunks = super().summarize(document)
        chunks = [Document(page_content=chunk) for chunk in relevant_chunks.split("\n- ")]
        response = self.chain.invoke({"context": chunks})["output_text"]
        return response
