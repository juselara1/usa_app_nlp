from typing import Self, List, Optional
from pydantic import BaseModel, PositiveInt
from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

class RagConfig(BaseModel):
    chunk_size: PositiveInt
    chunk_overlap: int

class Rag:
    chain: Runnable
    config: RagConfig
    chroma: Chroma

    def set_elements(self, chain: Runnable, config: RagConfig, chroma: Chroma) -> Self:
        self.chain = chain
        self.config = config
        self.chroma = chroma
        return self

    def get_chunks(self, document: Document) -> List[Document]:
        chunks = (
                RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                    )
                .split_documents([document])
                )
        return chunks

    def call(self, question: str, document: Optional[Document]=None) -> str:
        if document is not None:
            chunks = self.get_chunks(document)
            self.chroma.add_documents(chunks)
        response = self.chain.invoke({"question": question})
        return response
