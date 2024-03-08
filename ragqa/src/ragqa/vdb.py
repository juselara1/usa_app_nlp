from pydantic import BaseModel
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain.docstore.document import Document
from chromadb import HttpClient
from typing import List

class ChromaConfig(BaseModel):
    host: str
    port: int
    collection: str

class ChromaBuilder:
    config: ChromaConfig
    db: Chroma
    embeddings: Embeddings

    def set_elements(
        self,
        config: ChromaConfig,
        embeddings: Embeddings
        ) -> "ChromaBuilder":
        self.config = config
        self.embeddings = embeddings
        return self

    def get_db(self) -> Chroma:
        return self.db

    def build(self) -> "ChromaBuilder":
        self.client = HttpClient(host=self.config.host, port=self.config.port)
        self.db = Chroma(
                client=self.client,
                collection_name=self.config.collection,
                embedding_function=self.embeddings
                )
        return self 
