from abc import ABC, abstractmethod
from pydantic import BaseModel
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from typing import Generic, TypeVar, Self
from langchain_core.documents import Document
from langchain_community.document_loaders import PlaywrightURLLoader

T = TypeVar('T')
class AbstractLoader(ABC, Generic[T]):
    config: T

    def set_config(self, config: T) -> Self:
        self.config = config
        return self

    @abstractmethod
    def load(self) -> Document:
        ...

class UrlConfig(BaseModel):
    url: str

class UrlLoader(AbstractLoader[UrlConfig]):
    def load(self) -> Document:
        document = PlaywrightURLLoader(urls=[self.config.url]).load()[0]
        return document

class PdfConfig(BaseModel):
    filename: str
    data: bytes

class PdfLoader(AbstractLoader[PdfConfig]):
    def load(self) -> Document:
        path = Path("/tmp/") / self.config.filename
        with open(path, "wb") as f:
            f.write(self.config.data)
        pages = PyMuPDFLoader(file_path=str(path)).load()
        document = Document(
                page_content="\n".join(
                    page.page_content 
                    for page in pages
                    ),
                metadata={"source": str(path)}
                )
        return document
