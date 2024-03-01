import streamlit as st
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pdfqa.app.base import AbstractComponent
from pdfqa.app.session import AppStates

class PdfLoader(AbstractComponent):
    def call(self) -> None:
        file = st.file_uploader(
            label="Load PDF file",
            accept_multiple_files=False
        )
        if file is not None:
            data = file.getvalue()
            name = file.name
            path = Path("/tmp/") / name
            with open(path, "wb") as f:
                f.write(data)
            pages = PyMuPDFLoader(file_path=str(path)).load()
            document = Document(
                    page_content="\n".join(
                        page.page_content 
                        for page in pages
                        ),
                    metadata={"source": str(path)}
                    )
            st.session_state[AppStates.DOCUMENT] = document
        return False
