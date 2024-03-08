import streamlit as st
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from ragqa.app.base import AbstractComponent
from ragqa.app.session import AppStates

class ConfigLoader(AbstractComponent):
    def call(self) -> None:
        with st.form(key="llm-form", clear_on_submit=True):
            file = st.file_uploader(
                label="Load PDF file",
                accept_multiple_files=False
            )
            submitted = st.form_submit_button("Upload")
            if submitted:
                if file is None:
                    st.error("You've not uploaded a file")
                    return False
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
                st.session_state[AppStates.RAG] = None
            return False
