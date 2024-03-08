import streamlit as st
from pathlib import Path
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from pdfqa.model.llm import ModelEnum
from pdfqa.app.base import AbstractComponent
from pdfqa.app.session import AppStates

class ConfigLoader(AbstractComponent):
    def call(self) -> None:
        with st.form(key="llm-form", clear_on_submit=True):
            file = st.file_uploader(
                label="Load PDF file",
                accept_multiple_files=False
            )
            temperature = st.slider(
                label="Temperature",
                help=(
                    "Controls how creative is the LLM, lower values are more deterministic "
                    "while higher values are more creative but erratic."
                    ),
                min_value=0.0,
                max_value=2.0,
                value=st.session_state[AppStates.TEMPERATURE]
            )
            selected_option = st.selectbox(
                    label="Model",
                    help="Version of the GPT model.",
                    options=[i.name for i in ModelEnum]
                    )
            model_type = ModelEnum[selected_option]
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
                st.session_state[AppStates.TEMPERATURE] = temperature
                st.session_state[AppStates.MODEL_TYPE] = model_type
                st.session_state[AppStates.CHAIN] = None
            return False
