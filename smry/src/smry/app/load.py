import streamlit as st
from enum import StrEnum, auto
from smry.app.base import AbstractComponent
from smry.load import PdfConfig, PdfLoader, UrlConfig, UrlLoader
from smry.app.session import AppStates

class SourceEnum(StrEnum):
    PDF = auto()
    URL = auto()

class PdfLoadComponent(AbstractComponent):
    def call(self) -> bool:
        with st.form(key="pdf-load", clear_on_submit=True):
            file = st.file_uploader(
                label="Upload `.pdf` file",
                accept_multiple_files=False
            )
            submitted = st.form_submit_button(label="Upload")
            if submitted:
                config = PdfConfig(
                        filename=file.name,
                        data=file.getvalue()
                        )
                document = PdfLoader().set_config(config).load()
                st.session_state[AppStates.DOCUMENT] = document
        return False

class UrlLoadComponent(AbstractComponent):
    def call(self) -> bool:
        with st.form(key="url-load", clear_on_submit=True):
            url = st.text_input(label="Add `url`", placeholder="https://example.com")
            submitted = st.form_submit_button(label="Upload")
            if submitted:
                config = UrlConfig(url=url)
                with st.spinner(text="Loading page"):
                    document = UrlLoader().set_config(config).load()
                    st.session_state[AppStates.DOCUMENT] = document
        return False

class FileLoadComponent(AbstractComponent):
    def call(self) -> bool:
        tab_names = list(map(lambda x: x.name, SourceEnum))
        tabs = st.tabs(tab_names)
        for name, tab in zip(tab_names, tabs):
            source = SourceEnum[name]
            with tab:
                match source:
                    case SourceEnum.PDF:
                        PdfLoadComponent().call()
                    case SourceEnum.URL:
                        UrlLoadComponent().call()
                    case _:
                        st.warning("Not implemented")
        return False
