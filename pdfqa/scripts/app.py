import streamlit as st
from pdfqa.app.base import ListComponent, PageComponent
from pdfqa.app.header import HeaderComponent
from pdfqa.app.session import StateInitializer
from pdfqa.app.load import ConfigLoader
from pdfqa.app.chat import DocumentValidator, ChatComponent, ParametersViewer
from pdfqa.app.text import ChatInfo

def main() -> int:
    header = ListComponent().set_components([
        HeaderComponent(),
        StateInitializer()
        ])
    sidebar = ListComponent().set_components([
        ConfigLoader()
        ])
    content = ListComponent().set_components([
        DocumentValidator(),
        ChatInfo(),
        ParametersViewer(),
        ChatComponent()
    ])

    page = PageComponent().set_elements(
            header=header,
            sidebar=sidebar,
            page=content
            )
    page.call()
    return 0

if __name__ == "__main__":
    exit(main())
