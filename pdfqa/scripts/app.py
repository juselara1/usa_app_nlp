import streamlit as st
from pdfqa.app.base import ListComponent, PageComponent
from pdfqa.app.header import HeaderComponent
from pdfqa.app.session import StateInitializer
from pdfqa.app.load import PdfLoader
from pdfqa.app.chat import DocumentValidator
from pdfqa.app.text import ExampleComponent

def main() -> int:
    header = ListComponent().set_components([
        HeaderComponent(),
        StateInitializer()
        ])
    sidebar = ListComponent().set_components([
        PdfLoader()
        ])
    content = ListComponent().set_components([
        DocumentValidator(),
        ExampleComponent()
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
