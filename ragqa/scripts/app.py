import streamlit as st
from ragqa.app.base import ListComponent, PageComponent
from ragqa.app.header import HeaderComponent
from ragqa.app.session import StateInitializer
from ragqa.app.load import ConfigLoader
from ragqa.app.chat import DocumentValidator, ChatComponent
from ragqa.app.text import ChatInfo

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
