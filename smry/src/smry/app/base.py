import streamlit as st
from abc import ABC, abstractmethod
from typing import List

class AbstractComponent(ABC):
    @abstractmethod
    def call(self) -> bool:
        ...

class ListComponent(AbstractComponent):
    components: List[AbstractComponent]

    def set_components(self, components: List[AbstractComponent]) -> "ListComponent":
        self.components = components
        return self

    def call(self) -> bool:
        for component in self.components:
            exit_code = component.call()
            if exit_code:
                return True
        return False

class PageComponent(AbstractComponent):
    sidebar: AbstractComponent
    page: AbstractComponent
    header: AbstractComponent
    
    def set_elements(
        self,
        sidebar: AbstractComponent,
        page: AbstractComponent,
        header: AbstractComponent
        ) -> "PageComponent":
        self.sidebar = sidebar
        self.page = page
        self.header = header
        return self

    def call(self) -> bool:
        code = self.header.call()
        if code: return True
        with st.sidebar:
            code = self.sidebar.call()
            if code: return True
        return self.page.call()

