import streamlit as st
from abc import ABC, abstractmethod
from typing import List

class AbstractComponent(ABC):
    @abstractmethod
    def call(self) -> None:
        ...

class ListComponent(AbstractComponent):
    components: List[AbstractComponent]

    def set_components(self, components: List[AbstractComponent]) -> "ListComponent":
        self.components = components
        return self

    def call(self) -> None:
        for component in self.components:
            component.call()
