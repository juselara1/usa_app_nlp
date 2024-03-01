import streamlit as st
from pdfqa.app.base import AbstractComponent

class ExampleComponent(AbstractComponent):
    def call(self) -> bool:
        st.write("Example")
        return False
