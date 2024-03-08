import streamlit as st
from ragqa.app.base import AbstractComponent

class ChatInfo(AbstractComponent):
    def call(self) -> bool:
        st.markdown(
                """
                # Chat with Your Pdf
                ---

                Start asking questions about your uploaded document and a LLM will provide you answers.
                """
                )
        return False
