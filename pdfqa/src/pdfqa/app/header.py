import streamlit as st
from pdfqa.app.base import AbstractComponent

class HeaderComponent(AbstractComponent):
    def call(self) -> bool:
        st.set_page_config(page_title="PdfQA", layout="centered")
        st.markdown(
                """
                <style>
                .reportview-container {
                    margin-top: -2em;
                    }
                #MainMenu {visibility: hidden;}
                .stDeployButton {display:none;}
                footer {visibility: hidden;}
                #stDecoration {display:none;}
                </style>
                """,
                unsafe_allow_html=True
                )
        return False
