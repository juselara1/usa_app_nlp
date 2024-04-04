import streamlit as st
from smry.app.base import AbstractComponent

class HeaderComponent(AbstractComponent):
    def call(self) -> bool:
        with open("static/style.css") as f:
            css = f.read()

        st.set_page_config(page_title="Summarization", layout="centered")

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
                """ + 
                f"""
                {css}
                </style>
                """,
                unsafe_allow_html=True
                )
        return False
