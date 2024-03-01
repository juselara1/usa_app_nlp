import os
import streamlit as st
from chatapp.model.llm import OpenAIConfig, OpenAIChainBuilder, ModelEnum
from chatapp.app.base import ListComponent
from chatapp.app.chat import ChatComponent
from chatapp.app.settings import LLMSettingsComponent
from chatapp.app.session import SessionManager, AppStates

import sys
import logging

logging.basicConfig(
        filename="app.log",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
        )

def main() -> int:
    session_manager = SessionManager()
    with st.sidebar:
        sb = ListComponent().set_components([LLMSettingsComponent(session_manager)])
        sb.call()
    page = ListComponent().set_components([ChatComponent()])
    page.call()
    return 0

if __name__ == "__main__":
    exit(main())
