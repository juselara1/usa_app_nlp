import streamlit as st
from enum import StrEnum, auto
from typing import Any
from pdfqa.app.base import AbstractComponent
from uuid import uuid4

class AppStates(StrEnum):
    SESSION_ID = auto()
    DOCUMENT = auto()

class StateInitializer(AbstractComponent):
    def call(self) -> bool:
        if AppStates.SESSION_ID not in st.session_state:
            st.session_state[AppStates.SESSION_ID] = str(uuid4())
            st.session_state[AppStates.DOCUMENT] = None
        return False
