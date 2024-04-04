import streamlit as st
from enum import StrEnum, auto
from typing import Any
from uuid import uuid4
from smry.app.base import AbstractComponent

class AppStates(StrEnum):
    SESSION_ID = auto()
    DOCUMENT = auto()
    SUMMARY_TYPE = auto()
    SUMMARIZER = auto()

class StateInitializer(AbstractComponent):
    def call(self) -> bool:
        if AppStates.SESSION_ID not in st.session_state:
            st.session_state[AppStates.SESSION_ID] = str(uuid4())
            st.session_state[AppStates.DOCUMENT] = None
            st.session_state[AppStates.SUMMARY_TYPE] = None
            st.session_state[AppStates.SUMMARIZER] = None
        return False
