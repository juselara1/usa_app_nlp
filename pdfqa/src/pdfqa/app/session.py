import streamlit as st
from enum import StrEnum, auto
from typing import Any
from uuid import uuid4
from pdfqa.model.llm import ModelEnum
from pdfqa.app.base import AbstractComponent

class AppStates(StrEnum):
    SESSION_ID = auto()
    DOCUMENT = auto()
    CHAIN = auto()
    TEMPERATURE = auto()
    MODEL_TYPE = auto()


class StateInitializer(AbstractComponent):
    def call(self) -> bool:
        if AppStates.SESSION_ID not in st.session_state:
            st.session_state[AppStates.SESSION_ID] = str(uuid4())
            st.session_state[AppStates.DOCUMENT] = None
            st.session_state[AppStates.CHAIN] = None
            st.session_state[AppStates.TEMPERATURE] = 0.7
            st.session_state[AppStates.MODEL_TYPE] = ModelEnum.GPT35
        return False
