import streamlit as st
from enum import Enum
from typing import Any

class AppStates(Enum):
    CHAIN = "CHAIN"
    CHAIN_CONFIG = "CHAIN_CONFIG"

class SessionManager:
    def register(self, state: AppStates, obj: Any) -> None:
        if state not in st.session_state:
            st.session_state[state] = obj

    def update(self, state: AppStates, obj: Any) -> None:
        st.session_state[state] = obj
