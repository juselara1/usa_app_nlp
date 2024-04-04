import streamlit as st
from smry.app.base import AbstractComponent
from smry.app.session import AppStates

class SummaryTitle(AbstractComponent):
    def call(self) -> bool:
        if st.session_state[AppStates.DOCUMENT] is None:
            st.markdown(
                    """
                    <div class="welcome-container">
                        <h1>Summarization App</h1>
                        <div class="welcome-description">
                            <p>Welcome 👋, we can</p>
                            <div class="text-gradient"><p>Summarize</p></div>
                            <p>Any PDF or URL document.</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                    )
            return True
        return False
