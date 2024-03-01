import streamlit as st
from langchain.chains.base import Chain
from pdfqa.app.base import AbstractComponent
from pdfqa.app.session import AppStates

class DocumentValidator(AbstractComponent):
    def call(self) -> bool:
        if st.session_state[AppStates.DOCUMENT] is None:
            st.markdown("Please upload a pdf file to start the chat... 😊")
            return True
        return False

class ChatComponent(AbstractComponent):
    def call(self) -> bool:
        messages = st.container(height=700, border=True)
        if prompt := st.chat_input("Chat with AI"):
            chain = st.session_state[AppStates.CHAIN]
            response = chain.invoke({"message": prompt})
            history = response["history"]
            for user_msg, assistant_msg in zip(history[::2], history[1::2]):
                messages.chat_message("user").write(user_msg.content)
                messages.chat_message("assistant").write(assistant_msg.content)
        return False
