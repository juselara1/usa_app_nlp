import streamlit as st
from langchain.chains.base import Chain
from chatapp.app.base import AbstractComponent
from chatapp.app.session import AppStates
import logging

class ChatComponent(AbstractComponent):
    def call(self) -> None:
        messages = st.container(height=700, border=True)
        if prompt := st.chat_input("Chat with AI"):
            chain = st.session_state[AppStates.CHAIN]
            response = chain.invoke({"message": prompt})
            history = response["history"]
            for user_msg, assistant_msg in zip(history[::2], history[1::2]):
                logging.info(user_msg)
                logging.info(assistant_msg)
                messages.chat_message("user").write(user_msg.content)
                messages.chat_message("assistant").write(assistant_msg.content)
