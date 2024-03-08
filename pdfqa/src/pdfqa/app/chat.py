import os
import streamlit as st
from langchain.chains.base import Chain
from pdfqa.model.llm import OpenAIConfig, OpenAIChainBuilder, ModelEnum
from pdfqa.app.base import AbstractComponent
from pdfqa.app.session import AppStates

class DocumentValidator(AbstractComponent):
    def call(self) -> bool:
        if st.session_state[AppStates.DOCUMENT] is None:
            st.markdown("Please upload a pdf file to start the chat... 😊")
            return True
        if st.session_state[AppStates.CHAIN] is None:
            system_prompt = f"""
            Given the following content, you must answer some questions.

            {st.session_state[AppStates.DOCUMENT].page_content}

            Questions:
            """
            config = OpenAIConfig(
                    api_key=os.environ["OPENAI_API_KEY"],
                    model=st.session_state[AppStates.MODEL_TYPE],
                    temperature=st.session_state[AppStates.TEMPERATURE],
                    system_prompt=system_prompt
                    )
            st.session_state[AppStates.CHAIN] = (
                    OpenAIChainBuilder(config=config)
                    .build()
                    .get_chain()
                    )
        return False

class ParametersViewer(AbstractComponent):
    def call(self) -> bool:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown(f"**Temperature**\n{st.session_state[AppStates.TEMPERATURE]}")

        with col2:
            with st.container(border=True):
                st.markdown(f"**Model Type**\n{st.session_state[AppStates.MODEL_TYPE].name}")
        return False

class ChatComponent(AbstractComponent):
    def call(self) -> bool:
        messages = st.container(height=400, border=True)
        if prompt := st.chat_input("Chat with AI"):
            chain = st.session_state[AppStates.CHAIN]
            response = chain.invoke({"message": prompt})
            history = response["history"]
            for user_msg, assistant_msg in zip(history[::2], history[1::2]):
                messages.chat_message("user").write(user_msg.content)
                messages.chat_message("assistant").write(assistant_msg.content)
        return False
