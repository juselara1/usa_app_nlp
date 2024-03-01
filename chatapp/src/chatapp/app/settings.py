import os
import streamlit as st
from chatapp.app.base import AbstractComponent
from chatapp.app.session import SessionManager, AppStates
from chatapp.model.llm import ModelEnum
from chatapp.model.llm import OpenAIConfig, OpenAIChainBuilder

class LLMSettingsComponent(AbstractComponent):
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def call(self) -> None:
        system_prompt = st.text_input(label="System prompt:", value="You're an usefull AI assistant")
        model_type = st.selectbox(
            label="Choose model:",
            options=list(map(lambda x: x.name, ModelEnum)),
        ) 
        temperature = st.slider(
            label="Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
        )

        config = OpenAIConfig(
                api_key=os.environ["OPENAI_API_KEY"],
                model=ModelEnum[model_type],
                temperature=temperature,
                system_prompt=system_prompt
                )
        last_config = None
        if AppStates.CHAIN_CONFIG in st.session_state:
            last_config = st.session_state[AppStates.CHAIN_CONFIG]

        if config != last_config:
            builder = OpenAIChainBuilder(config=config).build()
            chain = builder.get_chain()
            self.session_manager.update(state=AppStates.CHAIN, obj=chain)
            self.session_manager.update(state=AppStates.CHAIN_CONFIG, obj=config)
