import os
import streamlit as st
from uuid import uuid4
from ragqa.model import (
        OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder,
        OpenAILLMConfig, OpenAILLMBuilder
        )
from ragqa.prompt import RagPromptConfig, RagPromptBuilder
from ragqa.vdb import ChromaConfig, ChromaBuilder
from ragqa.chain import RagChainBuilder
from ragqa.rag import Rag, RagConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from ragqa.app.base import AbstractComponent
from ragqa.app.session import AppStates

class DocumentValidator(AbstractComponent):
    def call(self) -> bool:
        document = st.session_state[AppStates.DOCUMENT]
        if document is None:
            st.markdown("Please upload a pdf file to start the chat... 😊")
            return True

        if st.session_state[AppStates.RAG] is None:
            # Embeddings
            embeddings_config = OpenAIEmbeddingsConfig(
                    model="text-embedding-3-small",
                    api_key=os.environ["OPENAI_API_KEY"]
                    )
            embeddings = (
                    OpenAIEmbeddingsBuilder()
                    .set_config(config=embeddings_config)
                    .build()
                    .get_model()
                    )

            # chroma
            chroma_config = ChromaConfig(
                    host="ragqa_chroma",
                    port=8000,
                    collection=str(uuid4())
                    )
            chroma = (
                    ChromaBuilder()
                    .set_elements(
                        config=chroma_config,
                        embeddings=embeddings
                        )
                    .build()
                    .get_db()
                    )

            # llm
            llm_config = OpenAILLMConfig(
                    model="gpt-3.5-turbo",
                    api_key=os.environ["OPENAI_API_KEY"],
                    temperature=0.7,
                    frequency_penalty=0.5
                    )
            llm = (
                    OpenAILLMBuilder()
                    .set_config(config=llm_config)
                    .build()
                    .get_model()
                    )

            # prompt
            prompt_config = RagPromptConfig(
                    system_prompt=(
                        "You're an usefule assistant for question answering from scientific papers. " +
                        "Use the following sentences (context) extracted from the paper to generate " +
                        "an answer given an user's question (question).\n" +
                        "You must return ONLY the answer without any format or additional comments, if " +
                        "the answer is not in the context you must respond: 'I don't know'."
                        ),
                    template="""Context:
                    {context}
                    Question:
                    {question}
                    """
                    )
            prompt = (
                    RagPromptBuilder()
                    .set_config(prompt_config)
                    .build()
                    .get_prompt()
                    )

            # chain
            chain = (
                    RagChainBuilder(k=10)
                    .set_elements(prompt, chroma)
                    .set_llm(llm)
                    .build()
                    .get_chain()
                    )

            # rag
            rag_config = RagConfig(
                    chunk_size=500,
                    chunk_overlap=0
                    )
            st.session_state[AppStates.RAG] = (
                    Rag()
                    .set_elements(chain, rag_config, chroma)
                    )
            st.session_state[AppStates.RAG].call("dummy question", document)
        return False

class ChatComponent(AbstractComponent):
    def call(self) -> bool:
        messages = st.container(height=400, border=True)
        if prompt := st.chat_input("Chat with AI"):
            rag = st.session_state[AppStates.RAG]
            response = rag.call(prompt)
            messages.chat_message("user").write(prompt)
            messages.chat_message("assistant").write(response)
        return False
