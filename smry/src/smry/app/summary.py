import os
import streamlit as st
from uuid import uuid4
from smry.model import (
        OpenAILLMConfig, OpenAILLMBuilder,
        OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder
        )
from smry.vdb import ChromaConfig, ChromaBuilder
from smry.app.base import AbstractComponent
from smry.app.session import AppStates
from smry.chain import DefaultSummaryEnum, DefaultSummaryChainBuilder
from smry.summary import (
        ExtractiveSummarizer, ExtractiveSummaryConfig, SummaryEnum,
        RagSummarizer, DefaultSummarizer, BaseSummaryConfig, AbstractSummarizer
        )

class SummarySelector(AbstractComponent):
    def call(self) -> bool:
        opts = list(map(lambda x: x.name, SummaryEnum))
        option = st.selectbox(label="Select summarizer", options=opts)
        st.session_state[AppStates.SUMMARY_TYPE] = SummaryEnum[option]
        return False

class ExtractiveSummaryComponent(AbstractComponent):
    def call(self) -> bool:
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

        # vector store
        chroma_config = ChromaConfig(
                host="chroma",
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

        # summarizer
        summary_config = ExtractiveSummaryConfig(
                chunk_size=500,
                chunk_overlap=0,
                n_clusters=10,
                seed=0
                )
        st.session_state[AppStates.SUMMARIZER] = (
                ExtractiveSummarizer()
                .set_chroma(chroma)
                .set_config(summary_config)
                )
        return False

class RagSummaryComponent(AbstractComponent):
    def call(self) -> bool:
        llm_config = OpenAILLMConfig(
                model="gpt-3.5-turbo",
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.7,
                frequency_penalty=0.0
                )
        llm = (
                OpenAILLMBuilder()
                .set_config(llm_config)
                .build()
                .get_model()
                )

        # chain
        chain = (
                DefaultSummaryChainBuilder(chain_type=DefaultSummaryEnum.STUFF)
                .set_llm(llm)
                .build()
                .get_chain()
                )

        # embeddings
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

        # vector store
        chroma_config = ChromaConfig(
                host="chroma",
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

        # summarizer
        summary_config = ExtractiveSummaryConfig(
                chunk_size=500,
                chunk_overlap=0,
                n_clusters=10,
                seed=0
                )
        st.session_state[AppStates.SUMMARIZER] = (
                RagSummarizer()
                .set_chroma(chroma)
                .set_chain(chain)
                .set_config(summary_config)
                )
        return False

class DefaultSummaryComponent(AbstractComponent):
    def call(self) -> bool:
        # llm
        llm_config = OpenAILLMConfig(
                model="gpt-3.5-turbo",
                api_key=os.environ["OPENAI_API_KEY"],
                temperature=0.7,
                frequency_penalty=0.0
                )
        llm = (
                OpenAILLMBuilder()
                .set_config(llm_config)
                .build()
                .get_model()
                )

        # chain
        summary_type: SummaryEnum = st.session_state[AppStates.SUMMARY_TYPE]
        chain = (
                DefaultSummaryChainBuilder(chain_type=DefaultSummaryEnum[summary_type.name])
                .set_llm(llm)
                .build()
                .get_chain()
                )

        # summarizer
        summary_config = BaseSummaryConfig(
                chunk_size=500,
                chunk_overlap=0,
                )
        st.session_state[AppStates.SUMMARIZER] = (
                DefaultSummarizer()
                .set_chain(chain)
                .set_config(summary_config)
                )
        return False

class SummarizerComponent(AbstractComponent):
    def call(self) ->  bool:
        summary_type: SummaryEnum = st.session_state[AppStates.SUMMARY_TYPE]
        match summary_type:
            case SummaryEnum.EXTRACTIVE:
                ExtractiveSummaryComponent().call()
            case SummaryEnum.RAG:
                RagSummaryComponent().call()
            case _:
                DefaultSummaryComponent().call()
        return False

class SummaryGenerator(AbstractComponent):
    def call(self) -> bool:
        document: Document = st.session_state[AppStates.DOCUMENT]
        summarizer: AbstractSummarizer = st.session_state[AppStates.SUMMARIZER]
        with st.spinner(text="Summarizing..."):
            summary = summarizer.summarize(document)

        st.markdown(
                f"""
                <div class="summary-container">
                    <h1>Generated Summary</h1>
                    <p>{summary}</p>
                </div>
                """,
                unsafe_allow_html=True
                )
        return False
