import os
from argparse import ArgumentParser
from uuid import uuid4
from smry.load import UrlConfig, UrlLoader
from smry.model import (
        OpenAILLMConfig, OpenAILLMBuilder,
        OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder
        )
from smry.vdb import ChromaConfig, ChromaBuilder
from smry.chain import DefaultSummaryEnum, DefaultSummaryChainBuilder
from smry.summary import ExtractiveSummaryConfig, RagSummarizer

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--chain_type",
        type=str,
        help="Type of default summary",
        choices=list(DefaultSummaryEnum),
    )
    parser.add_argument(
            "--n_clusters",
            type=int,
            help="Number of clusters for rag"
            )
    return parser

def main() -> int:
    parser = get_parser()
    args = parser.parse_args()
    chain_type = DefaultSummaryEnum[args.chain_type.upper()]
    n_clusters = args.n_clusters
    # document
    config = UrlConfig(
        url="https://developer.mozilla.org/es/docs/Web/HTML#tutoriales_para_principiantes",
    )
    document = UrlLoader().set_config(config).load()

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
    chain = (
            DefaultSummaryChainBuilder(chain_type=chain_type)
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
            n_clusters=n_clusters,
            seed=0
            )
    summarizer = (
            RagSummarizer()
            .set_chroma(chroma)
            .set_chain(chain)
            .set_config(summary_config)
            )

    print(summarizer.summarize(document))
    return 0

if __name__ == "__main__":
    exit(main())
