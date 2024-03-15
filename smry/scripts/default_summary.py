import os
from argparse import ArgumentParser
from uuid import uuid4
from smry.load import UrlConfig, UrlLoader
from smry.model import OpenAILLMConfig, OpenAILLMBuilder
from smry.chain import DefaultSummaryEnum, DefaultSummaryChainBuilder
from smry.summary import DefaultSummarizer, BaseSummaryConfig

def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        "--chain_type",
        type=str,
        help="Type of default summary",
        choices=list(DefaultSummaryEnum),
    )
    return parser

def main() -> int:
    parser = get_parser()
    args = parser.parse_args()
    chain_type = DefaultSummaryEnum[args.chain_type.upper()]
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

    # summarizer
    summary_config = BaseSummaryConfig(
            chunk_size=500,
            chunk_overlap=0,
            )
    summarizer = (
            DefaultSummarizer()
            .set_chain(chain)
            .set_config(summary_config)
            )

    print(summarizer.summarize(document))
    return 0

if __name__ == "__main__":
    exit(main())
