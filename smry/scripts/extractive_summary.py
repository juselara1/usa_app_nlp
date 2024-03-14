import os
from uuid import uuid4
from smry.load import UrlConfig, UrlLoader
from smry.model import OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder
from smry.vdb import ChromaConfig, ChromaBuilder
from smry.summary import ExtractiveSummarizer, ExtractiveSummaryConfig

def main() -> int:
    # document
    config = UrlConfig(
        url="https://developer.mozilla.org/es/docs/Web/HTML#tutoriales_para_principiantes",
    )
    document = UrlLoader().set_config(config).load()

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
            n_clusters=3,
            seed=0
            )
    summarizer = (
            ExtractiveSummarizer()
            .set_chroma(chroma)
            .set_config(summary_config)
            )

    print(summarizer.summarize(document))

    return 0

if __name__ == "__main__":
    exit(main())
