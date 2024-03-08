import os
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
# from langchain.globals import set_debug, set_verbose
#
# set_debug(True)
# set_verbose(True)

def main() -> int:
    # Document
    # pages = PyMuPDFLoader(file_path="./data/attention.pdf").load()
    pages = PyMuPDFLoader(file_path="./data/machinelearning.pdf").load()
    document = Document(page_content="\n".join(
        page.page_content
        for page in pages
    ))

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
                "You must return ONLY the answer without any format or additional comments."
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
    response = (
            Rag()
            .set_elements(chain, rag_config, chroma)
            .call(document, "what is a Gaussian Mixture Model?")
            )
    print(response)
    return 0

if __name__ == "__main__":
    exit(main())
