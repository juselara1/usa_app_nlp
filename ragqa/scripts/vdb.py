import os
from ragqa.model import OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder
from ragqa.vdb import ChromaConfig, ChromaBuilder
from langchain.docstore.document import Document

def main() -> int:
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

    chroma_config = ChromaConfig(
            host="ragqa_chroma",
            port=8000,
            collection="this"
            )
    documents = [
            Document(page_content="With days to go before its previously announced mid-March deadline of finalizing plans for a third party presidential ticket, No Labels still doesn’t have a candidate or a clear plan — even as it looks to take swings at President Joe Biden that its own officials acknowledge are more potent than accurate. "),
            Document(page_content="Time is running out for Elizabeth Goldman to have one more baby with the uterus she received in a transplant two years ago. Her IVF treatments were halted after a state Supreme Court ruling last month, and it’s a delay she says her family can’t afford. When she was 14, she was told that she had been born without a uterus and that she would never get pregnant"),
            Document(page_content="The highly-anticipated race pitted Coleman, a former world champion over 100 meters, against Lyles, the reigning world champion over 100 meters and 200 meters, just five months out from the Paris Olympics. I had a lot of confidence in myself coming in, Coleman said, according to World Athletics. You have to feel confident in yourself. I set my mind on letting my body do what I have been doing in practice and I came out with a win.")
            ]
    chroma_db = (
            ChromaBuilder()
            .set_elements(
                config=chroma_config,
                embeddings=embeddings
                )
            .build()
            .get_db()
            )
    chroma_db.add_documents(documents)
    print(
        chroma_db.similarity_search(
                query="Something about sports",
                k=1
                )
    )
    #print(chroma_db._collection.get())
    return 0

if __name__ == "__main__":
    exit(main())
