import os
from ragqa.model import OpenAIEmbeddingsConfig, OpenAIEmbeddingsBuilder

def main() -> int:
    config = OpenAIEmbeddingsConfig(
            model="text-embedding-3-small",
            api_key=os.environ["OPENAI_API_KEY"]
            )
    model = (
            OpenAIEmbeddingsBuilder()
            .set_config(config=config)
            .build()
            .get_model()
            )
    print(model.embed_query("This is an example text"))
    return 0

if __name__ == "__main__":
    exit(main())
