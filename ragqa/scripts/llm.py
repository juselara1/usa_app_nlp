import os
from ragqa.model import OpenAILLMConfig, OpenAILLMBuilder

def main() -> int:
    config = OpenAILLMConfig(
            model="gpt-3.5-turbo",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=0.7,
            frequency_penalty=0.5
            )
    model = (
            OpenAILLMBuilder()
            .set_config(config=config)
            .build()
            .get_model()
            )
    print(model.invoke("Hello, my name is Juan"))
    return 0

if __name__ == "__main__":
    exit(main())
