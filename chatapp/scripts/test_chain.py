import os
from chatapp.model.llm import OpenAIConfig, OpenAIChainBuilder

def main() -> int:
    api_key = os.environ["OPENAI_API_KEY"]
    config = OpenAIConfig(
            api_key=api_key,
            model="gpt-3.5-turbo",
            temperature=0.5
            )
    builder = OpenAIChainBuilder(config=config).build()
    chain = builder.get_chain()
    print(chain.invoke({"message": "Hello"}))
    print(chain.invoke({"message": "Tell me the result of the following operation: 33 + 77"}))
    return 0

if __name__ == "__main__":
    exit(main())
