import os

import dspy
from dotenv import load_dotenv
from phi.model.azure import AzureOpenAIChat
from phi.model.openai.like import OpenAILike

load_dotenv(override=True)

llm_model_opeailike = OpenAILike(
    id="deepseek-chat",
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    # id="glm-4-flash",
    # base_url=os.getenv("GLM_BASE_URL"),
    # api_key=os.getenv("GLM_API_KEY"),
)

azure_model_azure = AzureOpenAIChat(
    id="gpt4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

ds_llm = dspy.LM(
    model="azure/gpt4o-mini",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_API_VERSION"),
)

if __name__ == "__main__":
    from phi.agent import Agent

    agent = Agent(model=azure_model_azure, markdown=True)

    agent.print_response("how are you?", stream=True)
