from transformers.agents import ReactCodeAgent
from src.llm_engines import OpenAIEngine

llm_engine = OpenAIEngine(use_azure=False)

code_agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    max_iterations=10,
)


code_agent.run("print('Hello, world!')")
