from transformers.agents import ManagedAgent, ReactCodeAgent, ReactJsonAgent
from transformers.agents.search import DuckDuckGoSearchTool, VisitWebpageTool

from src.llm_engines import OpenAIEngine

llm_engine = OpenAIEngine(use_azure=False)

web_agent = ReactJsonAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    llm_engine=llm_engine,
    max_iterations=10,
)

managed_web_agent = ManagedAgent(
    agent=web_agent,
    name="search",
    description="Runs web searches for you. Give it your query as an argument.",
)

manager_agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    managed_agents=[managed_web_agent],
    additional_authorized_imports=["time", "datetime"],
)

manager_agent.run("How many years ago was Stripe founded?")
