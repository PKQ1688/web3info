from transformers.agents import ManagedAgent, ReactCodeAgent, ReactJsonAgent
from transformers.agents.search import DuckDuckGoSearchTool, VisitWebpageTool

from src.llm_engines import OpenAIEngine

llm_engine = OpenAIEngine(use_azure=False)

web_agent = ReactJsonAgent(
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
    llm_engine=llm_engine,
    max_iterations=50,
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
    max_iterations=50,
)

manager_agent.run("""
21 34 41 51 53 77 83 85 86按顺序将这9个数字插入1个空的3阶b+树.
这个b+树,怎么删除,才能在删除最少数据的前提下,把高度降低到2啊
""",
additional_notes="3阶树,一个节点只能有2个数字")
