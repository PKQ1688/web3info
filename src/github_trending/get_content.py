from transformers.agents import ReactJsonAgent
from transformers.agents.search import VisitWebpageTool

from src.llm_engines import OpenAIEngine

llm_engine = OpenAIEngine(use_azure=False)

agent = ReactJsonAgent(tools=[VisitWebpageTool()], llm_engine=llm_engine, verbose=0)

agent_prompt = """
通过我给你的url获取今天的热门github的项目,然后根据对应的库和地址,去分析对应的库。
然后给每个库给出一个分析报告,这其中要包含库的名称,库的地址,库主要的功能,库的优势,库的劣势,库的使用场景。
"""

agent_output = agent.run(agent_prompt, url="https://github.com/trending")
print(agent_output)

# res = VisitWebpageTool().forward(url="https://github.com/trending?since=daily")
# res = VisitWebpageTool().forward(url="https://github.com/OpenBB-finance/OpenBB")

# print(res)
# print("-------------------")

# git_trending_url = llm_engine(message_str=f"通过这个网页的内容,整理出对应的库和对应的地址,给出对应的json形式{res}")
# git_trending_url = llm_engine(message_str=f"概括对应网页的内容{res}")
# print(git_trending_url)
