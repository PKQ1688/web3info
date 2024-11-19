# from phi.agent import Agent
# from phi.tools.duckduckgo import DuckDuckGo

# from src.llm_engines import azure_model_azure
import dspy

from src.llm_engines import ds_llm

# web_agent = Agent(
#     name="Web Agent",
#     model=azure_model_azure,
#     tools=[DuckDuckGo()],
#     instructions=["Always include sources"],
#     show_tool_calls=True,
#     markdown=True,
# )
# web_agent.print_response("今天在法国发生了什么?", stream=True)


dspy.configure(lm=ds_llm)

document = """The 21-year-old made seven appearances for the Hammers and netted his only goal for them in a Europa League qualification round match against Andorran side FC Lustrains last season. Lee had two loan spells in League One last term, with Blackpool and then Colchester United. He scored twice for the U's but was unable to save them from relegation. The length of Lee's contract with the promoted Tykes has not been revealed. Find all the latest football transfers on our dedicated page."""

summarize = dspy.ChainOfThought("document -> summary")
response = summarize(document=document)

print(response.summary)

response.interact_history()
