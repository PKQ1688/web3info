import os
from transformers.agents import ReactCodeAgent
from src.llm_engines import OpenAIEngine

if not os.path.exists("figures"):
    os.mkdir("figures")

llm_engine = OpenAIEngine(model_name="deepseek-chat")

agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    additional_authorized_imports=["numpy", "pandas", "matplotlib.pyplot", "seaborn"],
    max_iterations=10,
)

additional_notes = """
### Variable Notes
pclass: A proxy for socio-economic status (SES)
1st = Upper
2nd = Middle
3rd = Lower
age: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5
sibsp: The dataset defines family relations in this way...
Sibling = brother, sister, stepbrother, stepsister
Spouse = husband, wife (mistresses and fiancés were ignored)
parch: The dataset defines family relations in this way...
Parent = mother, father
Child = daughter, son, stepdaughter, stepson
Some children traveled only with a nanny, therefore parch=0 for them.
"""

additional_notes_zh = """
### 变量注释
pclass：社会经济地位 (SES) 的代理
第一=上层
第二=中间
第三=较低
age：年龄小于1则为小数，如果是估计年龄，是否为xx.5的形式
sibsp：数据集以这种方式定义家庭关系......
兄弟姐妹=兄弟、姐妹、继兄弟、继姐妹
Spouse = 丈夫、妻子（情妇和未婚夫被忽略）
parch：数据集以这种方式定义家庭关系......
父母=母亲、父亲
孩子=女儿、儿子、继女、继子
有些孩子只和保姆一起旅行，因此对他们来说 parch=0。
"""

agent_prompt = """
You are an expert data analyst.
Please load the source file and analyze its content.
According to the variables you have, begin by listing 3 interesting questions that could be asked on this data, for instance about specific correlations with survival rate.
Then answer these questions one by one, by finding the relevant numbers.
Meanwhile, plot some figures using matplotlib/seaborn and save them to the (already existing) folder './figures/': take care to clear each figure with plt.clf() before doing another plot.

In your final answer: summarize these correlations and trends
After each number derive real worlds insights, for instance: "Correlation between is_december and boredness is 1.3453, which suggest people are more bored in winter".
Your final answer should have at least 3 numbered and detailed parts.
"""

agent_prompt_zh = """
您是一位专家数据分析师。
请加载源文件并分析其内容。
根据您拥有的变量，首先列出 3 个可以针对此数据提出的有趣问题，例如与生存率的具体相关性。
然后通过查找相关数字来一一回答这些问题。
同时，使用 matplotlib/seaborn 绘制一些图形并将它们保存到（已经存在的）文件夹“./figures/”：在进行另一个绘图之前，请注意使用 plt.clf() 清除每个图形。

在您的最终答案中：总结这些相关性和趋势
在每个数字得出现实世界的见解后，例如：“is_december 和无聊度之间的相关性为 1.3453，这表明人们在冬天更无聊”。
您的最终答案应至少包含 3 个编号且详细的部分。"""

analysis = agent.run(
    task=agent_prompt_zh,
    additional_notes=additional_notes_zh,
    source_file="titanic/train.csv",
)

print(analysis)
