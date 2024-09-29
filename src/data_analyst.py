import json
from pathlib import Path

from src.llm_engines import OpenAIEngine
from transformers.agents import ReactCodeAgent

if Path("output.csv").exists():
    Path("output.csv").unlink()

llm_engine = OpenAIEngine(use_azure=True)

agent = ReactCodeAgent(
    tools=[],
    llm_engine=llm_engine,
    additional_authorized_imports=[
        "numpy",
        "pandas",
        "matplotlib.pyplot",
        "seaborn",
        "sklearn",
    ],
    max_iterations=12,
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
Some children travelled only with a nanny, therefore parch=0 for them.
"""

agent_prompt = """
You are an expert data analyst.
Please load the source file and analyze its content.
According to the variables you have, begin by listing 3 interesting questions that could be asked on this data,
for instance about specific correlations with survival rate.
Then answer these questions one by one, by finding the relevant numbers.
Meanwhile, plot some figures using matplotlib/seaborn and save them to the (already existing) folder './figures/':
take care to clear each figure with plt.clf() before doing another plot.

In your final answer: summarize these correlations and trends
After each number derive real worlds insights,
for instance: "Correlation between is_december and boredness is 1.3453,
which suggest people are more bored in winter".
Your final answer should have at least 3 numbered and detailed parts.
"""

analysis = agent.run(
    task=agent_prompt,
    additional_notes=additional_notes,
    source_file="titanic/train.csv",
)
print(analysis)

analysis_str = json.dumps(analysis, indent=4)
output = agent.run(
    """You are an expert machine learning engineer.
Please train a ML model on "titanic/train.csv" to predict the survival for rows of "titanic/test.csv".
Output the results under './output.csv'.
Take care to import functions and modules before using them!
""",
    additional_notes=additional_notes + "\n" + analysis_str,
)

print(output)
