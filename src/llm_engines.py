import os

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI
from transformers.agents.llm_engine import MessageRole, get_clean_message_list

load_dotenv()

openai_role_conversions = {MessageRole.TOOL_RESPONSE: MessageRole.USER}


class OpenAIEngine:
    def __init__(self, model_name=None, use_azure=False):
        if model_name is not None:
            self.model_name = model_name
        elif use_azure:
            self.model_name = "4o0806"
        else:
            self.model_name = "deepseek-chat"

        if use_azure:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_API_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  # type: ignore
            )
        else:
            self.client = OpenAI(
                base_url=os.environ["DEEPSEEK_BASE_URL"],
                api_key=os.environ["DEEPSEEK_API_KEY"],
            )

    def __call__(self, messages, stop_sequences=None, grammar=None):
        if stop_sequences is None:
            stop_sequences = []
        messages = get_clean_message_list(
            messages, role_conversions=openai_role_conversions
        )

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5,
            response_format=grammar,
        )
        return response.choices[0].message.content
