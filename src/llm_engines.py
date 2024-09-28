from transformers.agents.llm_engine import MessageRole, get_clean_message_list
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class OpenAIEngine:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ["DEEPSEEK_BASE_URL"],
            api_key=os.environ["DEEPSEEK_API_KEY"],
        )

    def __call__(self, messages, stop_sequences=[], grammar=None):
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
