from abc import ABC, abstractmethod

from openai import OpenAI
from openai.types import ChatCompletion


class LLM(ABC):
    @abstractmethod
    def setup(self, model: str, system_prompt: str, tools: list | None = None) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str) -> tuple[ChatCompletion, str]:
        pass


class OpenAILLM(LLM):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.ready = False

    def setup(self,
              model: str = 'gpt-4o-mini',
              system_prompt: str = 'You are a helpful assistant.',
              tools: list | None = None) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.tools = tools or []
        self.ready = True

    def generate(self, prompt: str) -> tuple[ChatCompletion, str]:
        if not self.ready:
            raise ValueError("LLM is not ready. Call setup() first.")

        messages = self.messages + [{'role': 'user', 'content': prompt}]
        kwargs = {'model': self.model, 'messages': messages}
        if self.tools:
            kwargs['tools'] = self.tools

        response = self.client.chat.completions.create(**kwargs)
        return response, response.choices[0].message.content
