import os
import dashscope
# from config import RagConfig
from typing import List
from .llm import LLM, LLMConfig

class TongyiLLMConfig(LLMConfig):
    """"""
    multi_chat_max_rounds: int = 3

class TongyiLLM(LLM):
    def __init__(self, config: TongyiLLMConfig) -> None:
        self._config = config

    def chat(self, input_prompt: str) -> str:
        response = dashscope.Generation.call(
            model=self._config.llm_name,
            prompt=input_prompt,
            history=None,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=self._config.stream,
            incremental_output=self._config.incremental_output,
            top_p=self._config.top_p,
            temperature=self._config.temperature,
        )
        return response
        # if response.status_code == 200:
        #     return response.output.text
        # else:
        #     print(response.code, response.message)
        #     raise ValueError(f"{response.code}: {response.message}")
    
    def multi_chat(
        self,
        messages: List[dict],
        user_content,
        pure_user_content,
        use_for_history: bool = True,
    ):
        new_msgs = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
        new_msgs.extend(messages)
        new_msgs.append({
            'role': 'user',
            'content': user_content,
        })
        response = dashscope.Generation.call(
            model=self._config.llm_name,
            messages=new_msgs,
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            stream=self._config.stream,
            incremental_output=self._config.incremental_output,
            top_p=self._config.top_p,
            temperature=self._config.temperature,
            result_format='message'
        )
        if self._config.stream:
            return response
        
        if response.status_code == 200:
            if use_for_history:
                messages.append({
                    'role': 'user',
                    'content': pure_user_content,
                })
                messages.append({
                    'role': response.output.choices[0]['message']['role'],
                    'content': response.output.choices[0]['message']['content']
                })
            return response.status_code, response.output.choices[0]['message']['content'], messages
        else:
            return response.status_code, "", messages

