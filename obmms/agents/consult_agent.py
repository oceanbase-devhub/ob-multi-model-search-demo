from typing import List
from ..prompt import consult_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ConsultAgent(Agent):
    def __init__(self, enable_stream: bool = False):
        self.enable_stream = enable_stream
        config = TongyiLLMConfig(
            llm_name='qwen-plus',
            stream=enable_stream,
            incremental_output=enable_stream
        )
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm)
    
    def chat(
        self,
        necessay_list: List[str],
        chat_history: List[dict],
        user_content: str,
        **kwargs
    ) -> str:
        necessary_str = ""
        for idx, nec in enumerate(necessay_list):
            necessary_str += f"{idx + 1}. {nec}\n"
        
        prompted_message = consult_prompt.format(
            attraction_info=necessary_str,
            user_content=user_content,
        )

        logger.info(prompted_message)

        if self.enable_stream:
            return self.llm.multi_chat(
                messages=chat_history,
                user_content=prompted_message,
                pure_user_content=user_content,
            )

        retry_cnt = 0
        while retry_cnt < self.DEFAULT_RETRY_COUNT:
            # print(f"{chat_history} {prompted_message} {user_content}")
            ret_code, resp, _ = self.llm.multi_chat(
                messages=chat_history,
                user_content=prompted_message,
                pure_user_content=user_content,
            )
            if ret_code == 200:
                break
            retry_cnt += 1

        if ret_code != 200:
            raise ValueError(f"failed to chat with LLM: err-{ret_code}")
        return resp
