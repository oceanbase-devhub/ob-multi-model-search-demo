from typing import List
from ..prompt import summary_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SummaryAgent(Agent):
    def __init__(self):
        config = TongyiLLMConfig(llm_name='qwen-plus')
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm)
    
    def chat(
        self,
        chat_history: List[dict],
        user_content: str,
        **kwargs
    ) -> str:
        prompted_message = summary_prompt.format(
            user_content=user_content,
        )

        logger.info(prompted_message)

        retry_cnt = 0
        while retry_cnt < self.DEFAULT_RETRY_COUNT:
            ret_code, resp, _ = self.llm.multi_chat(
                messages=chat_history,
                user_content=prompted_message,
                pure_user_content=user_content,
                use_for_history=False,
            )
            logger.info(f"################################### {resp}")
            if ret_code == 200:
                break
            retry_cnt += 1
        
        if ret_code != 200:
            raise ValueError(f"failed to chat with LLM: err-{ret_code}")
        return resp
