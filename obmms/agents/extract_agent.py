from typing import List
from ..prompt import extract_info_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ExtractAgent(Agent):
    def __init__(self):
        config = TongyiLLMConfig(llm_name='qwen-plus')
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm)

    def chat(
        self,
        user_content: str,
        **kwargs
    ) -> dict:
        prompted_message = extract_info_prompt.format(
            user_info=user_content,
        )

        logger.info(prompted_message)

        retry_cnt = 0
        while retry_cnt < self.DEFAULT_RETRY_COUNT:
            resp = self.llm.chat(input_prompt=prompted_message)
            if resp.status_code == 200:
                try:
                    json_str = str(resp.output.text).replace(" ", "").replace("\n", "")
                    logger.info(f"============ resp: {json_str}")
                    res = json.loads(json_str)
                    break
                except Exception as e:
                    retry_cnt += 1
            else:
                retry_cnt += 1

        if resp.status_code != 200:
            raise ValueError(f"failed to chat with LLM: err-{resp.status_code}")
        
        if retry_cnt >= self.DEFAULT_RETRY_COUNT:
            raise ValueError(f"failed to parse result to JSON")
        
        return res
