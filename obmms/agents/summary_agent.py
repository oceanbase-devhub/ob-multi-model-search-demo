import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List
from ..prompt import summary_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
import logging

logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler('./log/app.log')
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

class SummaryAgent(Agent):
    def __init__(self, is_async: bool = False):
        config = TongyiLLMConfig(llm_name='qwen-plus')
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm)
        if is_async:
            self.executor = ThreadPoolExecutor(max_workers=8)
        else:
            self.executor = None
    
    async def run_blocking_func(self, func, *args):
        if asyncio.iscoroutinefunction(func):
            raise ValueError(f"The function {func} is not blocking function")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    async def achat(
        self,
        chat_history: List[dict],
        user_content: str,
        **kwargs
    ) -> str:
        if self.executor is None:
            raise ValueError("thread pool executor is None")
        
        prompted_message = summary_prompt.format(
            user_content=user_content,
        )

        logger.info(prompted_message)

        retry_cnt = 0
        while retry_cnt < self.DEFAULT_RETRY_COUNT:
            ret_code, resp, _ = await self.run_blocking_func(
                self.llm.multi_chat,
                chat_history,
                prompted_message,
                user_content,
                False,
            )
            logger.info(f"################################### {resp}")
            if ret_code == 200:
                break
            retry_cnt += 1
        
        if ret_code != 200:
            raise ValueError(f"failed to chat with LLM: err-{ret_code}")
        return resp

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
