import asyncio
from concurrent.futures import ThreadPoolExecutor
import re
import time
from typing import Dict, List, Any, Optional, Tuple
from ..prompt import plan_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
from ..tools import ObMMSTool
import logging

logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler('./log/app.log')
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

class PlanAgent(Agent):
    def __init__(
        self,
        obmms_tool: ObMMSTool,
        enable_stream: bool = False,
        is_async: bool = False,
        search_only: bool = False,
    ):
        self.enable_stream = enable_stream
        config = TongyiLLMConfig(
            llm_name='qwen-plus',
            stream=enable_stream,
            incremental_output=enable_stream
        )
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm, tool=obmms_tool)
        if is_async:
            self.executor = ThreadPoolExecutor(max_workers=8)
        else:
            self.executor = None
        self.search_only = search_only
    
    async def run_blocking_func(self, func, *args):
        if asyncio.iscoroutinefunction(func):
            raise ValueError(f"The function {func} is not blocking function")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)

    @classmethod
    def extract_floats(cls, point_string):
        pattern = r'POINT\(([-+]?\d*\.\d+)\s+([-+]?\d*\.\d+)\)'
        
        match = re.search(pattern, point_string)
        
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            return (x, y)
        else:
            raise ValueError("输入字符串格式不正确")

    async def achat(
        self,
        necessary_info: Dict[str, Any],
        chat_history: List[dict],
        summary: str,
        user_content: str,
        str_list: Optional[List[str]] = None,
        result_column_names: Optional[List] = None,
        result_rows: Optional[List[Tuple]] = None,
    ):
        if self.executor is None:
            raise ValueError("thread pool executor is None")
        
        start_time = time.time()
        res = await self.run_blocking_func(
            self.tool.call,
            necessary_info,
            summary,
            str_list,
        )
        end_time = time.time()

        if result_column_names is not None:
            result_column_names.extend(res.keys())

        attraction_infos = ""
        geos = None
        idx = 1
        for r in res.fetchall():
            if result_rows is not None:
                result_rows.append(r)
            if geos is None:
                geos = []
            info = ''.join([str(col) for col in (list(r))[:-1]])
            geos.append(PlanAgent.extract_floats(str((list(r))[-1])))
            if not self.search_only:
                attraction_infos += f"景点{idx}: {info}\n\n"
            idx += 1
        
        if (len(attraction_infos) == 0) and (not self.search_only):
            attraction_infos = "不存在可选旅行景点"
        
        # print("\n===================================\n")
        # print(attraction_infos)
        # print("\n===================================\n")
        resp = None
        if not self.search_only:
            prompted_message = plan_prompt.format(
                option_attractions=attraction_infos,
                user_content=user_content,
            )

            logger.info(prompted_message)

            if self.enable_stream:
                return await self.run_blocking_func(
                    self.llm.multi_chat,
                    chat_history,
                    prompted_message,
                    user_content,
                ), geos, end_time - start_time
            
            retry_cnt = 0
            while retry_cnt < self.DEFAULT_RETRY_COUNT:
                ret_code, resp, _ = await self.run_blocking_func(
                    self.llm.multi_chat,
                    chat_history,
                    prompted_message,
                    user_content,
                )
                if ret_code == 200:
                    break
                retry_cnt += 1

            if ret_code != 200:
                raise ValueError(f"failed to chat with LLM: err-{ret_code}")
        return resp, geos, end_time - start_time

    def chat(
        self,
        necessary_info: Dict[str, Any],
        chat_history: List[dict],
        summary: str,
        user_content: str,
        str_list: Optional[List[str]] = None,
        result_column_names: Optional[List] = None,
        result_rows: Optional[List[Tuple]] = None,
    ):
        start_time = time.time()
        res = self.tool.call(
            necessary_info=necessary_info,
            summary=summary,
            str_list=str_list,
        )
        end_time = time.time()

        if result_column_names is not None:
            result_column_names.extend(res.keys())

        attraction_infos = ""
        geos = None
        idx = 1
        for r in res.fetchall():
            if result_rows is not None:
                result_rows.append(r)
            if geos is None:
                geos = []
            info = ''.join([str(col) for col in (list(r))[:-1]])
            geos.append(PlanAgent.extract_floats(str((list(r))[-1])))
            attraction_infos += f"景点{idx}: {info}\n\n"
            idx += 1
        
        if len(attraction_infos) == 0:
            attraction_infos = "不存在可选旅行景点"
        
        print("\n===================================\n")
        print(attraction_infos)
        print("\n===================================\n")

        prompted_message = plan_prompt.format(
            option_attractions=attraction_infos,
            user_content=user_content,
        )

        logger.info(prompted_message)

        if self.enable_stream:
            return self.llm.multi_chat(
                messages=chat_history,
                user_content=prompted_message,
                pure_user_content=user_content,
            ), geos, end_time - start_time
        
        retry_cnt = 0
        while retry_cnt < self.DEFAULT_RETRY_COUNT:
            ret_code, resp, _  = self.llm.multi_chat(
                messages=chat_history,
                user_content=prompted_message,
                pure_user_content=user_content,
            )
            if ret_code == 200:
                break
            retry_cnt += 1

        if ret_code != 200:
            raise ValueError(f"failed to chat with LLM: err-{ret_code}")
        return resp, geos, end_time - start_time
