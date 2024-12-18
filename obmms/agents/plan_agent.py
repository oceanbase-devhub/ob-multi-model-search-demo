import re
from typing import Dict, List, Any
from ..prompt import plan_prompt
from .agent import Agent
from ..llm import TongyiLLMConfig, TongyiLLM
from ..tools import ObMMSTool
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class PlanAgent(Agent):
    def __init__(self, obmms_tool: ObMMSTool, enable_stream: bool = False):
        self.enable_stream = enable_stream
        config = TongyiLLMConfig(
            llm_name='qwen-plus',
            stream=enable_stream,
            incremental_output=enable_stream
        )
        llm = TongyiLLM(config=config)
        super().__init__(llm=llm, tool=obmms_tool)
    
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

    def chat(
        self,
        necessary_info: Dict[str, Any],
        chat_history: List[dict],
        summary: str,
        user_content: str,
        **kwargs
    ) -> str:
        res = self.tool.call(
            necessary_info=necessary_info,
            summary=summary,
        )

        attraction_infos = ""
        geos = None
        idx = 1
        for r in res.fetchall():
            if geos is None:
                geos = []
            info = ''.join([str(col) for col in (list(r))[:-1]])
            geos.append(PlanAgent.extract_floats(str((list(r))[-1])))
            attraction_infos += f"景点{idx}: {info}\n\n"
            idx += 1

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
            ), geos
        
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
        return resp, geos
