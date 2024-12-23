import re
import logging
import time
from typing import Dict, Optional, List, Tuple
from enum import Enum
from .agent_flow import AgentStat
from ..agents import ExtractAgent, ConsultAgent, SummaryAgent, PlanAgent
from ..tools import ObMMSTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Response(BaseModel):
    # next_stat: int = Field(description="告知前端下一次的状态码")
    reply: str = Field('', description="本轮大模型输出")
    need_reset: bool = Field(False, description="前端是否需要重置历史对话")
    sql: Optional[str] = Field(None, description="如果本次回复使用了SQL，会通过这个字段返回")
    datas: Optional[List[dict]] = Field(None, description="如果本次回复使用了SQL，查询结果会通过这个字段返回")
    lats: Optional[List[float]] = Field(None, description="如果本次回复会发生地图坐标动态变化，lats非空，包含一组维度值")
    longs: Optional[List[float]] = Field(None, description="如果本次回复会发生地图坐标动态变化，longs非空，包含一组维度值")
    departure: Optional[str] = Field(None, description="前端保存的旅行出发地")
    distance: Optional[str] = Field(None, description="前端保存的行程范围")
    score: Optional[int] = Field(None, description="前端保存的景点评分")
    season: Optional[str] = Field(None, description="前端保存的季节要求")

class StatelessAgentFlow:
    def __init__(
        self,
        table_name: str,
        topk: int,
        chat_history: List[dict],
        # stat: AgentStat,
        departure: Optional[str] = None,
        distance: Optional[str] = None,
        score: Optional[int] = None,
        season: Optional[str] = None,
        echo: bool = False,
    ):
        self.departure_name = "departure"
        self.distance_name = "distance"
        self.score_name = "score"
        self.season_name = "season"
        self.departure_name_cn = "旅行目的省市"
        self.distance_name_cn = "行程范围"
        self.score_name_cn = "景点评分"
        self.season_name_cn = "出行季节"
        self.necessary_str_map = {
            self.departure_name: self.departure_name_cn,
            self.distance_name: self.distance_name_cn,
            self.score_name: self.score_name_cn,
            self.season_name: self.season_name_cn,
        }

        self.stat = AgentStat.STAT_EXTRACT
        self.chat_history = chat_history
        # self.messages = messages[:-1]
        # self.user_content = messages[-1]
        self.user_info = {
            self.departure_name: departure,
            self.distance_name: distance,
            self.score_name: score,
            self.season_name: season,
        }

        self.extract_agent = ExtractAgent()
        self.consult_agent = ConsultAgent(enable_stream=True)
        self.summary_agent = SummaryAgent()
        self.obmms_tool = ObMMSTool(
            table_name=table_name,
            topk=topk,
            echo=echo,
        )
        self.plan_agent = PlanAgent(
            obmms_tool=self.obmms_tool,
            enable_stream=True,
        )

    @classmethod
    def parse_season_str(cls, season_str: str) -> int:
        if "四季" in season_str or "全年" in season_str:
            return (1 << 4) - 1
        res = 0
        if "春" in season_str:
            res |= 1
        if "夏" in season_str:
            res |= (1 << 1)
        if "秋" in season_str:
            res |= (1 << 2)
        if "冬" in season_str:
            res |= (1 << 3)
        if res == 0:
            return (1 << 4) - 1
        return res
    
    def update_user_info(self, new_json):
        if new_json[self.departure_name] is not None:
            self.user_info[self.departure_name] = new_json[self.departure_name]
        if new_json[self.distance_name] is not None:
            self.user_info[self.distance_name] = new_json[self.distance_name]
        if new_json[self.score_name] is not None:
            if self.user_info[self.score_name] is None:
                self.user_info[self.score_name] = new_json[self.score_name]
            else:
                self.user_info[self.score_name] = max(self.user_info[self.score_name], new_json[self.score_name])
        if new_json[self.season_name] is not None:
            if self.user_info[self.season_name] is None:
                self.user_info[self.season_name] = new_json[self.season_name]
            else:
                self.user_info[self.season_name] = self.user_info[self.season_name] + new_json[self.season_name]
    
    def get_none_user_info_keys(self):
        necessary_list = []
        for k, v in self.user_info.items():
            if v is None:
                necessary_list.append(self.necessary_str_map[k])
        return necessary_list
    
    def set_next_stat(self):
        if self.stat == AgentStat.STAT_EXTRACT:
            necessary_list = self.get_none_user_info_keys()
            if len(necessary_list) == 0:
                self.stat = AgentStat.STAT_SUMMARY
            else:
                self.stat = AgentStat.STAT_CONSULT
        elif self.stat == AgentStat.STAT_CONSULT:
            raise ValueError("CONSULT state is an ending state.")
        elif self.stat == AgentStat.STAT_SUMMARY:
            self.stat = AgentStat.STAT_PLAN
        elif self.stat == AgentStat.STAT_PLAN:
            raise ValueError("PLAN state is an ending state.")

    def replace_folded_vectors(self, text):
        return re.sub(r'\'\[.*?\]\'', '<FOLDED VECTOR DATA>', text)

    def chat(self, user_content: str):
        chat_resp = Response()
        summary_resp = ""

        while True:
            if self.stat == AgentStat.STAT_EXTRACT:
                new_json = self.extract_agent.chat(
                    user_content=user_content
                )
                self.update_user_info(new_json=new_json)
                self.set_next_stat()
                continue
            elif self.stat == AgentStat.STAT_CONSULT:
                streamer = self.consult_agent.chat(
                    necessay_list=self.get_none_user_info_keys(),
                    chat_history=self.chat_history,
                    user_content=user_content,
                )
                
                # chat_resp.reply = self.chat_history[-1]
                if self.user_info[self.departure_name] is not None:
                    lat, long = self.obmms_tool.geocode(self.user_info[self.departure_name])
                    chat_resp.lats = [lat]
                    chat_resp.longs = [long]
                chat_resp.departure = self.user_info[self.departure_name]
                chat_resp.distance = self.user_info[self.distance_name]
                chat_resp.score = self.user_info[self.score_name]
                chat_resp.season = self.user_info[self.season_name]
                return streamer, chat_resp
            elif self.stat == AgentStat.STAT_SUMMARY:
                summary_resp = self.summary_agent.chat(
                    chat_history=self.chat_history,
                    user_content=user_content,
                )
                self.set_next_stat()
                continue
            elif self.stat == AgentStat.STAT_PLAN:
                sql_stmts = []
                result_column_names = []
                result_rows = []

                streamer, geos, _ = self.plan_agent.chat(
                    necessary_info=self.user_info,
                    chat_history=self.chat_history,
                    summary=summary_resp,
                    user_content=user_content,
                    str_list=sql_stmts,
                    result_column_names=result_column_names,
                    result_rows=result_rows,
                )

                # chat_resp.reply = self.chat_history[-1]
                chat_resp.need_reset = True
                if geos is not None:
                    chat_resp.lats = [p[0] for p in geos]
                    chat_resp.longs = [p[1] for p in geos]
                else:
                    chat_resp.lats = []
                    chat_resp.longs = []
                
                assert len(sql_stmts) == 1

                chat_resp.sql = self.replace_folded_vectors(sql_stmts[0])
                datas = []

                for row in result_rows:
                    datas.append(dict(zip(result_column_names, row)))
                
                chat_resp.datas = datas
                chat_resp.departure = self.user_info[self.departure_name]
                chat_resp.distance = self.user_info[self.distance_name]
                chat_resp.score = self.user_info[self.score_name]
                chat_resp.season = self.user_info[self.season_name]
                return streamer, chat_resp
        