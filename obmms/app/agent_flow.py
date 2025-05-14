import logging
import time
from typing import Dict, Optional, List, Tuple
from enum import Enum
from ..agents import ExtractAgent, ConsultAgent, SummaryAgent, PlanAgent
from ..tools import ObMMSTool

logger = logging.getLogger(__name__)
# file_handler = logging.FileHandler('./log/app.log')
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

class AgentStat(Enum):
    STAT_EXTRACT = 0
    STAT_CONSULT = 1
    STAT_SUMMARY = 2
    STAT_PLAN = 3

class AgentFlow:
    def __init__(
        self,
        table_name: str,
        topk: int,
        echo: bool = False,
        enable_stream: bool = False,
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
        self.messages = []
        self.user_info = {
            self.departure_name: None,
            self.distance_name: None,
            self.score_name: None,
            self.season_name: None,
        }
        self.extract_agent = ExtractAgent()
        self.consult_agent = ConsultAgent(enable_stream=enable_stream)
        self.summary_agent = SummaryAgent()
        self.obmms_tool = ObMMSTool(
            table_name=table_name,
            topk=topk,
            echo=echo,
        )
        self.plan_agent = PlanAgent(
            obmms_tool=self.obmms_tool,
            enable_stream=enable_stream
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
    
    def reset(self):
        self.stat = AgentStat.STAT_EXTRACT
        self.messages = []
        self.user_info = {
            self.departure_name: None,
            self.distance_name: None,
            self.score_name: None,
            self.season_name: None,
        }

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
            self.stat = AgentStat.STAT_EXTRACT
        elif self.stat == AgentStat.STAT_SUMMARY:
            self.stat = AgentStat.STAT_PLAN
        elif self.stat == AgentStat.STAT_PLAN:
            self.reset()

    def chat(
        self,
        user_content: str,
    ):
        summary_resp = ""
        while True:
            if self.stat == AgentStat.STAT_EXTRACT:
                start_time = time.time()
                new_json = self.extract_agent.chat(
                    user_content=user_content
                )
                self.update_user_info(new_json=new_json)
                self.set_next_stat()
                # yield None, None, f"抽取必要查询信息（{time.time() - start_time:.2f}s）", None, None, None
            elif self.stat == AgentStat.STAT_CONSULT:
                logger.info(f"################### current undefined keys: {self.get_none_user_info_keys()}")
                start_time = time.time()
                resp = self.consult_agent.chat(
                    necessay_list=self.get_none_user_info_keys(),
                    chat_history=self.messages,
                    user_content=user_content,
                )
                self.set_next_stat()

                if self.user_info[self.departure_name] is None:
                    geo = None
                else:
                    geo = [self.obmms_tool.geocode(self.user_info[self.departure_name])]
                return resp, geo, f"询问用户必要需求（{time.time() - start_time:.2f}s）", None, None, None
            elif self.stat == AgentStat.STAT_SUMMARY:
                start_time = time.time()
                summary_resp = self.summary_agent.chat(
                    chat_history=self.messages,
                    user_content=user_content,
                )
                print(f"===================== summary ==================\n {summary_resp}\n ===============================\n")
                self.set_next_stat()
                # yield None, None, f"分析用户其他需求（{time.time() - start_time:.2f}s）", None, None, None
            elif self.stat == AgentStat.STAT_PLAN:
                sql_stmts = []
                result_column_names = []
                result_rows = []
                resp, geos, duration = self.plan_agent.chat(
                    necessary_info=self.user_info,
                    chat_history=self.messages,
                    summary=summary_resp,
                    user_content=user_content,
                    str_list=sql_stmts,
                    result_column_names=result_column_names,
                    result_rows=result_rows,
                )
                self.set_next_stat()
                return resp, geos, f"生成景点推荐（{time.time() - start_time:.2f}s）", [f"（{duration:.2f}s）: {sql_stmts[0]}"], result_column_names, result_rows
