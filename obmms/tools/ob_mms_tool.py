import os
import re
import requests
import json
import logging
from typing import Dict, List, Any, Optional
from pyobvector import ObVecClient, ST_GeomFromText, st_dwithin
from .tool import Tool
from sqlalchemy import Table, func, and_, text
import dashscope
from http import HTTPStatus

DEFAULT_DEPATURE_NAME = "departure"
DEFAULT_DISTANCE_NAME = "distance"
DEFAULT_SCORE_NAME = "score"
DEFAULT_SEASON_NAME = "season"
logger = logging.getLogger(__name__)

class ObMMSTool(Tool):
    def __init__(
        self,
        table_name: str,
        topk: int,
        echo: bool = False,
        departure_name: str = DEFAULT_DEPATURE_NAME,
        distance_name: str = DEFAULT_DISTANCE_NAME,
        score_name: str = DEFAULT_SCORE_NAME,
        season_name: str = DEFAULT_SEASON_NAME,
        **kwargs,
    ):
        self.table_name = table_name
        self.client = ObVecClient(echo=echo, **kwargs)
        self.topk = topk
        self.departure_name = departure_name
        self.distance_name = distance_name
        self.score_name = score_name
        self.season_name = season_name
    
    @classmethod
    def embedding(cls, query: str):
        res = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v3,
            input=[query],
        )
        if res.status_code == HTTPStatus.OK:
            return [eb['embedding'] for eb in res.output['embeddings']][0]
        else:
            raise ValueError(f"embedding error: {res}")

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

    @classmethod
    def geocode(cls, address):
        params = {
            'address': address,
            'key': os.environ.get("AMAP_API_KEY", ""),
        }
        url = 'https://restapi.amap.com/v3/geocode/geo'
        res = requests.get(url, params)
        result = json.loads(res.text)
        print(result)
        long_lat_strs = result['geocodes'][0]['location'].split(",")
        return (float(long_lat_strs[1]), float(long_lat_strs[0]))
    
    @classmethod
    def parse_distance(cls, distance_str) -> float:
        import re
        print(f"=================== distance: {distance_str}")
        match = re.match(r'([\d.]+)([a-zA-Z]+)', distance_str)
        if not match:
            raise ValueError("Invalid distance format")
        
        value, unit = match.groups()
        value = float(value)
        
        if unit.lower() == 'm':
            return value
        elif unit.lower() == 'km':
            return value * 1000
        elif unit.lower() == 'cm':
            return value / 100
        elif unit.lower() == 'mm':
            return value / 1000
        else:
            raise ValueError("Unsupported unit")

    def call(
        self,
        necessary_info: Dict[str, Any],
        summary: str,
        str_list: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        departure = ObMMSTool.geocode(necessary_info[self.departure_name])
        distance_str = necessary_info[self.distance_name]
        distance = ObMMSTool.parse_distance(distance_str)
        season = ObMMSTool.parse_season_str(necessary_info[self.season_name])
        score = necessary_info[self.score_name]
        summary_vec = ObMMSTool.embedding(summary)

        table = Table(
            self.table_name,
            self.client.metadata_obj,
            autoload_with=self.client.engine
        )

        where_clause = [
            and_(
                text(f"score >= {score} AND season & {season} = {season}"),
                st_dwithin(table.c["address"], ST_GeomFromText(departure, 4326), distance),
            ),
        ]

        res = self.client.post_ann_search(
            table_name=self.table_name,
            vec_data=summary_vec,
            vec_column_name="intro_vec",
            distance_func=func.l2_distance,
            with_dist=False,
            topk=self.topk,
            output_column_names=[
                "attraction_name",
                "address_text",
                "intro",
                "score",
                "season",
                "ticket",
            ],
            extra_output_cols=[text("st_astext(address)")],
            where_clause=where_clause,
            str_list=str_list,
        )

        return res
