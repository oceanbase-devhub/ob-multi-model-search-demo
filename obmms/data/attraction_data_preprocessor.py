import os
import pandas as pd
import requests
import json
from typing import List
import random
import logging
import re
import time
import dotenv
import traceback

from pyobvector import *
from sqlalchemy import Column, Integer, String, JSON, Index
from sqlalchemy.dialects.mysql import LONGTEXT
from tqdm import tqdm
import dashscope
from http import HTTPStatus

dotenv.load_dotenv()

DEFAULT_OBMMS_TABLE_NAME = "obmms_demo"
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def geocode(address):
    params = {
        'address': address,
        'key': os.getenv("AMAP_API_KEY", ""),
    }
    url = 'https://restapi.amap.com/v3/geocode/geo'

    while True:
        res = requests.get(url, params)
        result = json.loads(res.text)
        try:
            long_lat_strs = result['geocodes'][0]['location'].split(",")
        except KeyError:
            if result['info'] == 'CUQPS_HAS_EXCEEDED_THE_LIMIT':
                time.sleep(2)
                continue
            else:
                raise KeyError(f"{address} {res} {result}")
        return (float(long_lat_strs[1]), float(long_lat_strs[0]))


def embedding(query: List[str]):
    res = dashscope.TextEmbedding.call(
        model=dashscope.TextEmbedding.Models.text_embedding_v3,
        input=query,
        api_key=os.getenv("DASHSCOPE_API_KEY"),
    )
    if res.status_code == HTTPStatus.OK:
        return [eb['embedding'] for eb in res.output['embeddings']]
    else:
        raise ValueError(f"embedding error: {res}")

def parse_season_str(season_str: str) -> int:
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


def create_obmms_table():
    ssl_ca_path = os.getenv("OB_DB_SSL_CA_PATH")
    if ssl_ca_path is not None:
        connect_args = {
            "ssl_ca": ssl_ca_path,
        }
    uri = os.getenv("OB_URL", "127.0.0.1:2881")
    user = os.getenv("OB_USER", "root@test")
    db_name = os.getenv("OB_DB_NAME", "test")
    pwd = os.getenv("OB_PWD", "")
    if ssl_ca_path:
        client = ObVecClient(
            uri=uri,
            user=user,
            password=pwd,
            db_name=db_name,
            connect_args=connect_args,
        )
    else:
        client = ObVecClient(
            uri=uri,
            user=user,
            password=pwd,
            db_name=db_name,
        )
    if client.check_table_exists(table_name=DEFAULT_OBMMS_TABLE_NAME):
        return
    
    cols = [
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("attraction_name", String(1024), nullable=False),
        Column("address_text", LONGTEXT, nullable=False),
        Column("address", POINT(srid=4326), nullable=False),
        Column("intro", LONGTEXT, nullable=False),
        Column("intro_vec", VECTOR(1024), nullable=False),
        Column("img_url", String(1024), nullable=False),
        Column("score", Integer, nullable=False),
        Column("season", Integer, nullable=False),
        # Column("season_vec", VECTOR(1024), nullable=False),
        Column("ticket", JSON),
    ]

    indexes = [
        Index("address_idx", "address"),
        VectorIndex(
            "intro_vidx",
            "intro_vec",
            params="distance=l2, type=hnsw, lib=vsag"
        ),
        # VectorIndex(
        #     "season_vidx",
        #     "season_vec",
        #     params="distance=l2, type=hnsw, lib=vsag"
        # )
    ]

    client.create_table(
        table_name=DEFAULT_OBMMS_TABLE_NAME,
        columns=cols,
        indexes=indexes
    )


def load_csv(csv_path: str, delete_after_loaded: bool = False):
    ssl_ca_path = os.getenv("OB_DB_SSL_CA_PATH")
    if ssl_ca_path is not None:
        connect_args = {
            "ssl_ca": ssl_ca_path,
        }
    uri = os.getenv("OB_URL", "127.0.0.1:2881")
    user = os.getenv("OB_USER", "root@test")
    db_name = os.getenv("OB_DB_NAME", "test")
    pwd = os.getenv("OB_PWD", "")
    if ssl_ca_path:
        client = ObVecClient(
            uri=uri,
            user=user,
            password=pwd,
            db_name=db_name,
            connect_args=connect_args,
        )
    else:
        client = ObVecClient(
            uri=uri,
            user=user,
            password=pwd,
            db_name=db_name,
        )
    df = pd.read_csv(csv_path)

    pattern = r'地址:\n(.*?)\n'
    for _, record in tqdm(df.iterrows(), total=df.shape[0]):
        if pd.isna(record["介绍"]) or pd.isna(record["图片链接"]):
            continue
        
        season_str = "四季皆宜" if pd.isna(record["建议季节"]) else record["建议季节"]
        match = re.search(pattern, str(record["地址"]), re.DOTALL)
        if not match:
            continue
        geo_str = match.group(1)

        try:
            lat_long = geocode(geo_str)
        except Exception:
            continue
        
        data = {
            "attraction_name": record["名字"],
            "address_text": record["地址"],
            "address": ST_GeomFromText(lat_long, 4326),
            "intro": record["介绍"],
            "intro_vec": embedding([record["介绍"]])[0],
            "img_url": record["图片链接"],
            "score": random.randint(95, 100),
            "season": parse_season_str(season_str),
            # "season_vec": embedding([season_str])[0],
            "ticket": None if pd.isna(record["门票"]) else record["门票"]
        }
        client.insert(table_name=DEFAULT_OBMMS_TABLE_NAME, data=data)
    
    if delete_after_loaded:
        try:
            os.remove(csv_path)
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    create_obmms_table()
    for root, _, files in os.walk("./citydata"):
        for file in files:
            file_path = os.path.join(root, file)
            print(f"{file_path}:")
            load_csv(file_path, delete_after_loaded=True)
    # load_csv("./citydata/杭州.csv")
