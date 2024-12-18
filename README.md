# OceanBase Multi-Model Search Demo

## Setup

1. Deply a standalone OceanBase server with docker:

```bash
docker run --name=ob433 -e MODE=mini -e OB_MEMORY_LIMIT=8G -e OB_DATAFILE_SIZE=10G -e OB_CLUSTER_NAME=ailab2024_dbgpt -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d quay.io/oceanbase/oceanbase-ce:4.3.3.1-101000012024102216
```

2. Visit https://www.aliyun.com/product/bailian to obtain the model service API key and save it in the `DASHSCOPE_API_KEY` system variable.

3. Visit https://lbs.amap.com/ to obtain the map service API key and save it in the `AMAP_API_KEY` system variable.

4. Visit https://www.kaggle.com/datasets/audreyhengruizhang/china-city-attraction-details to obtain the dataset and store it in the manualy created `citydata` directory under this project directory.

5. Install this python project with `poetry install`.

6. Import datas into OceanBase with following command:

```bash
python ./obmms/data/attraction_data_preprocessor.py
```

7. Start chat server with following command:

```bash
streamlit run ./ui.py
```
