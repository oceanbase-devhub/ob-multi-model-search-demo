# OceanBase Multi-Model Search Demo

## Setup

1. Deploy a standalone OceanBase server with docker:

```bash
docker run --name=ob433 -e MODE=mini -e OB_MEMORY_LIMIT=8G -e OB_DATAFILE_SIZE=10G -e OB_CLUSTER_NAME=ailab2024_dbgpt -e OB_SERVER_IP=127.0.0.1 -p 2881:2881 -d quay.io/oceanbase/oceanbase-ce:4.3.3.1-101000012024102216
```

You can also use a [free OceanBase cloud instance](https://www.oceanbase.com/free-trial)

2. Create `.env` file in this project directory and set configurations.

```bash
vim .env
```

```plain
OB_URL="your-ob-url"
OB_USER="your-ob-user"
OB_DB_NAME="your-db-name"
OB_PWD="your-pwd"
OB_DB_SSL_CA_PATH="optional-ssl-ca-path"
```

3. Visit https://www.aliyun.com/product/bailian to obtain the model service API key and save it in the `DASHSCOPE_API_KEY` system variable.

4. Visit https://lbs.amap.com/ to obtain the map service API key and save it in the `AMAP_API_KEY` system variable.

5. Visit https://www.kaggle.com/datasets/audreyhengruizhang/china-city-attraction-details to obtain the dataset and store it in the manualy created `citydata` directory under this project directory.

6. (Optional but Recomanded) Create an Python3.10 environment with [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

```bash
conda create -n obmms python=3.10 && conda activate obmms
```

7. Install this python project with `poetry install`.

8. Import datas into OceanBase with following command:

```bash
python ./obmms/data/attraction_data_preprocessor.py
```

9. Start chat server with following command:

```bash
streamlit run ./ui.py
```
