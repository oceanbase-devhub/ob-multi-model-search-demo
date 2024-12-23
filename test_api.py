import json
import requests
from typing import Optional
from obmms import Response

url = "http://localhost:8000/obmms/chat"
messages = []
departure: Optional[str] = None
distance: Optional[str] = None
score: Optional[int] = None
season: Optional[str] = None

def handle_response(resp: Response, new_input: str):
    global messages, departure, distance, score, season
    print("==========================================\n")
    # resp.sql 请添加到前端显示
    if resp.sql:
        print(f"使用sql查询：{resp.sql}")
    # resp.datas 请添加到前端显示
    if resp.datas:
        print("查询结果：")
        for d in resp.datas:
            print(f"{d}")
    # resp.lats 请添加到前端显示
    if resp.lats:
        print(f"景点纬度列表：{resp.lats}")
    # resp.longs 请添加到前端显示
    if resp.longs:
        print(f"景点经度列表：{resp.longs}")
    print("==========================================\n")

    if resp.need_reset:
        messages = []
        departure = None
        distance = None
        score = None
        season = None
        return

    messages.append({
        'role': 'user',
        'content': new_input,
    })
    # messages.append(resp.reply)
    # print(messages)
    departure = resp.departure
    distance = resp.distance
    score = resp.score
    season = resp.season


if __name__ == '__main__':
    while True:
        new_input = input("> ")

        data = {
            "messages": messages,
            "new_input": new_input,
            "departure": departure,
            "distance": distance,
            "score": score,
            "season": season,
        }

        response = requests.post(url, json=data, stream=True)

        if response.status_code == 200:
            # resp_data = response.json()
            try:
                # handle_response(Response.model_validate(resp_data), new_input)
                reply = ""
                for line in response.iter_lines():
                    if line:
                        resp_data: str = line.decode("utf-8")
                        # if 'meta' in resp_data.keys
                        if resp_data.startswith("meta:"):
                            handle_response(
                                Response.model_validate(json.loads(resp_data[5:])),
                                new_input
                            )
                        else:
                            reply += resp_data[5:]
                            print(resp_data[5:], end='',flush=True)

                messages.append({
                    'role': 'assistant',
                    'content': reply,
                })
                print("\n")
            except Exception as e:
                print("handle response error: ", e)
        else:
            print("请求失败，状态码:", response.status_code)
            print("响应内容:", response.text)
