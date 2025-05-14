import streamlit as st
import pandas as pd
from obmms import AgentFlow

st.set_page_config(
    page_title="旅行规划助手",
    layout="wide"
)

col1, col2 = st.columns([1, 1])

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "lats" not in st.session_state:
    st.session_state["lats"] = [39.9042]

if "longs" not in st.session_state:
    st.session_state["longs"] = [116.4074]

# if "locations" not in st.session_state:
#     st.session_state["locations"] = "北京"

if "agents" not in st.session_state:
    st.session_state["agents"] = AgentFlow(
        table_name="obmms_demo",
        topk=20,
        # echo=True,
        enable_stream=True,
    )

with col1:
    st.header("景点地图")
    data = pd.DataFrame({
        'latitude': st.session_state["lats"],  # 示例坐标
        'longitude': st.session_state["longs"],
        # 'city': st.session_state["locations"]
    })
    st.map(data)
    
def gen_stream_resp(resp, msg):
    for res in resp:
        msg.append(res.output.choices[0].message.content)
        yield res.output.choices[0].message.content

avatar_m = {
    "assistant": "🌏",
    "user": "🧑‍💻",
}

with col2:
    st.header("旅行咨询")
    prompt = st.chat_input("输入你的消息...")
    
    with st.container(height=600):
        for msg in st.session_state.messages:
            st.chat_message(msg["role"], avatar=avatar_m[msg["role"]]).write(msg["content"])

        msg = []
        if prompt is not None:
            st.chat_message("user", avatar=avatar_m["user"]).write(prompt)
            
            resp, geo, _, _, _, _ = st.session_state["agents"].chat(user_content=prompt)
            while resp is None:
                resp, geo, _, _, _, _ = st.session_state["agents"].chat(user_content=prompt)
            st.chat_message("assistant", avatar=avatar_m["assistant"]).write_stream(
                gen_stream_resp(resp, msg)
            )

            if geo is not None:
                st.session_state["lats"] = [p[0] for p in geo]
                st.session_state["longs"] = [p[1] for p in geo]
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({"role": "assistant", "content": ''.join(msg)})

            st.rerun()

# 定制杭州周边游
# 景点评分要求96分以上,希望在秋季游玩,我偏好自然风光尤其是湖泊