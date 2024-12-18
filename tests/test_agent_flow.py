import unittest
from obmms import AgentFlow
import logging

logger = logging.getLogger(__name__)


class AgentFlowTest(unittest.TestCase):
    def test_basic(self):
        table_name = 'obmms_demo'
        agent_flow = AgentFlow(
            table_name=table_name,
            topk=20,
            # echo=True,
            enable_stream=True,
        )

        while True:
            user_input = input("> ")
            resp, _ = agent_flow.chat(user_content=user_input)
            print("===================================\n")
            for res in resp:
                print(res.output.choices[0].message.content, end='', flush=True)
            print("\n===================================\n")

        # 定制杭州周边游
        # 景点评分要求96分以上,希望在秋季游玩,我偏好自然风光尤其是湖泊
