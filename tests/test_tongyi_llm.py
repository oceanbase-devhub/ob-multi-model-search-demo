import unittest
from obmms import TongyiLLMConfig, TongyiLLM
import logging

logger = logging.getLogger(__name__)


class TongyiLLMTest(unittest.TestCase):
    def test_non_stream_chat(self):
        config = TongyiLLMConfig(llm_name='qwen-plus')
        llm = TongyiLLM(config=config)
        resp = llm.chat("Hello")
        logger.info(resp.output.text)
    
    def test_stream_chat(self):
        config = TongyiLLMConfig(llm_name='qwen-plus', stream=True, incremental_output=True)
        llm = TongyiLLM(config=config)
        resp = llm.chat("你好")

        # for res in resp:
        #     print(res.output.text, end='', flush=True)
    
    def test_stream_chat(self):
        config = TongyiLLMConfig(llm_name='qwen-plus', stream=True, incremental_output=True)
        message = []
        llm = TongyiLLM(config=config)

        resp = llm.multi_chat(messages=message, user_content="你好", pure_user_content="你好")
        # for res in resp:
        #     print(res.output.choices[0].message.content, end='', flush=True)
