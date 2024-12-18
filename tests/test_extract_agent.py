import unittest
from obmms import ExtractAgent
import logging

logger = logging.getLogger(__name__)


class ExtractAgentTest(unittest.TestCase):
    def test_basic(self):
        extra_agent = ExtractAgent()
        res = extra_agent.chat("我想了解一下杭州的秋景")
        logger.info(res)
    
    def test_mismatch(self):
        extra_agent = ExtractAgent()
        res = extra_agent.chat("今天天气不错")
        logger.info(res)

    def test_distance(self):
        extra_agent = ExtractAgent()
        res = extra_agent.chat("规划杭州市内游")
        logger.info(res)

        res = extra_agent.chat("浙江省内游")
        logger.info(res)
