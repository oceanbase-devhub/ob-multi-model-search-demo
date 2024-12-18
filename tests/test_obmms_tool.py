import unittest
from obmms import ObMMSTool
import logging

logger = logging.getLogger(__name__)

class ObMMSToolTest(unittest.TestCase):
    def test_basic(self):
        obm_tool = ObMMSTool(
            table_name="obmms_demo",
            topk=20,
            # echo=True,
        )

        res = obm_tool.call(
            necessary_info={
                "departure": "杭州市西湖区",
                "distance": "10km",
                "score": "96",
                "season": "秋",
            },
            summary="杭州秋景天花板"
        )
        for r in res.fetchall():
            logger.info(r)
