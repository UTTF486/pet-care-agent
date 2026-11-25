import os
import unittest

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "local"

from config import Settings  # noqa: E402
from src.graph import _contextual_query, _extract_profile  # noqa: E402
from src.knowledge import _stable_id  # noqa: E402
from src.recommender import ServiceRecommender  # noqa: E402


class AgentCoreTest(unittest.TestCase):

    def test_profile_keeps_newest_issues(self):
        profile = _extract_profile(
            "最近又开始掉毛",
            {"pet_type": "猫", "issues": "呕吐,发烧,不吃"},
        )
        self.assertEqual(profile["issues"], "掉毛,呕吐,发烧")

    def test_contextual_query_contains_previous_slots(self):
        query = _contextual_query(
            "猫",
            {"pet_type": "猫", "age": "2岁", "issues": "掉毛"},
            [{"role": "user", "content": "下周出差，需要预约上门"}],
        )
        self.assertIn("2岁", query)
        self.assertIn("掉毛", query)
        self.assertIn("出差", query)

    def test_stable_id_is_repeatable(self):
        self.assertEqual(
            _stable_id("doc", "title", "content", "0"),
            _stable_id("doc", "title", "content", "0"),
        )

    def test_forced_provider_without_key_falls_back(self):
        configured = Settings(
            llm_provider="deepseek",
            deepseek_api_key="",
            dashscope_api_key="",
            ark_api_key="",
            compat_api_key="",
        )
        self.assertEqual(configured.effective_provider, "mock")
        self.assertFalse(configured.llm_available)

    def test_health_recommendation_requires_rule_match(self):
        recommender = ServiceRecommender()
        question = "狗狗拉稀两天了，精神还可以"
        recommended = recommender.recommend(
            question,
            {"pet_type": "狗"},
            top_n=3,
            require_match=True,
        )
        self.assertTrue(recommended)
        self.assertTrue(all("拉稀" in item["reason"] for item in recommended))


if __name__ == "__main__":
    unittest.main()
