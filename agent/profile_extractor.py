"""
画像增量抽取器 - L4 多轮对话核心
LLM 看 (历史画像 + 当前 query) → 输出增量 delta
失败自动降级为空 delta（不污染画像）
"""
import json
import re
from typing import Dict
from langchain_core.messages import HumanMessage


PROMPT_TEMPLATE = """你是用户画像分析师。基于历史画像和最新对话，输出画像【增量更新】。

【历史画像】
{profile_json}

【本轮用户输入】
{query}

【任务】
仅输出本轮的【新变化】，不要重复历史已有信息。

【严格 JSON 输出格式】
{{
  "update_demographics": {{}},
  "add_preferences": {{"styles": [], "colors": [], "scenes": []}},
  "update_constraints": {{}},
  "current_intent": {{"scene": null, "category": null}}
}}

【字段规则】
- update_demographics: 用户明确说性别/年龄时填 {{"gender":"女"}} 等；没说就 {{}}
- add_preferences: 本轮新提到的风格/颜色/场景；历史已有的别重复
- update_constraints: 本轮明确价格；"500以内"→{{"price_max":500}}
- current_intent: 本轮想买啥的场景+品类

【示例1】
历史: {{"demographics":{{"gender":null}}}}
本轮: 我是女性，要上班穿的
输出: {{"update_demographics":{{"gender":"女"}},"add_preferences":{{"scenes":["通勤"]}},"update_constraints":{{}},"current_intent":{{"scene":"通勤","category":null}}}}

【示例2】
历史: {{"demographics":{{"gender":"女"}},"preferences":{{"scenes":["通勤"]}}}}
本轮: 预算500以内
输出: {{"update_demographics":{{}},"add_preferences":{{}},"update_constraints":{{"price_max":500}},"current_intent":{{}}}}

【示例3】闲聊不更新
历史: {{}}
本轮: 你好
输出: {{"update_demographics":{{}},"add_preferences":{{}},"update_constraints":{{}},"current_intent":{{}}}}

只输出 JSON，不要任何解释。

现在处理："""


class ProfileExtractor:
    """LLM 画像增量抽取"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def extract_delta(self, profile: Dict, query: str) -> Dict:
        """
        输入: 当前画像 dict + 用户 query
        输出: delta dict (4 个字段)
        失败: 返回空 delta，不污染画像
        """
        empty = {
            "update_demographics": {},
            "add_preferences": {},
            "update_constraints": {},
            "current_intent": {},
        }
        
        if not query or not query.strip():
            return empty
        
        try:
            prompt = PROMPT_TEMPLATE.format(
                profile_json=json.dumps(self._simplify(profile), ensure_ascii=False),
                query=query,
            )
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, "content") else str(resp)
            
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                print(f"[ProfileExtractor降级] 无 JSON: {text[:80]}")
                return empty
            
            data = json.loads(match.group())
            
            # 字段防御
            for k in empty:
                if k not in data or not isinstance(data[k], dict):
                    data[k] = {}
            
            # null 字符串清洗
            for k, v in list(data["update_demographics"].items()):
                if v in ("null", "None", "", None):
                    data["update_demographics"][k] = None
            for k, v in list(data["update_constraints"].items()):
                if v in ("null", "None", ""):
                    data["update_constraints"][k] = None
            
            # add_preferences 字段防御（必须是 list）
            for k in ["styles", "colors", "scenes"]:
                v = data["add_preferences"].get(k)
                if not isinstance(v, list):
                    data["add_preferences"][k] = []
            
            return data
        
        except Exception as e:
            print(f"[ProfileExtractor降级] {e}")
            return empty
    
    @staticmethod
    def _simplify(profile: Dict) -> Dict:
        """精简画像，节省 token"""
        return {
            "demographics": profile.get("demographics", {}),
            "preferences": profile.get("preferences", {}),
            "constraints": profile.get("constraints", {}),
        }


# ============================================================================
# Mock 自测（无需真实 LLM）
# ============================================================================
if __name__ == "__main__":
    print("🧪 ProfileExtractor Mock 自测")
    print("=" * 60)
    
    class MockLLM:
        responses = [
            '{"update_demographics":{"gender":"女"},"add_preferences":{"scenes":["通勤"]},"update_constraints":{},"current_intent":{"scene":"通勤","category":null}}',
            '{"update_demographics":{},"add_preferences":{},"update_constraints":{"price_max":500},"current_intent":{}}',
            '{"update_demographics":{},"add_preferences":{"colors":["黑色"],"styles":["简约"]},"update_constraints":{},"current_intent":{}}',
        ]
        idx = 0
        def invoke(self, msgs):
            class R: pass
            r = R()
            r.content = self.responses[self.idx]
            self.idx = (self.idx + 1) % len(self.responses)
            return r
    
    from core.profile_manager import ProfileManager
    
    pm = ProfileManager()
    extractor = ProfileExtractor(MockLLM())
    sid = "mock_user"
    
    queries = [
        "我是女性，要上班穿的衣服",
        "预算500以内",
        "我喜欢黑色简约风",
    ]
    
    for i, q in enumerate(queries, 1):
        profile = pm.get(sid)
        delta = extractor.extract_delta(profile, q)
        new_profile = pm.merge_delta(sid, delta, q)
        print(f"\n--- 轮 {i}: {q} ---")
        print(f"  Δ demographics: {delta['update_demographics']}")
        print(f"  Δ preferences:  {delta['add_preferences']}")
        print(f"  Δ constraints:  {delta['update_constraints']}")
        print(f"  → 累积画像: gender={new_profile['demographics']['gender']}, "
              f"scenes={new_profile['preferences']['scenes']}, "
              f"colors={new_profile['preferences']['colors']}, "
              f"price_max={new_profile['constraints']['price_max']}")
    
    print("\n" + "=" * 60)
    print("✅ Mock 自测完成 - 抽取+合并链路通了")
