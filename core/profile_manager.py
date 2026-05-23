"""
用户画像管理器
- 结构化画像（人口属性 + 偏好 + 约束 + 当前意图）
- 增量合并（追加 / 覆盖 / 重置 不同字段不同策略）
"""
from typing import Dict, Optional
from copy import deepcopy

from core.profile_storage import ProfileStorage, get_storage


# 显式清空标记
# - LLM 输出该值表示用户主动放弃该字段
# - 写入端用该值在单边输入（如"超过300"）时清掉对侧（price_max）
SENTINEL_CLEAR = "__clear__"


class ProfileManager:
    """单例式画像管理器"""
    
    def __init__(self, storage: Optional[ProfileStorage] = None):
        self.storage = storage or get_storage()
    
    @staticmethod
    def new_profile() -> Dict:
        """新会话的空画像模板"""
        return {
            "demographics": {
                "gender": None,        # "男" / "女" / None
                "age_range": None,     # "20-30" 等
            },
            "preferences": {
                "styles": [],          # 风格累积
                "colors": [],          # 颜色累积
                "scenes": [],          # 场景历史
            },
            "constraints": {
                "price_max": None,
                "price_min": None,
                "must_have": [],
                "avoid": [],
            },
            "current_intent": {
                "scene": None,
                "category": None,
            },
            "_meta": {
                "turn": 0,
                "last_query": "",
            },
        }
    
    def get(self, session_id: str) -> Dict:
        """取画像，不存在则创建"""
        profile = self.storage.get(session_id)
        if profile is None:
            profile = self.new_profile()
            self.storage.set(session_id, profile)
        return profile
    
    def reset(self, session_id: str):
        """重置画像（用户主动清除）"""
        self.storage.set(session_id, self.new_profile())
    
    def merge_delta(self, session_id: str, delta: Dict, query: str = "") -> Dict:
        """
        合并 LLM 增量更新到画像
        delta 结构:
            {
                "update_demographics": {...},   # 覆盖（None 跳过）
                "add_preferences": {...},       # 追加去重
                "update_constraints": {...},    # 覆盖；__clear__ 清空；None 跳过
                "current_intent": {...},        # 按字段覆盖；__clear__ 清空
            }
        """
        profile = self.get(session_id)
        profile = deepcopy(profile)  # 防止引用污染
        
        # 1. demographics: 覆盖（None 跳过）
        for k, v in delta.get("update_demographics", {}).items():
            if v is not None:
                profile["demographics"][k] = v
        
        # 2. preferences: 追加去重，最多保留 5 个
        for key, vals in delta.get("add_preferences", {}).items():
            if not isinstance(vals, list):
                continue
            current = profile["preferences"].get(key, [])
            merged = list(dict.fromkeys(current + [v for v in vals if v]))
            profile["preferences"][key] = merged[-5:]
        
        # 3. constraints: 覆盖
        #    - __clear__ → 主动清空字段（标量=None，列表=[]）
        #    - None → 跳过（沿用历史）
        #    - 其他 → 覆盖
        for k, v in delta.get("update_constraints", {}).items():
            if v == SENTINEL_CLEAR:
                if k in ("must_have", "avoid"):
                    profile["constraints"][k] = []
                else:
                    profile["constraints"][k] = None
            elif v is not None:
                profile["constraints"][k] = v
        
        # 4. current_intent: 按字段覆盖
        #    - 显式 __clear__ → 清空为 None
        #    - None / "" → 沿用上一轮（视为本轮未提及）
        #    - 其他真值 → 覆盖
        new_intent = delta.get("current_intent", {})
        for k, v in new_intent.items():
            if v == SENTINEL_CLEAR:
                profile["current_intent"][k] = None
            elif v not in (None, ""):
                profile["current_intent"][k] = v
            # 其余情况：保留上一轮
        
        # 5. meta
        profile["_meta"]["turn"] += 1
        profile["_meta"]["last_query"] = query
        
        # 写回
        self.storage.set(session_id, profile)
        return profile
    
    def to_dsl_hints(self, profile: Dict) -> Dict:
        """
        把画像转成可注入 DSL 的提示
        策略：当前 query 没说的，用画像补；说了的不覆盖
        """
        prefs = profile["preferences"]
        return {
            "hint_styles": prefs["styles"][-3:],     # 最近 3 个
            "hint_colors": prefs["colors"][-2:],     # 最近 2 个
            "hint_scenes": prefs["scenes"][-2:],
            "filters": {
                "gender": profile["demographics"]["gender"],
                "price_max": profile["constraints"]["price_max"],
                "price_min": profile["constraints"]["price_min"],
                "avoid": profile["constraints"]["avoid"],
            },
        }


# ============================================================================
# 自测
# ============================================================================
if __name__ == "__main__":
    print("🧪 ProfileManager 自测")
    print("=" * 60)
    
    pm = ProfileManager()
    sid = "test_user"
    pm.reset(sid)
    
    deltas = [
        {
            "update_demographics": {"gender": "女"},
            "add_preferences": {"scenes": ["通勤"]},
            "update_constraints": {},
            "current_intent": {"scene": "通勤", "category": "包"},
        },
        {
            # 单边：用户说"不超过200"，要清掉历史 price_min（这里历史本身没有，无影响）
            "update_demographics": {},
            "add_preferences": {},
            "update_constraints": {"price_max": 200, "price_min": SENTINEL_CLEAR},
            "current_intent": {},
        },
        {
            # 单边：用户说"超过300"，要清掉历史 price_max=200
            "update_demographics": {},
            "add_preferences": {},
            "update_constraints": {"price_min": 300, "price_max": SENTINEL_CLEAR},
            "current_intent": {},
        },
        {
            # 切场景，没提价格 → 价格沿用（仅 price_min=300）
            "update_demographics": {},
            "add_preferences": {"scenes": ["约会"]},
            "update_constraints": {},
            "current_intent": {"scene": "约会"},
        },
        {
            # 显式清空 category
            "update_demographics": {},
            "add_preferences": {},
            "update_constraints": {},
            "current_intent": {"category": SENTINEL_CLEAR},
        },
    ]
    
    queries = [
        "我是女性，找通勤包",
        "不超过200元",
        "看看超过300元的",
        "我要约会",
        "不要包了",
    ]
    
    for i, (q, d) in enumerate(zip(queries, deltas), 1):
        print(f"\n--- 轮次 {i}: {q} ---")
        profile = pm.merge_delta(sid, d, query=q)
        print(f"  gender={profile['demographics']['gender']}")
        print(f"  scenes={profile['preferences']['scenes']}")
        print(f"  price_min={profile['constraints']['price_min']} / price_max={profile['constraints']['price_max']}")
        print(f"  current={profile['current_intent']}")
    
    final = pm.get(sid)
    pmin = final["constraints"]["price_min"]
    pmax = final["constraints"]["price_max"]
    
    # 关键断言
    assert pmin == 300, f"price_min 应为 300，实际 {pmin}"
    assert pmax is None, f"price_max 应被清空，实际 {pmax}"
    assert final["current_intent"]["scene"] == "约会", "scene 应保留"
    assert final["current_intent"]["category"] is None, "category 应被清空"
    
    print("\n" + "=" * 60)
    print("✅ 自测完成：单边价格切换不再产生冲突 + category 显式清空正常")
