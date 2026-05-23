"""分层记忆系统"""
import json
import sqlite3
from datetime import datetime
from typing import List
from config import Config
from core.schemas import QueryDSL

# Redis 优雅降级
REDIS_AVAILABLE = False
_redis_client = None
try:
    import redis
    _redis_client = redis.Redis(host=Config.REDIS_HOST, port=Config.REDIS_PORT,
                                 decode_responses=True, socket_connect_timeout=2)
    _redis_client.ping()
    REDIS_AVAILABLE = True
    print("✅ Redis 连接成功")
except Exception:
    print("⚠️ Redis 不可用，使用内存降级")
    _memory_fallback = {}


# 不该进 style 的词（性别 / 占位 / 通用名词）
_NON_STYLE_WORDS = {
    "女", "男", "女生", "男生", "女式", "男式", "女款", "男款", "中性",
    "商品", "物品", "东西", "产品", "宝贝", "那个", "这个", "它", "它们",
}


class TieredMemory:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.buffer_size = Config.MEMORY_BUFFER_SIZE
        self._init_sqlite()
    
    def _init_sqlite(self):
        self.db = sqlite3.connect(Config.SQLITE_PATH, check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS user_profile(
            user_id TEXT PRIMARY KEY,
            preferences TEXT, budget TEXT, summary TEXT, updated_at TEXT
        )""")
        self.db.commit()
    
    # ------------------ 对话缓冲（Redis / 内存） ------------------
    def add_message(self, role: str, content: str):
        key = f"chat:{self.user_id}"
        msg = json.dumps({"role": role, "content": content[:500],
                          "ts": datetime.now().isoformat()})
        if REDIS_AVAILABLE:
            _redis_client.rpush(key, msg)
            _redis_client.expire(key, 86400)
        else:
            global _memory_fallback
            _memory_fallback.setdefault(key, []).append(msg)
            _memory_fallback[key] = _memory_fallback[key][-50:]
    
    def get_buffer(self) -> List[dict]:
        key = f"chat:{self.user_id}"
        if REDIS_AVAILABLE:
            raw = _redis_client.lrange(key, -self.buffer_size, -1)
        else:
            raw = _memory_fallback.get(key, [])[-self.buffer_size:]
        return [json.loads(r) for r in raw]
    
    # ------------------ 长期画像（SQLite） ------------------
    def update_preference(self, dsl: QueryDSL):
        row = self.db.execute(
            "SELECT preferences, budget, summary FROM user_profile WHERE user_id=?",
            (self.user_id,)
        ).fetchone()
        
        prefs = json.loads(row[0]) if row and row[0] else {
            "brand_counter": {}, "style": [], "color_avoid": []
        }
        budget = json.loads(row[1]) if row and row[1] else {}
        summary = row[2] if row else ""
        
        # 品牌计数
        if dsl.brand:
            prefs["brand_counter"][dsl.brand] = prefs["brand_counter"].get(dsl.brand, 0) + 1
        
        # 风格：过滤掉性别/占位词，保持插入顺序去重
        if dsl.style:
            clean = [s for s in dsl.style if s and s not in _NON_STYLE_WORDS]
            if clean:
                prefs["style"] = list(dict.fromkeys(prefs["style"] + clean))[-10:]
        
        # 预算：意图切换检测
        # - 只给 price_min  → 用户在说"至少多少"，清掉旧 max
        # - 只给 price_max  → 用户在说"不超过多少"，清掉旧 min
        # - 都给            → 完全用本轮
        # - 都没给          → 不动
        user_min = dsl.price_min
        user_max = dsl.price_max
        if user_min is not None and user_max is None:
            budget = {"min": user_min}
        elif user_max is not None and user_min is None:
            budget = {"max": user_max}
        elif user_min is not None and user_max is not None:
            budget = {"min": user_min, "max": user_max}
        # else: 保持不变
        
        # 区间合理性兜底
        if budget.get("min") is not None and budget.get("max") is not None:
            if float(budget["min"]) > float(budget["max"]):
                budget.pop("min", None)
        
        self.db.execute("""INSERT OR REPLACE INTO user_profile VALUES (?,?,?,?,?)""",
            (self.user_id, json.dumps(prefs, ensure_ascii=False),
             json.dumps(budget), summary, datetime.now().isoformat()))
        self.db.commit()
    
    def render_profile_context(self) -> str:
        """
        渲染长期画像（跨 session）
        新的多轮画像（ProfileManager）已经覆盖本会话状态，
        这里只输出长期累积的差异化信息（品牌偏好等）。
        """
        row = self.db.execute(
            "SELECT preferences, budget FROM user_profile WHERE user_id=?",
            (self.user_id,)
        ).fetchone()
        if not row:
            return ""
        
        prefs = json.loads(row[0] or "{}")
        budget = json.loads(row[1] or "{}")
        
        top_brand = max(prefs.get("brand_counter", {}).items(),
                       key=lambda x: x[1], default=(None, 0))[0]
        
        parts = ["[长期画像]"]
        if top_brand:
            parts.append(f"- 偏好品牌: {top_brand}")
        if prefs.get("style"):
            parts.append(f"- 历史风格: {', '.join(prefs['style'][-5:])}")
        if budget:
            bmin = budget.get("min")
            bmax = budget.get("max")
            if bmin or bmax:
                parts.append(
                    f"- 历史预算: ¥{int(bmin) if bmin else '-'} ~ ¥{int(bmax) if bmax else '-'}"
                )
        
        return "\n".join(parts) if len(parts) > 1 else ""
    
    # ------------------ 维护 ------------------
    def reset_profile(self):
        """清空当前用户的长期画像（调试 / 数据迁移用）"""
        self.db.execute("DELETE FROM user_profile WHERE user_id=?", (self.user_id,))
        self.db.commit()
