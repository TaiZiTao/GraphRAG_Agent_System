"""
画像集成器 - L4 双阶段更新
─────────────────────────────────────
- 同步阶段: 从 DSL 已抽字段秒级合并到画像 → 当前轮立即生效
- 异步阶段: 后台线程跑 LLM 抽取深层信息 → 下一轮生效
- 增强阶段: 画像反哺 DSL 的空字段（仅跨轮意图，不反哺软偏好）
"""
import re
import threading
from core.profile_manager import ProfileManager, SENTINEL_CLEAR
from agent.profile_extractor import ProfileExtractor


# 纯过滤/排序指令模式：本轮只是改预算/小修小补，没有新主题
_FILTER_ONLY_PAT = re.compile(
    r'^\s*('
    r'[\d¥￥\.\s元块钱]+(以内|之内|左右|上下|以下|以上)?'
    r'|不要?超过\s*[\d\.]+\s*(元|块|块钱)?'
    r'|预算\s*[\d\.]+\s*(元|块|块钱)?'
    r'|再?便宜点?的?|贵点?的?|高端点?的?'
    r'|换个?颜色|其他的?|换一个?|再来几个?'
    r')\s*$'
)


# LLM 占位/泛指词：parser 没识别清楚时常吐这种值
_PLACEHOLDER_TOKENS = {
    "商品", "物品", "东西", "产品", "宝贝", "那个", "这个", "它", "??", "?", "无", "空"
}


# 价格抱怨模式（与 query_parser 对齐）
_PRICE_UP_PAT = re.compile(r'太便宜|便宜了?点的?不要|不要太便宜|价格?太低')
_PRICE_DOWN_PAT = re.compile(r'太贵|贵了?点的?不要|不要太贵|价格?太高')


class ProfileIntegrator:
    """连接 DSL × ProfileManager × ProfileExtractor"""
    
    # 类级缓存：跨实例共享
    _last_qtext: dict = {}      # {session_id: 上一轮清洗后的主题词}
    _pending_logs: dict = {}    # {session_id: [异步线程产生的日志行]}
    
    def __init__(self, llm):
        self.pm = ProfileManager()
        self.extractor = ProfileExtractor(llm)
    
    # =================================================================
    # 阶段 1：同步快速合并（毫秒级，不调 LLM）
    # =================================================================
    def sync_merge_from_dsl(self, session_id: str, dsl, query: str):
        """从 DSL 已抽字段构造 delta 合并到画像"""
        if dsl is None:
            return
        
        # 单边价格输入要清对侧（写入端根治冲突）
        # 用户说"不超过200" → 只有 price_max → 清掉历史 price_min
        # 用户说"超过300"   → 只有 price_min → 清掉历史 price_max
        cons_update = {
            "price_max": dsl.price_max,
            "price_min": dsl.price_min,
        }
        if dsl.price_max is not None and dsl.price_min is None:
            cons_update["price_min"] = SENTINEL_CLEAR
        elif dsl.price_min is not None and dsl.price_max is None:
            cons_update["price_max"] = SENTINEL_CLEAR
        
        delta = {
            "update_demographics": {},
            "add_preferences": {
                "styles": list(dsl.style) if dsl.style else [],
                "colors": list(dsl.color) if dsl.color else [],
                "scenes": [dsl.scene] if dsl.scene else [],
            },
            "update_constraints": cons_update,
            "current_intent": {
                "scene": dsl.scene,
                "category": dsl.category,
            },
        }
        self.pm.merge_delta(session_id, delta, query=query)
        
        # 记录本轮主题词，供下轮"纯过滤指令"继承
        if dsl.query_text and dsl.query_text.strip():
            cleaned = dsl.query_text.strip()
            if cleaned not in _PLACEHOLDER_TOKENS:
                ProfileIntegrator._last_qtext[session_id] = cleaned
    
    # =================================================================
    # 阶段 2：异步 LLM 补充（不阻塞主流程）
    # =================================================================
    def async_llm_enrich(self, session_id: str, query: str):
        """后台线程跑 LLM，补 DSL 没抽到的深层字段"""
        def task():
            try:
                profile = self.pm.get(session_id)
                delta = self.extractor.extract_delta(profile, query)
                
                # 只保留 DSL 不会抽的部分，避免重复合并
                filtered = {
                    "update_demographics": delta.get("update_demographics", {}),
                    "add_preferences": {},
                    "update_constraints": {},
                    "current_intent": {},
                }
                src_cons = delta.get("update_constraints", {})
                for k in ("must_have", "avoid"):
                    if k in src_cons and src_cons[k] is not None:
                        filtered["update_constraints"][k] = src_cons[k]
                
                if any(filtered[k] for k in filtered):
                    self.pm.merge_delta(session_id, filtered, query=query)
                    ProfileIntegrator._pending_logs.setdefault(session_id, []).append(
                        f"[👤 异步画像] LLM 补充: demo={filtered['update_demographics']} cons={filtered['update_constraints']}"
                    )
            except Exception as e:
                ProfileIntegrator._pending_logs.setdefault(session_id, []).append(
                    f"[ProfileIntegrator异步失败] {e}"
                )
        
        threading.Thread(target=task, daemon=True).start()
    
    def flush_pending_logs(self, session_id: str):
        """下一轮开头打印上一轮异步线程攒下的日志（避免与 input() 冲突）"""
        logs = ProfileIntegrator._pending_logs.pop(session_id, None)
        if logs:
            for line in logs:
                print(line)
    
    # =================================================================
    # 阶段 3：用画像增强 DSL 的空字段
    # =================================================================
    def enrich_dsl(self, dsl, session_id: str, query: str = ""):
        """
        画像反哺 DSL：仅在当前轮没明说时用历史填充
        必须在 sync_merge_from_dsl 之前调用
        """
        if dsl is None:
            return dsl
        
        profile = self.pm.get(session_id)
        cons = profile.get("constraints", {})
        intent = profile.get("current_intent", {})
        
        # ---- 主题词继承 ----
        # 三种情况都要继承上轮主题：
        # 1. 本轮 query_text 为空（纯指令如"超过300元"、"太便宜了吧"）
        # 2. 本轮 query_text 是占位词（"商品"、"东西"）
        # 3. 本轮 query_text 是纯过滤词（"200元以内"）
        qtext = (dsl.query_text or "").strip()
        last_qtext = ProfileIntegrator._last_qtext.get(session_id, "").strip()
        
        if last_qtext:
            should_inherit = False
            reason = ""
            if not qtext:
                should_inherit = True
                reason = "本轮无主题词"
            elif qtext in _PLACEHOLDER_TOKENS:
                should_inherit = True
                reason = f"'{qtext}' 是占位词"
            elif qtext != last_qtext and _FILTER_ONLY_PAT.match(qtext):
                should_inherit = True
                reason = f"'{qtext}' 是纯过滤词"
            
            if should_inherit:
                dsl.query_text = last_qtext
                print(f"[👤 增强] query_text ← '{last_qtext}' ({reason})")
        
        # ---- 价格区间反哺：本轮没给才继承历史 ----
        if dsl.price_max is None and dsl.price_min is None:
            hist_min = cons.get("price_min")
            hist_max = cons.get("price_max")
            
            # 兜底：万一历史出现冲突，全部丢弃
            if hist_min and hist_max and float(hist_min) > float(hist_max):
                print(f"[⚠️ 历史价格冲突] ≥¥{int(hist_min)} / ≤¥{int(hist_max)}，跳过继承")
            else:
                if hist_max:
                    dsl.price_max = hist_max
                    print(f"[👤 增强] price_max ← {dsl.price_max} (历史预算)")
                if hist_min:
                    dsl.price_min = hist_min
                    print(f"[👤 增强] price_min ← {dsl.price_min} (历史预算)")
        elif dsl.price_min is not None and dsl.price_max is None:
            print(f"[👤 增强] 本轮 price_min={dsl.price_min}，跳过历史 price_max 继承")
        elif dsl.price_max is not None and dsl.price_min is None:
            print(f"[👤 增强] 本轮 price_max={dsl.price_max}，跳过历史 price_min 继承")
        
        # ---- 价格抱怨调整：基于继承后的价格反向推算 ----
        q = query or ""
        if _PRICE_UP_PAT.search(q):
            # 嫌便宜 → 抬高下限，清掉上限
            if dsl.price_min:
                new_min = float(dsl.price_min) * 1.5
            elif dsl.price_max:
                new_min = float(dsl.price_max)
            else:
                new_min = 300.0
            dsl.price_min = new_min
            dsl.price_max = None
            print(f"[👤 增强] 价格抱怨'太便宜' → price_min=¥{int(new_min)}, 清除 price_max")
        elif _PRICE_DOWN_PAT.search(q):
            # 嫌贵 → 压低上限，清掉下限
            if dsl.price_max:
                new_max = float(dsl.price_max) * 0.7
            elif dsl.price_min:
                new_max = float(dsl.price_min)
            else:
                new_max = 200.0
            dsl.price_max = new_max
            dsl.price_min = None
            print(f"[👤 增强] 价格抱怨'太贵' → price_max=¥{int(new_max)}, 清除 price_min")
        
        # ---- 区间合理性自检（最后一道防线）----
        if dsl.price_min is not None and dsl.price_max is not None:
            if float(dsl.price_min) > float(dsl.price_max):
                print(f"[⚠️ 价格冲突] price_min={dsl.price_min} > price_max={dsl.price_max}，丢弃 price_min")
                dsl.price_min = None
        
        # ---- 跨轮意图：scene / category ----
        if not dsl.scene and intent.get("scene"):
            dsl.scene = intent["scene"]
            print(f"[👤 增强] scene ← {dsl.scene} (当前意图)")
        
        if not dsl.category and intent.get("category"):
            dsl.category = intent["category"]
            print(f"[👤 增强] category ← {dsl.category} (当前意图)")
        
        return dsl
    
    # =================================================================
    # 渲染给 prompt 的画像文本
    # =================================================================
    def render_context(self, session_id: str) -> str:
        profile = self.pm.get(session_id)
        if not profile or profile.get("_meta", {}).get("turn", 0) == 0:
            return ""
        
        demo = profile.get("demographics", {})
        prefs = profile.get("preferences", {})
        cons = profile.get("constraints", {})
        intent = profile.get("current_intent", {})
        
        parts = ["[多轮画像]"]
        
        demo_bits = []
        if demo.get("gender"):
            demo_bits.append(f"性别{demo['gender']}")
        if demo.get("age_range"):
            demo_bits.append(f"年龄{demo['age_range']}")
        if demo_bits:
            parts.append(f"- 用户: {', '.join(demo_bits)}")
        
        if prefs.get("styles"):
            parts.append(f"- 偏好风格: {', '.join(prefs['styles'][-3:])}")
        if prefs.get("colors"):
            parts.append(f"- 偏好颜色: {', '.join(prefs['colors'][-2:])}")
        if prefs.get("scenes"):
            parts.append(f"- 历史场景: {' → '.join(prefs['scenes'][-3:])}")
        
        cons_bits = []
        if cons.get("price_min"):
            cons_bits.append(f"≥¥{int(cons['price_min'])}")
        if cons.get("price_max"):
            cons_bits.append(f"≤¥{int(cons['price_max'])}")
        if cons.get("avoid"):
            cons_bits.append(f"避免{cons['avoid'][:3]}")
        if cons_bits:
            parts.append(f"- 约束: {' / '.join(cons_bits)}")
        
        if intent.get("scene"):
            parts.append(f"- 本轮场景: {intent['scene']}")
        
        return "\n".join(parts) if len(parts) > 1 else ""


# =============================================================================
# 自测
# =============================================================================
if __name__ == "__main__":
    from core.schemas import QueryDSL, IntentType
    
    print("🧪 ProfileIntegrator 自测")
    print("=" * 60)
    
    class MockLLM:
        def invoke(self, msgs):
            class R:
                pass
            r = R()
            r.content = '{"update_demographics":{},"add_preferences":{},"update_constraints":{},"current_intent":{}}'
            return r
    
    integrator = ProfileIntegrator(MockLLM())
    sid = "test_integ"
    integrator.pm.reset(sid)
    
    cases = [
        ("女式通勤包",     QueryDSL(intent=IntentType.SEARCH, query_text="包", scene="通勤")),
        ("不超过200元",    QueryDSL(intent=IntentType.SEARCH, query_text="", price_max=200.0)),
        ("超过300元",      QueryDSL(intent=IntentType.SEARCH, query_text="", price_min=300.0)),
        ("我要约会",       QueryDSL(intent=IntentType.SEARCH, query_text="衣服", scene="约会")),
    ]
    
    for i, (q, dsl) in enumerate(cases, 1):
        print(f"\n--- 轮 {i}: {q} ---")
        print(f"  原始 DSL: query_text='{dsl.query_text}', price_min={dsl.price_min}, price_max={dsl.price_max}")
        integrator.enrich_dsl(dsl, sid, query=q)
        print(f"  增强 DSL: query_text='{dsl.query_text}', price_min={dsl.price_min}, price_max={dsl.price_max}")
        integrator.sync_merge_from_dsl(sid, dsl, q)
        print(f"  画像: {integrator.render_context(sid)}")
    
    print("\n" + "=" * 60)
    print("✅ 自测完成")
