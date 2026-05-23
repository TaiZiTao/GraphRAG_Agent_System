"""意图解析器（不依赖 structured_output）"""
import json
import re
from langchain_core.messages import SystemMessage, HumanMessage
from core.schemas import QueryDSL, IntentType
from core.guardrails import Guardrails


# 价格抱怨模式：用户对当前价格段不满，但没给具体数字
# 走 SEARCH 让下游 enrich_dsl 继承上轮主题词 + 反向推算价格
_PRICE_COMPLAINT_PAT = re.compile(
    r'^\s*('
    r'太便宜了?吧?|便宜了?点的?不要|不要太便宜|价格?太低'
    r'|太贵了?吧?|贵了?点的?不要|不要太贵|价格?太高'
    r')\s*$'
)


PARSE_PROMPT = """你是电商搜索查询解析器。把用户输入解析为 JSON。

输出 JSON 格式（严格）:
{{
  "intent": "search/compare/consult/add_cart/after_sale/chitchat",
  "query_text": "核心商品描述（仅商品本身，不含场景/价格/风格修饰）",
  "brand": "品牌或 null",
  "category": "类目或 null",
  "color": ["颜色列表"],
  "price_min": 数字或 null,
  "price_max": 数字或 null,
  "scene": "场景或 null",
  "style": ["风格列表"],
  "sort_by": "relevance/price_asc/price_desc/sales"
}}

【意图判定优先级】（从上到下匹配，命中即停）
1. add_cart: 含"加购/加入购物车/下单/我要这个/要这个/拍下/买它"
2. compare: 含"对比/哪个好/比较/选哪个" 或 "X 和 Y 比"句式
3. after_sale: 含"退货/退款/物流/客服/到货/发货"
4. consult（需同时满足 a 和 b）：
   a) 含"推荐/搭配/什么值得/送什么/送...礼物/给...买/穿什么/选什么/什么风格"
   b) **且 query 里没有具体商品名**（包/鞋/裙子/T恤/外套/连衣裙/裤子/衬衫等）
   - "通勤包推荐" → search（有商品"包"）
   - "板鞋推荐" → search（有商品"鞋"）
   - "我应该穿什么风格" → consult（无具体商品）
   - "送女朋友什么好" → consult（无具体商品）
5. chitchat: 短问候、纯寒暄、无任何商品/购物语义
   - "我应该穿什么风格"不算 chitchat，是 consult
6. 其余 → search（默认）

【字段规则】
- "便宜的" / "性价比" → sort_by: "price_asc"
- "高端" / "贵的" → sort_by: "price_desc"
- "500块左右" → price_min: 400, price_max: 600
- "500以内" → price_max: 500
- "通勤" / "上班" / "办公" / "职场" → scene: "通勤"
- "约会" / "聚会" / "派对" → scene: "约会"
- "运动" / "健身" / "跑步" → scene: "运动"
- "旅游" / "度假" / "出游" → scene: "旅游"
- query_text 只放商品本身。"通勤包"→query_text="包",scene="通勤"；"约会裙"→query_text="裙子",scene="约会"
- query_text 不允许为 null/空，无具体商品时填短描述（如"对比"/"推荐"/"风格建议"）
- 性别词（女式/男士）属于过滤维度，不要放进 style；style 仅放风格词

【示例】
输入: "300以内的潮牌板鞋"
输出: {{"intent":"search","query_text":"板鞋","price_max":300,"style":["潮牌"]}}

输入: "通勤包推荐"
输出: {{"intent":"search","query_text":"包","scene":"通勤"}}

输入: "夏天的连衣裙"
输出: {{"intent":"search","query_text":"连衣裙","scene":"夏日清爽"}}

输入: "女式通勤包，气质风"
输出: {{"intent":"search","query_text":"包","scene":"通勤","style":["气质"]}}

输入: "我要这个"
输出: {{"intent":"add_cart","query_text":"我要这个"}}

输入: "下单"
输出: {{"intent":"add_cart","query_text":"下单"}}

输入: "这两个哪个好"
输出: {{"intent":"compare","query_text":"对比"}}

输入: "对比一下这两件"
输出: {{"intent":"compare","query_text":"对比"}}

输入: "帮我搭配一下"
输出: {{"intent":"consult","query_text":"搭配建议"}}

输入: "我应该穿什么风格"
输出: {{"intent":"consult","query_text":"风格建议"}}

输入: "你们家什么最值得买"
输出: {{"intent":"consult","query_text":"推荐"}}

输入: "我要给我男朋友买礼物"
输出: {{"intent":"consult","query_text":"礼物"}}

输入: "我要退货"
输出: {{"intent":"after_sale","query_text":"退货"}}

输入: "在吗"
输出: {{"intent":"chitchat","query_text":"在吗"}}

只输出 JSON，不要任何解释。

用户输入: {query}
用户是否上传图片: {has_image}

只输出 JSON："""



class QueryParser:
    def __init__(self, llm):
        self.llm = llm

    def parse(self, user_text: str, has_image: bool) -> QueryDSL:
        # 1. Guardrails 检查
        ok, reason = Guardrails.check(user_text)
        if not ok:
            return QueryDSL(intent=IntentType.REJECT, blocked=True, block_reason=reason)

        # 2. 价格抱怨快捷路径：判 SEARCH，主题词留空让 enrich_dsl 继承上轮
        if _PRICE_COMPLAINT_PAT.match(user_text.strip()):
            print(f"[Parser规则] 价格抱怨表达 → 走 SEARCH 由下游补全")
            return QueryDSL(
                intent=IntentType.SEARCH,
                query_text="",
                use_image=has_image,
            )

        # 3. LLM 抽取
        try:
            prompt = PARSE_PROMPT.format(query=user_text, has_image=has_image)
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            text = resp.content if hasattr(resp, 'content') else str(resp)

            # 从输出中提取 JSON
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                raise ValueError("无 JSON")

            data = json.loads(match.group())

            # ⭐ 关键防御：query_text 是必填 str，不允许 None
            if data.get("query_text") in (None, "null", "None", ""):
                data["query_text"] = user_text  # 退回原文，不丢信息

            # 字段清洗：可空字段
            for k in ["brand", "category", "scene"]:
                if data.get(k) in ("null", "None", "", None):
                    data[k] = None

            # 字段清洗：必为列表
            for k in ["color", "style"]:
                if not isinstance(data.get(k), list):
                    data[k] = []

            dsl = QueryDSL(**data)
            dsl.use_image = has_image
            return dsl

        except Exception as e:
            print(f"[Parser降级] {e}")
            return self._rule_parse(user_text, has_image)

    def _rule_parse(self, text: str, has_image: bool) -> QueryDSL:
        """规则降级（LLM 失败时兜底）"""
        # 先用关键词判意图，再走 SEARCH 默认
        intent = IntentType.SEARCH
        t = text.strip()

        if any(w in t for w in ["加购", "加入购物车", "下单", "我要这个", "要这个", "拍下", "买它"]):
            intent = IntentType.ADD_CART
        elif any(w in t for w in ["对比", "哪个好", "比较", "选哪个"]) or re.search(r'.{1,8}和.{1,8}比', t):
            intent = IntentType.COMPARE
        elif any(w in t for w in ["退货", "退款", "物流", "客服", "到货", "发货"]):
            intent = IntentType.AFTER_SALE
        elif any(w in t for w in ["推荐", "搭配", "什么值得", "送什么", "买礼物", "买什么好"]):
            intent = IntentType.CONSULT
        elif t in ("在吗", "你好", "嗨", "hi", "hello") or len(t) <= 2:
            # 极短无信息量
            if not any(c.isdigit() for c in t):
                intent = IntentType.CHITCHAT

        dsl = QueryDSL(intent=intent, query_text=text, use_image=has_image)

        # 提取价格
        m1 = re.search(r'(\d+)\s*(?:元|块|rmb)?\s*以内', text)
        if m1:
            dsl.price_max = float(m1.group(1))
        m2 = re.search(r'(\d+)\s*(?:元|块|rmb)?\s*左右', text)
        if m2:
            v = float(m2.group(1))
            dsl.price_min = v * 0.8
            dsl.price_max = v * 1.2
        m3 = re.search(r'(\d+)\s*到\s*(\d+)', text)
        if m3:
            dsl.price_min = float(m3.group(1))
            dsl.price_max = float(m3.group(2))

        # 提取风格
        styles = []
        for s in ["潮牌", "复古", "港风", "美式", "英伦", "嘻哈", "休闲", "商务"]:
            if s in text:
                styles.append(s)
        dsl.style = styles

        # 提取场景
        for sc in ["大理旅游", "海边度假", "沙滩", "通勤", "约会", "街头", "运动", "派对", "音乐节", "户外", "日常"]:
            if sc in text:
                dsl.scene = sc
                break

        # 排序
        if any(w in text for w in ["便宜", "性价比", "实惠"]):
            dsl.sort_by = "price_asc"
        elif any(w in text for w in ["高端", "贵", "旗舰"]):
            dsl.sort_by = "price_desc"

        return dsl
