"""ReAct Agent 工具集"""
import json
from typing import List
from core.schemas import QueryDSL, IntentType


class ToolRegistry:
    """工具中心：每轮对话开始前 reset()"""

    def __init__(self, retriever):
        self.retriever = retriever
        self._last_results: List[dict] = []
        self._image = None
        self._session_id = ""
        self._profile_ctx = ""

    def reset(self, session_id: str, image=None, initial_results=None, profile_ctx: str = ""):
        self._session_id = session_id
        self._image = image
        self._last_results = list(initial_results) if initial_results else []
        self._profile_ctx = profile_ctx or ""

    # =================================================================
    def get_schemas(self) -> str:
        return """【可用工具】

1. search_products: 用新条件重新检索（覆盖 last_results）
   参数: query(str必填), scene(str), color(list), price_max(float), price_min(float),
         must_have(list 必含关键词，如 ["女款"]),
         avoid(list 必不含关键词，如 ["男款"])
   例: {"query":"礼物","scene":"送礼","must_have":["男款"],"price_max":500}

2. filter_last_results: 在上次结果上软过滤（不重新检索，更快）
   参数: price_max(float), price_min(float), color(list),
         must_have(list 必含), avoid(list 必不含)
   例: {"price_max":300, "must_have":["女款"]}

3. get_user_profile: 读取用户画像（偏好/历史/预算）
   参数: 无, 例: {}

4. finalize: 选定最终推荐商品（必须最后调用）
   参数: indices(list, 1-based 序号; 不传则采用全部 last_results)
   例: {"indices":[1,2,3]}
"""

    def execute(self, name: str, args: dict) -> str:
        try:
            if name == "search_products":       return self._search(args)
            elif name == "filter_last_results": return self._filter(args)
            elif name == "get_user_profile":    return self._get_profile()
            elif name == "finalize":            return self._finalize(args)
            return f"未知工具: {name}"
        except Exception as e:
            return f"工具执行失败: {e}"

    # =================================================================
    def _search(self, args):
        if not args.get("query"):
            return "search_products 需要 query 参数"

        dsl = QueryDSL(
            intent=IntentType.SEARCH,
            query_text=args.get("query", ""),
            scene=args.get("scene") or None,
            color=args.get("color", []) or [],
            price_max=args.get("price_max"),
            price_min=args.get("price_min"),
        )
        # 多召回一些，给 must_have / avoid 内存过滤留余量
        results = self.retriever.retrieve(dsl, image=self._image, top_k=10)

        # 🆕 内存层硬过滤（schema 不动，tools 自己消化）
        must_have = args.get("must_have") or []
        avoid = args.get("avoid") or []

        if must_have or avoid:
            filtered = []
            for p in results:
                text = self._blob(p)
                if must_have and not all(kw in text for kw in must_have):
                    continue
                if avoid and any(kw in text for kw in avoid):
                    continue
                filtered.append(p)

            if filtered:
                results = filtered[:5]
            else:
                # 兜底：过滤后空了，返回未过滤 top5 + 提示
                results = results[:5]
                self._last_results = results
                return (f"⚠️ must_have={must_have} avoid={avoid} 过滤后无结果，"
                        f"返回未过滤的 top5（建议换 query 或放宽条件）：\n" + self._fmt(results))
        else:
            results = results[:5]

        self._last_results = results
        return self._fmt(results)

    # =================================================================
    def _filter(self, args) -> str:
        if not self._last_results:
            return "上轮没有搜索结果，请先调用 search_products"

        out = []
        for p in self._last_results:
            # 价格
            try:
                price = float(p.get("price", 0) or 0)
            except (TypeError, ValueError):
                price = 0.0
            if args.get("price_max") is not None and price > args["price_max"]:
                continue
            if args.get("price_min") is not None and price < args["price_min"]:
                continue

            text = self._blob(p)

            # color：任一命中
            colors = args.get("color") or []
            if colors and not any(c in text for c in colors):
                continue

            # must_have：全部命中
            must_have = args.get("must_have") or []
            if must_have and not all(kw in text for kw in must_have):
                continue

            # avoid：不能命中
            avoid = args.get("avoid") or []
            if avoid and any(kw in text for kw in avoid):
                continue

            out.append(p)

        if not out:
            return f"过滤后无结果（上轮共 {len(self._last_results)} 个候选都不符合）"
        self._last_results = out
        return self._fmt(out)

    # =================================================================
    def _get_profile(self):
        return self._profile_ctx if self._profile_ctx else "（用户画像为空）"

    # =================================================================
    def _finalize(self, args):
        idxs = args.get("indices", [])
        if not idxs:
            return f"已采用当前全部 {len(self._last_results)} 个商品"
        try:
            picked = [self._last_results[i-1] for i in idxs if 1 <= i <= len(self._last_results)]
        except Exception:
            return f"序号无效: {idxs}"
        if not picked:
            return f"序号 {idxs} 越界（last_results 共 {len(self._last_results)} 个）"
        self._last_results = picked
        return f"已选定 {len(picked)} 个商品作为最终推荐"

    # =================================================================
    @staticmethod
    def _blob(p: dict) -> str:
        """拼接商品全部文本字段用于关键词匹配，防 None"""
        parts = [
            p.get("name") or "",
            p.get("title") or "",
            p.get("description") or "",
            p.get("dense_caption") or "",
            p.get("category") or "",
            p.get("brand") or "",
        ]
        attrs = p.get("attrs")
        if isinstance(attrs, (list, tuple)):
            parts.append(" ".join(str(a) for a in attrs if a))
        elif attrs:
            parts.append(str(attrs))
        return " ".join(parts)

    # =================================================================
    @staticmethod
    def _fmt(results):
        if not results:
            return "无结果"
        lines = [f"找到 {len(results)} 个商品："]
        for i, p in enumerate(results, 1):
            title = (p.get("name")
                     or p.get("title")
                     or (p.get("description") or "")[:30]
                     or "(无标题)")
            pid = p.get("id", "?")
            price = p.get("price", "?")
            category = p.get("category") or "-"
            lines.append(f"[{i}] id={pid} | {title} | ¥{price} | 品类:{category}")
        return "\n".join(lines)

    # =================================================================
    @property
    def last_results(self):
        return self._last_results
