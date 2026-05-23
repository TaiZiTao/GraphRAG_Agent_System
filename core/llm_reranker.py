"""LLM 重排器 - L5 精华
基于场景理解 + KG 解释路径的语义重排
"""
import json
import re
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage


PROMPT_TEMPLATE = """你是电商推荐系统的重排器。基于用户的真实意图和商品的图谱属性，对候选商品重新排序。

【用户需求】
原始 Query: {query_text}
识别场景: {scenes}
风格偏好: {style}
颜色偏好: {colors}

【候选商品】（已通过 RRF + BGE 初筛）
{candidates}

【任务】
1. 评估每件商品对用户**真实意图**的匹配度（0-10 分）
   - 不是字面相似度，要结合场景常识
   - 例如"约会显气质"应优先推质感单品，不是运动鞋
2. 给出每件商品的核心卖点（1 句话，≤30 字）
3. 按分数降序排列

【严格输出 JSON 数组，不要任何其他文字】
[
  {{"id": "<商品id>", "score": <0-10整数>, "reason": "<卖点>"}},
  ...
]
"""


class LLMReranker:
    """LLM 重排器：基于场景语义的精排"""
    
    def __init__(self, llm, max_candidates: int = 10):
        self.llm = llm
        self.max_candidates = max_candidates
    
    def rerank(self, dsl, candidates: List[Dict], 
               scenes: Optional[List[str]] = None) -> List[Dict]:
        """
        输入: candidates (List[Dict])，每个含 id/name/price/explain
        输出: 重排后的 candidates，附加 _llm_score / _llm_reason 字段
        失败降级: 返回原候选列表
        """
        if not candidates or len(candidates) < 2:
            return candidates
        
        cands = candidates[:self.max_candidates]
        prompt = self._build_prompt(dsl, cands, scenes or [])
        
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            content = resp.content if hasattr(resp, 'content') else str(resp)
            ranked = self._parse_json(content)
        except Exception as e:
            print(f"⚠️ [LLM重排] 调用失败: {e}")
            return candidates
        
        if not ranked:
            print(f"⚠️ [LLM重排] 解析失败，降级")
            return candidates
        
        # 按 LLM 顺序重组
        cand_map = {c["id"]: c for c in cands}
        reranked = []
        for item in ranked:
            pid = item.get("id")
            if pid in cand_map:
                c = cand_map[pid]
                c["_llm_score"] = item.get("score", 0)
                c["_llm_reason"] = item.get("reason", "")
                reranked.append(c)
        
        # 容错：LLM 漏掉的拼回末尾
        seen = {c["id"] for c in reranked}
        for c in cands:
            if c["id"] not in seen:
                reranked.append(c)
        
        # 后面没参与重排的也接上
        rest = candidates[self.max_candidates:]
        reranked.extend(rest)
        
        print(f"[LLM重排] {len(cands)} 个候选 → 重排成功 "
              f"(Top1: {reranked[0].get('_llm_reason', '')[:30]})")
        return reranked
    
    def _build_prompt(self, dsl, candidates: List[Dict], scenes: List[str]) -> str:
        lines = []
        for i, c in enumerate(candidates, 1):
            explain = c.get("explain") or {}
            matched = explain.get("matched_attrs", [])
            all_attrs = explain.get("all_attrs", [])
            
            line = f"{i}. id={c['id']} | {(c.get('name') or '')[:50]} | ¥{c.get('price', 0)}"
            line += f"\n   品牌={c.get('brand', 'N/A')} 分类={c.get('category', 'N/A')}"
            if matched:
                line += f"\n   命中场景属性: {matched}"
            elif all_attrs:
                line += f"\n   商品属性: {all_attrs[:8]}"
            lines.append(line)
        
        return PROMPT_TEMPLATE.format(
            query_text=getattr(dsl, "query_text", "") or "(无)",
            scenes=scenes if scenes else "(未识别)",
            style=getattr(dsl, "style", []) or "(无)",
            colors=getattr(dsl, "color", []) or "(无)",
            candidates="\n".join(lines),
        )
    
    def _parse_json(self, content: str) -> List[Dict]:
        """容错解析：直接JSON / ```json``` / 裸数组"""
        if not content:
            return []
        
        content = content.strip()
        
        # 1. 直接 parse
        try:
            r = json.loads(content)
            if isinstance(r, list):
                return r
        except Exception:
            pass
        
        # 2. ```json ... ```
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        
        # 3. 裸 [ ... ]
        m = re.search(r"\[\s*\{.*\}\s*\]", content, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        
        return []
