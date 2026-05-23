"""三路混合检索 + KG 召回 + RRF + Cross-Encoder Rerank"""
import jieba
import numpy as np
from typing import List, Optional
from core.encoders import encode_text, encode_image
from core.schemas import QueryDSL
from config import Config


# 这些词是场景/风格/排序词，不是品类，不参与品类硬过滤
_NON_CATEGORY_WORDS = {
    # 场景
    "通勤", "约会", "派对", "聚会", "运动", "健身", "跑步",
    "旅游", "度假", "海边", "居家", "出游", "上班", "办公",
    # 风格
    "气质", "简约", "潮牌", "复古", "甜美", "休闲", "百搭",
    "文艺", "可爱", "性感", "高端", "轻奢", "OL",
    # 排序/修饰
    "便宜", "性价比", "推荐", "好看", "时尚",
    # 性别（应在 demographics）
    "女", "男", "女生", "男生", "女式", "男式", "女款", "男款",
}


class HybridRetriever:
    def __init__(self, products, collection, scene_kg):
        self.products = {p["id"]: p for p in products}
        self.collection = collection
        self.scene_kg = scene_kg
        
        # BM25 索引
        self.corpus_ids = list(self.products.keys())
        self.corpus_tokens = [
            list(jieba.cut(self.products[pid].get("dense_caption") or 
                           self.products[pid].get("description","")))
            for pid in self.corpus_ids
        ]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        
        
        self.llm_reranker = None

    
    def retrieve(self, dsl: QueryDSL, image=None, top_k=5) -> List[dict]:
        # ============ KG 场景识别 ============
        matched_scenes = []
        if dsl.scene and hasattr(self.scene_kg, 'match_scenes_by_query'):
            matched_scenes = self.scene_kg.match_scenes_by_query(dsl.scene)
            if not matched_scenes:
                matched_scenes = self.scene_kg.match_scenes_by_query(dsl.query_text)
        elif dsl.scene:
            matched_scenes = [dsl.scene]
        elif hasattr(self.scene_kg, 'match_scenes_by_query'):
            matched_scenes = self.scene_kg.match_scenes_by_query(dsl.query_text)
        
        # ============ KG 扩展属性 ============
        expanded = []
        for scene in matched_scenes:
            expanded.extend(self.scene_kg.expand_scene(scene))
        
        if dsl.brand and hasattr(self.scene_kg, 'expand_brand'):
            brand_attrs = self.scene_kg.expand_brand(dsl.brand)
            expanded.extend(brand_attrs)
            if brand_attrs:
                print(f"[KG] 品牌 '{dsl.brand}' → {brand_attrs}")
        
        expanded = list(set(expanded))
        if matched_scenes:
            print(f"[KG] 场景 {matched_scenes} → 属性 {expanded[:10]}")
        
        enhanced_query = dsl.query_text
        if expanded:
            enhanced_query = f"{enhanced_query} {' '.join(expanded)}".strip()
        
        scores = {}
        
        # === 1. 文本向量召回 ===
        if enhanced_query:
            try:
                qv = encode_text(enhanced_query)
                where = dsl.to_chroma_where()
                r = self.collection.query(
                    query_embeddings=[qv], n_results=Config.RECALL_TOP_K, where=where
                )
                for i, meta in enumerate(r["metadatas"][0]):
                    pid = meta["id"]
                    scores.setdefault(pid, {})["vec"] = 1 - r["distances"][0][i]
            except Exception as e:
                print(f"[文本召回失败] {e}")
        
        # === 2. 图像向量召回 ===
        if image is not None:
            try:
                qv = encode_image(image)
                where = dsl.to_chroma_where_image()
                r = self.collection.query(
                    query_embeddings=[qv], n_results=Config.RECALL_TOP_K, where=where
                )
                for i, meta in enumerate(r["metadatas"][0]):
                    pid = meta["id"]
                    scores.setdefault(pid, {})["img"] = 1 - r["distances"][0][i]
            except Exception as e:
                print(f"[图像召回失败] {e}")
        
        # === 3. BM25 关键词召回 ===
        if enhanced_query and self.bm25:
            tokens = list(jieba.cut(enhanced_query))
            bm = self.bm25.get_scores(tokens)
            if bm.max() > 0: bm = bm / bm.max()
            top_idx = np.argsort(-bm)[:Config.RECALL_TOP_K]
            for idx in top_idx:
                pid = self.corpus_ids[idx]
                if not self._match_filter(pid, dsl): continue
                if bm[idx] > 0.1:
                    scores.setdefault(pid, {})["bm25"] = float(bm[idx])
        
        # === 4. KG 图谱召回（新增第四路）===
        if matched_scenes and hasattr(self.scene_kg, 'find_products_by_scenes'):
            try:
                kg_hits = self.scene_kg.find_products_by_scenes(
                    matched_scenes, limit=Config.RECALL_TOP_K
                )
                max_mc = max((h.get('match_count', 1) for h in kg_hits), default=1)
                for h in kg_hits:
                    pid = h['id']
                    if pid not in self.products: continue
                    if not self._match_filter(pid, dsl): continue
                    scores.setdefault(pid, {})["kg"] = h.get('match_count', 1) / max_mc
                print(f"[KG召回] 命中 {len(kg_hits)} 个商品")
            except Exception as e:
                print(f"[KG召回失败] {e}")
        
        if not scores: return []
        
        # === RRF 融合（含 kg 第四路）===
        fused = self._rrf_fusion(scores)
        
        # === 品类关键词硬过滤（关键修复）===
        # KG 场景召回会把同场景下其它品类也带回来（如"旅游"召回草帽/凉鞋/裙子）
        # 当 query_text 是短品类词（如"包"/"鞋"/"裙"），要求商品文本中真的包含该词
        cat_kw = self._extract_category_keyword(dsl)
        if cat_kw:
            before = len(fused)
            filtered = [(pid, s) for pid, s in fused if self._product_contains(pid, cat_kw)]
            if filtered:  # 有结果才生效，全空就降级保留
                fused = filtered
                print(f"[品类过滤] '{cat_kw}': {before} → {len(fused)} 候选")
            else:
                print(f"[品类过滤] '{cat_kw}' 无匹配，降级保留原候选")
        
        candidates = fused[:Config.RERANK_INPUT_K]
        print(f"[召回] 融合后 {len(candidates)} 候选")
        
        
        # === 取前 top_k*3 候选用于精排 ===
        results = [dict(self.products[pid]) for pid, _ in candidates[:top_k*3]]
        
        # === 注入 explain（L3 解释路径，给 LLM 重排器看）===
        if matched_scenes and hasattr(self.scene_kg, 'get_product_explain'):
            for r in results:
                try:
                    r["explain"] = self.scene_kg.get_product_explain(
                        r["id"], matched_scenes
                    )
                except Exception:
                    r["explain"] = {"scenes": [], "matched_attrs": [], "all_attrs": []}
        else:
            for r in results:
                r["explain"] = {"scenes": [], "matched_attrs": [], "all_attrs": []}
        
        # === L5: LLM 语义重排（带 KG 解释路径）===
        if self.llm_reranker is not None and len(results) >= 3:
            try:
                results = self.llm_reranker.rerank(dsl, results, matched_scenes)
            except Exception as e:
                print(f"⚠️ LLM 重排异常，降级: {e}")
        
        # === 排序偏好（价格优先级最高）===
        if dsl.sort_by == "price_asc":
            results.sort(key=lambda x: float(x.get("price", 9999)))
        elif dsl.sort_by == "price_desc":
            results.sort(key=lambda x: -float(x.get("price", 0)))
        
        return results[:top_k]

    
    def _rrf_fusion(self, scores: dict, k=60):
        """RRF 融合（含 kg 第四路）"""
        rank_dicts = {}
        for src in ["vec", "img", "bm25", "kg"]:
            items = [(pid, s.get(src, 0)) for pid, s in scores.items() if src in s]
            items.sort(key=lambda x: -x[1])
            rank_dicts[src] = {pid: r+1 for r, (pid, _) in enumerate(items)}
        fused = {pid: sum(1/(k + rd.get(pid, 10000)) for rd in rank_dicts.values())
                 for pid in scores}
        return sorted(fused.items(), key=lambda x: -x[1])
    
    def _match_filter(self, pid, dsl: QueryDSL) -> bool:
        p = self.products[pid]
        if dsl.brand and p.get("brand") != dsl.brand: return False
        if dsl.category and p.get("category") != dsl.category: return False
        if dsl.price_max and float(p.get("price",0)) > dsl.price_max: return False
        if dsl.price_min and float(p.get("price",0)) < dsl.price_min: return False
        return True

    # ============================================================
    # 品类关键词硬过滤
    # ============================================================
    def _extract_category_keyword(self, dsl: QueryDSL) -> Optional[str]:
        """
        从 dsl 中识别品类词。
        - 优先使用 dsl.category
        - 其次取 dsl.query_text，但要求长度 ≤ 6 且不在场景/风格词黑名单
        - 长查询和闲聊不参与，避免误杀
        """
        if dsl.category:
            return str(dsl.category).strip()
        
        qt = (dsl.query_text or "").strip()
        if not qt or len(qt) > 6:
            return None
        if qt in _NON_CATEGORY_WORDS:
            return None
        # 必须含中文或字母，纯数字/标点不算
        if not any('\u4e00' <= ch <= '\u9fff' or ch.isalpha() for ch in qt):
            return None
        return qt
    
    def _product_contains(self, pid, keyword: str) -> bool:
        """商品任一文本字段包含关键词"""
        p = self.products.get(pid)
        if not p:
            return False
        text = " ".join([
            str(p.get("title", "")),
            str(p.get("dense_caption", "")),
            str(p.get("description", "")),
            str(p.get("category", "")),
        ])
        return keyword in text
