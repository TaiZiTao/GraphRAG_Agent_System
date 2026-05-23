"""Neo4j 知识图谱 - 生产级实现"""
from neo4j import GraphDatabase
from typing import List, Dict, Optional
from config import Config
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

class Neo4jKG:
    """Neo4j 知识图谱客户端"""
    
    def __init__(self):
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USER, Config.NEO4J_PASSWORD)
        )
        self._verify_connection()
    
    def _verify_connection(self):
        try:
            with self.driver.session(database=Config.NEO4J_DATABASE) as session:
                result = session.run("RETURN 1 AS ok")
                result.single()
            print("✅ Neo4j 连接成功")
        except Exception as e:
            print(f"❌ Neo4j 连接失败: {e}")
            raise
    
    def close(self):
        self.driver.close()
    
    # ========================================================
    # 写入操作
    # ========================================================
    
    def clear_all(self):
        """⚠️ 清空所有数据（仅开发用）"""
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("🗑️ 已清空 Neo4j 数据库")
    
    def create_indexes(self):
        """创建索引（提速 100 倍）"""
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            queries = [
                "CREATE INDEX scene_name IF NOT EXISTS FOR (s:Scene) ON (s.name)",
                "CREATE INDEX attr_name IF NOT EXISTS FOR (a:Attribute) ON (a.name)",
                "CREATE INDEX brand_name IF NOT EXISTS FOR (b:Brand) ON (b.name)",
                "CREATE INDEX category_name IF NOT EXISTS FOR (c:Category) ON (c.name)",
                "CREATE INDEX product_id IF NOT EXISTS FOR (p:Product) ON (p.id)",
            ]
            for q in queries:
                session.run(q)
            print("✅ 索引已创建")
    
    def add_scene(self, name: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("MERGE (s:Scene {name: $name})", name=name)
    
    def add_attribute(self, name: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("MERGE (a:Attribute {name: $name})", name=name)
    
    def add_brand(self, name: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("MERGE (b:Brand {name: $name})", name=name)
    
    def add_category(self, name: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("MERGE (c:Category {name: $name})", name=name)
    
    def add_product(self, product: Dict):
        """添加商品节点（包含完整属性）"""
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MERGE (p:Product {id: $id})
                SET p.name = $name,
                    p.price = $price,
                    p.description = $description,
                    p.image_path = $image_path
            """, 
                id=product["id"],
                name=product.get("name", ""),
                price=float(product.get("price", 0)),
                description=product.get("description", "")[:500],
                image_path=product.get("image_path", "")
            )
    
    def link_scene_requires_attr(self, scene: str, attr: str, weight: float = 1.0):
        """场景 -REQUIRES-> 属性"""
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (s:Scene {name: $scene})
                MATCH (a:Attribute {name: $attr})
                MERGE (s)-[r:REQUIRES]->(a)
                SET r.weight = $weight
            """, scene=scene, attr=attr, weight=weight)
    
    def link_attr_often_has(self, src: str, tgt: str, weight: float = 0.5):
        """属性 -OFTEN_HAS-> 属性（二跳）"""
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (a1:Attribute {name: $src})
                MATCH (a2:Attribute {name: $tgt})
                MERGE (a1)-[r:OFTEN_HAS]->(a2)
                SET r.weight = $weight
            """, src=src, tgt=tgt, weight=weight)
    
    def link_brand_has_style(self, brand: str, style: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (b:Brand {name: $brand})
                MATCH (a:Attribute {name: $style})
                MERGE (b)-[:HAS_STYLE]->(a)
            """, brand=brand, style=style)
    
    def link_product_to_brand(self, product_id: str, brand: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (p:Product {id: $pid})
                MATCH (b:Brand {name: $brand})
                MERGE (p)-[:BELONGS_TO]->(b)
            """, pid=product_id, brand=brand)
    
    def link_product_to_category(self, product_id: str, category: str):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (p:Product {id: $pid})
                MATCH (c:Category {name: $cat})
                MERGE (p)-[:IS_A]->(c)
            """, pid=product_id, cat=category)
    
    def link_product_to_attr(self, product_id: str, attr: str, weight: float = 1.0):
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            session.run("""
                MATCH (p:Product {id: $pid})
                MATCH (a:Attribute {name: $attr})
                MERGE (p)-[r:HAS_ATTR]->(a)
                SET r.weight = $weight
            """, pid=product_id, attr=attr, weight=weight)
    
    # ========================================================
    # 查询操作（核心能力）
    # ========================================================
    
    def expand_scene(self, scene: str, max_hops: int = 2) -> List[str]:
        """场景扩展：从场景出发，沿 REQUIRES 和 OFTEN_HAS 边推理属性"""
        cypher = """
        MATCH (s:Scene {name: $scene})-[:REQUIRES]->(a1:Attribute)
        OPTIONAL MATCH (a1)-[:OFTEN_HAS*0..1]->(a2:Attribute)
        WITH collect(DISTINCT a1.name) + collect(DISTINCT a2.name) AS all_attrs
        UNWIND all_attrs AS attr
        WITH attr WHERE attr IS NOT NULL
        RETURN DISTINCT attr
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, scene=scene)
            return [r["attr"] for r in result]
    
    def expand_brand(self, brand: str) -> List[str]:
        """品牌扩展：找出该品牌的风格属性"""
        cypher = """
        MATCH (b:Brand {name: $brand})-[:HAS_STYLE]->(a:Attribute)
        RETURN a.name AS attr
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, brand=brand)
            return [r["attr"] for r in result]
    
    def find_products_by_scene(self, scene: str, limit: int = 20) -> List[Dict]:
        """⭐ 强大查询：基于场景推荐商品"""
        cypher = """
        MATCH (s:Scene {name: $scene})-[:REQUIRES]->(a:Attribute)
        WITH collect(a.name) AS scene_attrs
        MATCH (p:Product)-[:HAS_ATTR]->(pa:Attribute)
        WHERE pa.name IN scene_attrs
        WITH p, count(DISTINCT pa) AS match_count, scene_attrs
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN p.id AS id, p.name AS name, p.price AS price, 
               match_count, scene_attrs
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, scene=scene, limit=limit)
            return [dict(r) for r in result]
    
    def find_similar_products(self, product_id: str, limit: int = 5) -> List[Dict]:
        """⭐ 基于图的相似商品推荐"""
        cypher = """
        MATCH (p1:Product {id: $pid})-[:HAS_ATTR]->(a:Attribute)<-[:HAS_ATTR]-(p2:Product)
        WHERE p1 <> p2
        WITH p2, count(a) AS shared_attrs
        
        OPTIONAL MATCH (p1)-[:BELONGS_TO]->(b:Brand)<-[:BELONGS_TO]-(p2)
        WITH p2, shared_attrs, CASE WHEN b IS NOT NULL THEN 2 ELSE 0 END AS brand_bonus
        
        OPTIONAL MATCH (p1)-[:IS_A]->(c:Category)<-[:IS_A]-(p2)
        WITH p2, shared_attrs + brand_bonus + CASE WHEN c IS NOT NULL THEN 1 ELSE 0 END AS score
        
        RETURN p2.id AS id, p2.name AS name, p2.price AS price, score
        ORDER BY score DESC
        LIMIT $limit
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, pid=product_id, limit=limit)
            return [dict(r) for r in result]
    
    def get_reasoning_path(self, scene: str, product_id: str) -> List[str]:
        """⭐ 推荐可解释性：返回「为什么推荐」的路径"""
        cypher = """
        MATCH path = (s:Scene {name: $scene})-[:REQUIRES]->(a:Attribute)<-[:HAS_ATTR]-(p:Product {id: $pid})
        RETURN [n IN nodes(path) | coalesce(n.name, n.id)] AS reasoning
        LIMIT 3
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, scene=scene, pid=product_id)
            return [r["reasoning"] for r in result]
    
    def get_stats(self) -> Dict:
        """统计图谱规模"""
        cypher = """
        MATCH (n) 
        RETURN labels(n)[0] AS label, count(*) AS cnt
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            stats = {r["label"]: r["cnt"] for r in session.run(cypher)}
            
            rel_result = session.run("""
                MATCH ()-[r]->() 
                RETURN type(r) AS rel, count(*) AS cnt
            """)
            relations = {r["rel"]: r["cnt"] for r in rel_result}
            
            return {"nodes": stats, "relationships": relations}
    
    # ========================================================
    # L3 新增：场景模糊匹配 + 解释路径
    # ========================================================
    
    def get_all_scenes(self) -> List[str]:
        """获取所有场景名"""
        cypher = "MATCH (s:Scene) RETURN s.name AS name ORDER BY s.name"
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            return [r["name"] for r in session.run(cypher)]
    
    # ========================================================
    # ⭐ L5 GraphRAG: 3 层兜底场景识别（reranker 版）
    # ========================================================
    def match_scenes_by_query(self, query: str, max_scenes: int = 3) -> List[str]:
        if not query:
            return []
        
        if not getattr(self, "_scene_cache_loaded", False):
            self._init_scene_cache()
        
        if not self._scene_attrs_cache:
            return []
        
        scored: Dict[str, float] = {}
        
        # 第 1 层：字面匹配
        for scene in self._scene_attrs_cache.keys():
            if scene in query or query in scene:
                scored[scene] = max(scored.get(scene, 0), 1.0)
                continue
            char_hits = sum(1 for c in scene if c in query)
            if char_hits >= 2:
                scored[scene] = max(scored.get(scene, 0), 0.5 + 0.1 * char_hits)
        
        # 第 2 层：属性反向触发
        for scene, attrs in self._scene_attrs_cache.items():
            hit_attrs = [a for a in (attrs or []) if a and a in query]
            if hit_attrs:
                s = min(0.6 + 0.1 * len(hit_attrs), 0.95)
                scored[scene] = max(scored.get(scene, 0), s)
        
        # 第 3 层：Reranker 跨编码语义对齐
        if self._scene_reranker is not None and self._scene_descs:
            try:
                pairs = [(query, desc) for desc in self._scene_descs.values()]
                names = list(self._scene_descs.keys())
                raw = self._scene_reranker.predict(pairs)
                import numpy as np
                # ============ 自适应阈值 ============
                norm = 1 / (1 + np.exp(-np.asarray(raw)))  # sigmoid → [0,1]
                top_score = float(np.max(norm)) if len(norm) > 0 else 0.0
        
                # 按 top_score 自适应
                if top_score >= 0.65:
                    # 高置信：允许多场景共存（如"海边度假"激活海边+旅游）
                    abs_thr = 0.55
                    rel_thr = top_score - 0.20
                elif top_score >= 0.50:
                    # 中置信：只保留榜首，次榜大概率是噪声
                    abs_thr = top_score - 0.001
                    rel_thr = 0.0
                else:
                    # 低置信：完全放弃语义层（让向量/BM25 兜底）
                    abs_thr = 1.01
                    rel_thr = 1.01
        
                # 收集 + 排序 + 截断
                candidates = []
                for name, score in zip(names, norm):
                    score = float(score)
                    if score >= abs_thr and score >= rel_thr:
                        candidates.append((name, score))
                
                candidates.sort(key=lambda x: -x[1])
                for name, score in candidates[:2]:  # 最多 2 个场景
                    scored[name] = max(scored.get(name, 0), score)
        


            except Exception as e:
                print(f"⚠️ [KG] reranker 异常: {e}")
        
        ranked = sorted(scored.items(), key=lambda x: -x[1])
        result = [s for s, _ in ranked[:max_scenes]]
        if result:
            top = ", ".join(f"{n}({s:.2f})" for n, s in ranked[:max_scenes])
            print(f"🌐 [KG] '{query}' → [{top}]")
        else:
            print(f"⚠️ [KG] '{query}' 未匹配到场景")
        return result
    
    def _init_scene_cache(self):
        """懒加载：场景缓存 + reranker（仅一次）"""
        self._scene_cache_loaded = True
        self._scene_attrs_cache: Dict[str, List[str]] = {}
        self._scene_descs: Dict[str, str] = {}
        self._scene_reranker = None
        
        # 1. 拉场景 + 属性
        cypher = """
        MATCH (s:Scene)
        OPTIONAL MATCH (s)-[:REQUIRES]->(a:Attribute)
        RETURN s.name AS name, collect(DISTINCT a.name) AS attrs
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            for r in session.run(cypher):
                attrs = [a for a in r["attrs"] if a]
                self._scene_attrs_cache[r["name"]] = attrs
                self._scene_descs[r["name"]] = f"{r['name']} {' '.join(attrs)}".strip()
        
        if not self._scene_attrs_cache:
            print("⚠️ [KG] 场景缓存为空")
            return
        
        # 2. 加载 reranker (CrossEncoder)，强制本地离线
        try:
            from sentence_transformers import CrossEncoder
            from pathlib import Path
            ROOT = Path(__file__).resolve().parent.parent
            #model_path = ROOT / "models" / "BAAI" / "bge-reranker-v2-m3"
            model_path = ROOT / "models" /  "bge-reranker-v2-m3-ft"
            if not model_path.exists():
                print(f"⚠️ [KG] reranker 不存在: {model_path}，降级")
                return
            
            self._scene_reranker = CrossEncoder(
                str(model_path),
                max_length=128,
                local_files_only=True,   # 🔒 不联网
            )
            print(f"✅ [KG] 场景缓存就绪 ({len(self._scene_attrs_cache)} 个场景, reranker={model_path.name})")
        except Exception as e:
            print(f"⚠️ [KG] reranker 加载失败: {e}")
            self._scene_reranker = None


    
    def find_products_by_scenes(self, scenes: List[str], limit: int = 30) -> List[Dict]:
        """多场景联合召回（取并集，按命中数排序）"""
        if not scenes:
            return []
        cypher = """
        UNWIND $scenes AS scene_name
        MATCH (s:Scene {name: scene_name})-[:REQUIRES]->(a:Attribute)
        WITH collect(DISTINCT a.name) AS scene_attrs
        MATCH (p:Product)-[:HAS_ATTR]->(pa:Attribute)
        WHERE pa.name IN scene_attrs
        WITH p, count(DISTINCT pa) AS match_count, 
             collect(DISTINCT pa.name) AS hits
        ORDER BY match_count DESC
        LIMIT $limit
        RETURN p.id AS id, p.name AS name, p.price AS price,
               match_count, hits
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            result = session.run(cypher, scenes=scenes, limit=limit)
            return [dict(r) for r in result]
    
    def get_product_explain(self, product_id: str, scenes: List[str]) -> Dict:
        """
        获取商品在指定场景下的解释路径
        
        返回:
        {
          "scenes": ["旅游度假"],           # 命中的场景
          "matched_attrs": ["棉麻","宽松"],  # 商品 ∩ 场景的共同属性
          "all_attrs": [...]                # 商品所有属性（参考）
        }
        """
        empty = {"scenes": [], "matched_attrs": [], "all_attrs": []}
        if not product_id or not scenes:
            return empty
        
        cypher_attrs = """
        MATCH (p:Product {id: $pid})-[:HAS_ATTR]->(a:Attribute)
        RETURN collect(DISTINCT a.name) AS attrs
        """
        cypher_scene = """
        MATCH (s:Scene {name: $scene})-[:REQUIRES]->(a:Attribute)
        RETURN collect(DISTINCT a.name) AS attrs
        """
        with self.driver.session(database=Config.NEO4J_DATABASE) as session:
            r = session.run(cypher_attrs, pid=product_id).single()
            prod_attrs = (r["attrs"] if r else []) or []
            
            matched_scenes = []
            all_matched = set()
            for scene in scenes:
                r2 = session.run(cypher_scene, scene=scene).single()
                scene_attrs = (r2["attrs"] if r2 else []) or []
                common = set(prod_attrs) & set(scene_attrs)
                if common:
                    matched_scenes.append(scene)
                    all_matched.update(common)
        
        return {
            "scenes": matched_scenes,
            "matched_attrs": sorted(all_matched),
            "all_attrs": prod_attrs
        }
if __name__ == "__main__":
    kg = Neo4jKG()
    test_queries = [
        "我要去健身",
        "大理旅游穿什么", 
        "约会穿搭",
        "上班通勤",
        "去爬山",
        "海边度假",
    ]
    for q in test_queries:
        scenes = kg.match_scenes_by_query(q)
        print(f"  query='{q}' → {scenes}\n")
    kg.close()
