"""
L1.5 v3: 基于实际数据重写 Scene 词典 + 修商品名
关键改动:
1. Scene 词典对齐 LLM 实际抽出的高频词（日常/旅游/潮牌/夏 等）
2. 商品 name 取自 description（截断 30 字）
3. 同义词归一化扩展
4. 末尾打印场景召回覆盖检查表
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
LLM_ATTRS_PATH = ROOT / "data" / "llm_attributes.json"

# ========= 新 Scene 词典：基于商品实际属性设计 =========
SCENE_ATTRS = {
    "日常休闲":   ["日常", "休闲", "舒适", "宽松", "纯色", "百搭"],
    "旅游度假":   ["旅游", "度假", "拍照", "透气", "棉麻", "宽松"],
    "海边度假":   ["海边", "度假", "拍照", "透气", "宽松", "夏"],
    "夏日清爽":   ["夏", "夏季", "透气", "宽松", "纯色", "凉感"],
    "潮流街头":   ["潮牌", "潮流", "嘻哈", "港风", "设计感", "复古"],
    "复古文艺":   ["复古", "棉麻", "印花", "文艺", "纯色"],
    "通勤简约":   ["通勤", "简约", "纯色", "百搭", "气质", "OL"],
    "约会甜美":   ["甜美", "显瘦", "修身", "印花", "气质"],
    "运动户外":   ["运动", "宽松", "透气", "舒适", "百搭"],
    "学生校园":   ["学生", "学院", "休闲", "潮流", "甜美"],
    "轻熟优雅":   ["轻熟", "优雅", "气质", "修身", "简约"],
    "聚会派对":   ["设计感", "潮流", "印花", "亮色", "修身"],
}

# 属性二跳关系（常一起出现）
ATTR_PAIRS = [
    ("潮牌", "嘻哈"), ("潮牌", "复古"), ("潮流", "潮牌"),
    ("旅游", "度假"), ("度假", "拍照"), ("度假", "海边"),
    ("夏", "透气"), ("夏季", "凉感"), ("透气", "宽松"),
    ("休闲", "舒适"), ("休闲", "宽松"), ("日常", "百搭"),
    ("通勤", "简约"), ("通勤", "OL"), ("简约", "百搭"),
    ("复古", "文艺"), ("复古", "印花"), ("棉麻", "文艺"),
]

BRAND_STYLES = {
    "梵先生Mr.Vanset": ["潮牌", "复古", "嘻哈"],
    "JEEP SPIRIT":     ["休闲", "户外", "百搭"],
    "WASSUP Line":     ["潮流", "休闲", "百搭"],
    "NASA联名潮牌":     ["潮牌", "潮流", "印花"],
    "公牛世家":         ["商务", "通勤", "简约"],
    "it izzue":        ["潮流", "港风", "设计感"],
    "CatumKezaf":      ["设计感", "潮流", "拍照"],
}

# 同义词归一化
SYNONYM = {
    "民族风格": "民族风",
    "波西米亚": "民族风",
    "boho":     "民族风",
    "海滩": "海边",
    "海岛": "海边",
    "白领": "OL",
    "上班": "通勤",
    "职场": "通勤",
    "舒服": "舒适",
    "夏季": "夏",
    "夏天": "夏",
    "冬季": "冬",
    "春季": "春",
    "秋季": "秋",
    "凉爽": "凉感",
    "潮": "潮流",
    "国潮": "潮牌",
    "随性": "宽松",
}


def normalize(s: str) -> str:
    s = s.strip()
    return SYNONYM.get(s, s)


def truncate(text: str, n: int = 30) -> str:
    if not text:
        return ""
    text = text.strip()
    return text[:n] + ("..." if len(text) > n else "")


class KGBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER, PWD))
        print("✅ Neo4j 已连接")

    def close(self):
        self.driver.close()

    def run(self, cypher: str, **params):
        with self.driver.session() as s:
            return list(s.run(cypher, **params))

    def clear(self):
        self.run("MATCH (n) DETACH DELETE n")
        print("🗑️ 数据库已清空")

    def create_indexes(self):
        for q in [
            "CREATE INDEX scene_name IF NOT EXISTS FOR (s:Scene) ON (s.name)",
            "CREATE INDEX attr_name IF NOT EXISTS FOR (a:Attribute) ON (a.name)",
            "CREATE INDEX brand_name IF NOT EXISTS FOR (b:Brand) ON (b.name)",
            "CREATE INDEX cat_name IF NOT EXISTS FOR (c:Category) ON (c.name)",
            "CREATE INDEX product_id IF NOT EXISTS FOR (p:Product) ON (p.id)",
        ]:
            self.run(q)
        print("✅ 索引已建好")

    def build_scenes(self):
        print("\n📍 写场景与属性...")
        for scene, attrs in tqdm(SCENE_ATTRS.items()):
            self.run("MERGE (s:Scene {name: $name})", name=scene)
            for a in attrs:
                self.run(
                    """
                    MERGE (a:Attribute {name: $a})
                    WITH a
                    MATCH (s:Scene {name: $s})
                    MERGE (s)-[:REQUIRES]->(a)
                    """,
                    s=scene, a=normalize(a),
                )

    def build_attr_pairs(self):
        print("\n🔗 写属性二跳...")
        for a1, a2 in tqdm(ATTR_PAIRS):
            self.run(
                """
                MERGE (x:Attribute {name: $a1})
                MERGE (y:Attribute {name: $a2})
                MERGE (x)-[:OFTEN_HAS]->(y)
                MERGE (y)-[:OFTEN_HAS]->(x)
                """,
                a1=normalize(a1), a2=normalize(a2),
            )

    def build_brand_styles(self):
        print("\n🏷️ 写品牌风格...")
        for brand, styles in tqdm(BRAND_STYLES.items()):
            self.run("MERGE (b:Brand {name: $b})", b=brand)
            for s in styles:
                self.run(
                    """
                    MERGE (a:Attribute {name: $a})
                    WITH a
                    MATCH (b:Brand {name: $b})
                    MERGE (b)-[:HAS_STYLE]->(a)
                    """,
                    b=brand, a=normalize(s),
                )

    def build_products_with_llm(self):
        print("\n📦 写商品 + LLM 属性...")
        if not LLM_ATTRS_PATH.exists():
            print(f"❌ 找不到 {LLM_ATTRS_PATH}")
            return

        with open(LLM_ATTRS_PATH, encoding="utf-8") as f:
            llm_data = json.load(f)

        # 加载原 products
        products_raw = []
        for path in [ROOT / "data" / "products.json", ROOT / "data" / "items.json"]:
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    products_raw = data if isinstance(data, list) else data.get("products", [])
                break

        raw_index = {str(p.get("id") or p.get("product_id") or ""): p for p in products_raw}

        attr_edge_count = 0
        for pid, item in tqdm(llm_data.items()):
            raw = raw_index.get(pid, {})
            # 关键修复：name 用 description 截断
            name = item.get("name") or truncate(str(raw.get("description", "")))
            brand = str(raw.get("brand") or "").strip()
            category = str(raw.get("category") or raw.get("sub_category") or "服装")
            price = float(raw.get("price") or 0)

            self.run(
                """
                MERGE (p:Product {id: $pid})
                SET p.name = $name, p.price = $price
                """,
                pid=pid, name=name, price=price,
            )

            if category:
                self.run(
                    """
                    MERGE (c:Category {name: $c})
                    WITH c
                    MATCH (p:Product {id: $pid})
                    MERGE (p)-[:IS_A]->(c)
                    """,
                    c=category, pid=pid,
                )

            if brand:
                self.run(
                    """
                    MERGE (b:Brand {name: $b})
                    WITH b
                    MATCH (p:Product {id: $pid})
                    MERGE (p)-[:BELONGS_TO]->(b)
                    """,
                    b=brand, pid=pid,
                )

            attrs = item.get("attributes", {})
            for field, values in attrs.items():
                if not isinstance(values, list):
                    continue
                for v in values:
                    if not v or not isinstance(v, str):
                        continue
                    attr_name = normalize(v)
                    self.run(
                        """
                        MERGE (a:Attribute {name: $a})
                        WITH a
                        MATCH (p:Product {id: $pid})
                        MERGE (p)-[r:HAS_ATTR]->(a)
                        SET r.field = $field
                        """,
                        a=attr_name, pid=pid, field=field,
                    )
                    attr_edge_count += 1

        print(f"  └── 共写 {attr_edge_count} 条 HAS_ATTR")

    def stats(self):
        print("\n" + "=" * 50)
        print("📊 图谱统计")
        print("=" * 50)
        for r in self.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC"
        ):
            print(f"  {r['label']}: {r['cnt']}")
        print("\n关系:")
        for r in self.run(
            "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt ORDER BY cnt DESC"
        ):
            print(f"  {r['rel']}: {r['cnt']}")

    def coverage_check(self):
        """关键诊断：每个 Scene 能召回几个商品"""
        print("\n" + "=" * 50)
        print("🎯 场景召回覆盖检查")
        print("=" * 50)
        rows = self.run("""
            MATCH (s:Scene)
            OPTIONAL MATCH (s)-[:REQUIRES]->(a)<-[:HAS_ATTR]-(p:Product)
            RETURN s.name AS scene,
                   count(DISTINCT p) AS p_count,
                   count(DISTINCT a) AS a_count
            ORDER BY p_count DESC
        """)
        print(f"{'场景':<12} {'命中商品':<10} {'命中属性':<10}")
        print("-" * 36)
        for r in rows:
            mark = "✅" if r["p_count"] > 0 else "❌"
            print(f"{r['scene']:<12} {r['p_count']:<10} {r['a_count']:<10} {mark}")


def main():
    print("🚀 L1.5 v3: 重建 KG（修 name + 对齐 Scene）")
    print("=" * 50)
    builder = KGBuilder()
    try:
        builder.clear()
        builder.create_indexes()
        builder.build_scenes()
        builder.build_attr_pairs()
        builder.build_brand_styles()
        builder.build_products_with_llm()
        builder.stats()
        builder.coverage_check()
        print("\n✅ KG v3 重建完成")
    finally:
        builder.close()


if __name__ == "__main__":
    main()
