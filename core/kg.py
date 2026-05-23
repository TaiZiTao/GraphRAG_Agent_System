"""KG 统一入口 - 自动选择后端"""
from config import Config

def get_kg():
    """工厂方法：根据配置选择 KG 后端"""
    if Config.KG_BACKEND == "neo4j":
        try:
            from core.kg_neo4j import Neo4jKG
            return Neo4jKG()
        except Exception as e:
            print(f"⚠️ Neo4j 不可用，降级到 NetworkX: {e}")
            return _get_networkx_kg()
    return _get_networkx_kg()


def _get_networkx_kg():
    """原有 NetworkX 实现（保留）"""
    return SceneKG()


# ===== 保留原有 NetworkX 实现作为降级 =====
import networkx as nx
from typing import List

class SceneKG:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._bootstrap()
    
    def _bootstrap(self):
        scene_rules = {
            "大理旅游": ["民族风", "棉麻", "防晒", "宽松", "亮色", "度假风"],
            "海边度假": ["度假风", "亚麻", "雪纺", "防晒", "亮色", "宽松"],
            "通勤": ["商务", "简约", "正装", "深色", "修身"],
            "健身": ["运动", "速干", "弹力", "透气"],
            "约会": ["甜美", "气质", "修身", "法式"],
            "居家": ["舒适", "宽松", "棉质"],
            "派对": ["亮片", "礼服", "时尚"],
        }
        for scene, attrs in scene_rules.items():
            self.g.add_node(f"scene:{scene}", type="Scene")
            for a in attrs:
                self.g.add_node(f"attr:{a}", type="Attribute")
                self.g.add_edge(f"scene:{scene}", f"attr:{a}", rel="REQUIRES")
        
        secondary = {
            "民族风": ["刺绣", "印花"],
            "运动": ["速干"],
            "商务": ["衬衫"],
        }
        for src, tgts in secondary.items():
            for t in tgts:
                self.g.add_node(f"attr:{t}", type="Attribute")
                self.g.add_edge(f"attr:{src}", f"attr:{t}", rel="OFTEN_HAS")
    
    def expand_scene(self, scene: str) -> List[str]:
        node = f"scene:{scene}"
        if node not in self.g: return []
        direct = [v.split(":",1)[1] for u,v,d in self.g.out_edges(node, data=True)
                  if d.get("rel") == "REQUIRES"]
        extended = []
        for a in direct:
            an = f"attr:{a}"
            if an in self.g:
                extended.extend([v.split(":",1)[1] for u,v,d in self.g.out_edges(an, data=True)
                                 if d.get("rel") == "OFTEN_HAS"])
        return list(set(direct + extended))
    
    def find_products_by_scene(self, scene): return []
    def find_similar_products(self, pid): return []
    def expand_brand(self, b): return []
    def get_reasoning_path(self, scene, pid): return []
    def close(self): pass
