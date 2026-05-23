"""从评估集 + KG 生成 reranker 微调数据
passage 格式必须和 core/kg_neo4j.py 的 _scene_descs 完全一致：
  "{场景名} {attr1} {attr2} ... {attrN}"
"""
import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_dataset_graphrag import GRAPHRAG_DATASET
from core.kg_neo4j import Neo4jKG

random.seed(42)


def scene_passage(scene_name: str, attrs: list) -> str:
    """严格对齐 _scene_descs 格式"""
    return f"{scene_name} {' '.join(attrs)}".strip()


def main():
    kg = Neo4jKG()
    try:
        # 触发缓存加载
        kg._init_scene_cache()
        all_scenes = list(kg._scene_attrs_cache.keys())
        attrs_map = {s: list(kg._scene_attrs_cache[s]) for s in all_scenes}

        print(f"📊 KG 场景数: {len(all_scenes)}")
        print(f"📊 评估集样本数: {len(GRAPHRAG_DATASET)}")
        print(f"\n📝 passage 样例:")
        sample_scene = all_scenes[0]
        print(f"   {scene_passage(sample_scene, attrs_map[sample_scene])[:150]}...")

        train_records = []
        skipped = 0

        for item in GRAPHRAG_DATASET:
            q = item["query"]
            pos_scenes = item["acceptable_scenes"]
            
            # 校验场景名都在 KG 里
            valid_pos = [s for s in pos_scenes if s in attrs_map]
            if len(valid_pos) != len(pos_scenes):
                missing = set(pos_scenes) - set(valid_pos)
                print(f"⚠️  {item['id']} 缺失场景 {missing}, 跳过")
                skipped += 1
                continue
            
            neg_scenes = [s for s in all_scenes if s not in pos_scenes]
            
            pos_passages = [scene_passage(s, attrs_map[s]) for s in valid_pos]
            # 随机采 7 个负例
            neg_sample = random.sample(neg_scenes, min(7, len(neg_scenes)))
            neg_passages = [scene_passage(s, attrs_map[s]) for s in neg_sample]

            train_records.append({
                "query": q,
                "pos": pos_passages,
                "neg": neg_passages,
            })

        os.makedirs("data", exist_ok=True)
        out_path = "data/rerank_train.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in train_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
        print(f"\n✅ 写出 {len(train_records)} 条训练样本 → {out_path}")
        print(f"   跳过 {skipped} 条（场景名不在KG中）")
        
        # 预览前 2 条
        print(f"\n📋 前 2 条样本预览:")
        with open(out_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                rec = json.loads(line)
                print(f"\n--- 样本 {i+1} ---")
                print(f"query: {rec['query']}")
                print(f"pos[0]: {rec['pos'][0][:100]}...")
                print(f"neg[0]: {rec['neg'][0][:100]}...")
                print(f"pos 数: {len(rec['pos'])}, neg 数: {len(rec['neg'])}")
    finally:
        kg.close()


if __name__ == "__main__":
    main()
