"""GraphRAG 评估 - 不依赖 LLM，支持多接受答案"""
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.eval_dataset_graphrag import GRAPHRAG_DATASET
from core.kg_neo4j import Neo4jKG


def evaluate():
    kg = Neo4jKG()
    try:
        scene_attrs_cache = {s: set(kg.expand_scene(s)) for s in kg.get_all_scenes()}
        print(f"已加载 {len(scene_attrs_cache)} 个场景的属性\n")

        scene_correct = 0
        attr_iou_sum = 0.0
        by_abs = defaultdict(lambda: {"n": 0, "hit": 0, "iou": 0.0})
        failures = []

        for i, item in enumerate(GRAPHRAG_DATASET, 1):
            q = item["query"]
            acceptable = item["acceptable_scenes"]
            primary = acceptable[0]  # 主答案，用于 IoU
            ab = item["abstraction"]

            try:
                pred_scenes = kg.match_scenes_by_query(q, max_scenes=1)
                pred_scene = pred_scenes[0] if pred_scenes else None
            except Exception as e:
                print(f"  ⚠️ [{item['id']}] 异常: {e}")
                pred_scene = None

            # 属性 IoU 用主答案
            exp_attrs = scene_attrs_cache.get(primary, set())
            pred_attrs = scene_attrs_cache.get(pred_scene, set()) if pred_scene else set()
            if exp_attrs or pred_attrs:
                iou = len(exp_attrs & pred_attrs) / max(1, len(exp_attrs | pred_attrs))
            else:
                iou = 0.0

            hit = (pred_scene in acceptable)
            if hit:
                scene_correct += 1
            else:
                failures.append((item["id"], q, acceptable, pred_scene, ab, iou))
            attr_iou_sum += iou

            b = by_abs[ab]
            b["n"] += 1
            b["hit"] += int(hit)
            b["iou"] += iou

            mark = "✓" if hit else "✗"
            acc_str = "/".join(acceptable)
            print(f"  [{i:2d}/{len(GRAPHRAG_DATASET)}] {mark} [{item['id']}|{ab:4s}] "
                  f"{q!r:18s} 接受={acc_str:20s} 实际={(pred_scene or 'None'):6s} "
                  f"属性IoU={iou:.0%}")

        n = len(GRAPHRAG_DATASET)
        print("\n" + "=" * 70)
        print(f"场景命中率（多答案）: {scene_correct}/{n} = {scene_correct/n:.2%}")
        print(f"属性 IoU 平均:       {attr_iou_sum/n:.2%}")
        print("=" * 70)

        print("\n【按抽象级别】")
        for level in ["high", "mid", "low"]:
            b = by_abs[level]
            if b["n"] == 0: continue
            print(f"  {level:<5s} N={b['n']:2d}  "
                  f"场景命中={b['hit']/b['n']:>6.1%}  "
                  f"属性IoU={b['iou']/b['n']:>6.1%}")

        if failures:
            print(f"\n【场景识别失败 {len(failures)} 条】")
            for fid, q, acc, pred, ab, iou in failures:
                acc_str = "/".join(acc)
                print(f"  [{fid}|{ab:4s}] {q!r:18s} 接受={acc_str:20s} → 实际={pred or 'None':6s} (IoU={iou:.0%})")
    finally:
        kg.close()


if __name__ == "__main__":
    evaluate()
