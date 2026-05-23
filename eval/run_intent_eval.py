"""意图识别评估"""
import sys, os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.chat_models import ChatTongyi
from config import Config
from core.query_parser import QueryParser
from eval.eval_dataset import INTENT_DATASET


def evaluate():
    llm = ChatTongyi(model_name=Config.LLM_TURBO, temperature=0)
    parser = QueryParser(llm)

    correct = 0
    confusion = defaultdict(lambda: defaultdict(int))
    failures = []

    print(f"开始评估 {len(INTENT_DATASET)} 条意图识别...\n")

    for i, item in enumerate(INTENT_DATASET, 1):
        q, exp = item["query"], item["expected"]
        try:
            dsl = parser.parse(q, has_image=False)
            pred = dsl.intent.value if hasattr(dsl.intent, "value") else str(dsl.intent)
        except Exception as e:
            pred = "ERROR"
            print(f"  ⚠️ [{item['id']}] 异常: {e}")

        confusion[exp][pred] += 1
        ok = (pred == exp)
        if ok:
            correct += 1
        else:
            failures.append((item["id"], q, exp, pred))
        mark = "✓" if ok else "✗"
        print(f"  [{i:2d}/{len(INTENT_DATASET)}] {mark} [{item['id']}] "
              f"{q!r:32s} 期望={exp:10s} 实际={pred}")

    total = len(INTENT_DATASET)
    print("\n" + "=" * 70)
    print(f"总准确率: {correct}/{total} = {correct/total:.2%}")
    print("=" * 70)

    cls_total = defaultdict(int)
    for item in INTENT_DATASET:
        cls_total[item["expected"]] += 1
    cls_correct = {cls: confusion[cls].get(cls, 0) for cls in cls_total}

    print("\n【按类别准确率】")
    for cls in sorted(cls_total.keys()):
        c, t = cls_correct[cls], cls_total[cls]
        bar = "█" * int(c / t * 20) if t else ""
        print(f"  {cls:12s} {c}/{t} = {c/t:6.1%}  {bar}")

    print("\n【混淆矩阵】(行=期望, 列=实际)")
    all_labels = sorted(set(cls_total.keys()) | {p for preds in confusion.values() for p in preds})
    print("  期望\\实际    " + " ".join(f"{l[:6]:>7s}" for l in all_labels))
    for exp_cls in sorted(cls_total.keys()):
        row = " ".join(f"{confusion[exp_cls].get(p, 0):>7d}" for p in all_labels)
        print(f"  {exp_cls:12s} {row}")

    if failures:
        print(f"\n【失败详情 {len(failures)} 条】")
        for fid, q, exp, pred in failures:
            print(f"  [{fid}] {q!r:32s} 期望={exp:10s} → 实际={pred}")


if __name__ == "__main__":
    evaluate()
