"""微调 BGE Reranker v2 m3 - 健壮版"""
import os
import sys
import json
import random
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from sentence_transformers import CrossEncoder, InputExample

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = ROOT / "models" / "BAAI" / "bge-reranker-v2-m3"
OUTPUT_DIR = ROOT / "models" / "bge-reranker-v2-m3-ft"
TRAIN_FILE = ROOT / "data" / "rerank_train.jsonl"

EPOCHS = 3
BATCH_SIZE = 16
LR = 2e-5
MAX_LEN = 128
WARMUP_STEPS = 50
SEED = 42

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

random.seed(SEED)
torch.manual_seed(SEED)


def load_examples():
    examples = []
    pos_cnt, neg_cnt = 0, 0
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            q = rec["query"]
            for pos in rec["pos"]:
                examples.append(InputExample(texts=[q, pos], label=1.0))
                pos_cnt += 1
            for neg in rec["neg"]:
                examples.append(InputExample(texts=[q, neg], label=0.0))
                neg_cnt += 1
    print(f"📊 训练样本：{len(examples)} 条 (pos={pos_cnt}, neg={neg_cnt})")
    return examples


def main():
    print(f"🚀 开始微调 BGE Reranker")
    print(f"   base:   {BASE_MODEL}")
    print(f"   output: {OUTPUT_DIR}")
    print(f"   epochs={EPOCHS}, bs={BATCH_SIZE}, lr={LR}, max_len={MAX_LEN}")
    print(f"   torch={torch.__version__}, cuda={torch.cuda.is_available()}\n")

    if not BASE_MODEL.exists():
        raise FileNotFoundError(f"基座模型不存在: {BASE_MODEL}")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"训练数据不存在: {TRAIN_FILE}")

    # ===== Step 1: 加载基座 =====
    print("Step 1/4: 加载基座模型...")
    try:
        model = CrossEncoder(
            str(BASE_MODEL),
            num_labels=1,
            max_length=MAX_LEN,
            local_files_only=True,
        )
        print("  ✅ 基座加载成功\n")
    except Exception as e:
        print(f"  ❌ 基座加载失败: {e}")
        traceback.print_exc()
        print("\n👉 这通常是 numpy/scipy 版本问题，先修依赖再训练")
        sys.exit(1)

    # ===== Step 2: 准备数据 =====
    print("Step 2/4: 准备训练数据...")
    train_examples = load_examples()
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=BATCH_SIZE,
    )
    print(f"  ✅ DataLoader 就绪，共 {len(train_dataloader)} batch\n")

    # ===== Step 3: 训练 =====
    print("Step 3/4: 开始训练...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        model.fit(
            train_dataloader=train_dataloader,
            epochs=EPOCHS,
            warmup_steps=WARMUP_STEPS,
            optimizer_params={"lr": LR},
            output_path=str(OUTPUT_DIR),
            save_best_model=False,
            show_progress_bar=True,
            use_amp=True,
        )
        print("  ✅ fit() 调用完成\n")
    except Exception as e:
        print(f"  ❌ 训练失败: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ===== Step 4: 显式保存（保险） =====
    print("Step 4/4: 显式保存模型...")
    try:
        model.save(str(OUTPUT_DIR))
        print(f"  ✅ 模型已保存至 {OUTPUT_DIR}")
    except Exception as e:
        print(f"  ⚠️  显式 save 失败: {e}")

    # ===== 验证产物 =====
    print("\n📦 输出目录内容:")
    if OUTPUT_DIR.exists():
        files = list(OUTPUT_DIR.rglob("*"))
        for f in files[:20]:
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   {f.relative_to(OUTPUT_DIR)}  ({size_mb:.1f} MB)")
        total_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
        print(f"   总大小: {total_size:.1f} MB")
        
        # 关键文件检查
        must_have = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        weight_files = ["model.safetensors", "pytorch_model.bin"]
        missing = [f for f in must_have if not (OUTPUT_DIR / f).exists()]
        has_weights = any((OUTPUT_DIR / f).exists() for f in weight_files)
        
        if missing:
            print(f"\n⚠️  缺少关键文件: {missing}")
        if not has_weights:
            print(f"\n⚠️  缺少权重文件（model.safetensors 或 pytorch_model.bin）")
        if not missing and has_weights:
            print(f"\n✅ 所有关键文件齐全，可以切换路径使用了")
    else:
        print(f"   ❌ 输出目录不存在")


if __name__ == "__main__":
    main()
