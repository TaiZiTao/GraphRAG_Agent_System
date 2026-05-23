"""
L1: 用 LLM 抽取商品结构化属性
- 输入: 商品数据
- 输出: data/llm_attributes.json
- 特点: 实时保存（防中断）+ 缓存（避免重复花 token）
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, List
from http import HTTPStatus
from dotenv import load_dotenv
from tqdm import tqdm
import dashscope

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

OUTPUT_PATH = ROOT / "data" / "llm_attributes.json"

EXTRACT_PROMPT = """你是一个时尚商品属性分析专家。请从下面的商品信息中抽取结构化属性标签。

【商品名称】{name}
【商品描述】{description}
【品类】{category}
【品牌】{brand}

请严格按以下 JSON 格式输出（不要 markdown 包裹，不要任何额外文字）：
{{
  "style": ["风格1", "风格2"],
  "material": ["材质1"],
  "scene": ["场景1", "场景2"],
  "feeling": ["感受1", "感受2"],
  "color_tone": ["色调1"],
  "season": ["季节1"],
  "target_user": ["人群1"]
}}

要求：
1. 每个字段 1-4 个标签，标签 2-4 个字
2. 风格优先用：民族风/文艺/简约/优雅/复古/甜美/帅气/通勤/休闲/度假/性感/学院
3. 材质如：棉麻/真丝/雪纺/针织/牛仔/蕾丝/纯棉
4. 场景如：旅游/约会/通勤/聚会/拍照/海边/婚礼/日常
5. 感受如：飘逸/显瘦/舒适/透气/保暖/修身/宽松
6. 色调如：莫兰迪/亮色/暗色/纯色/印花
7. 季节：春/夏/秋/冬/四季
8. 目标人群：少女/轻熟/成熟/学生/职场
"""


def call_llm(prompt: str, max_retry: int = 3) -> str:
    """调通义千问，带指数退避重试"""
    for i in range(max_retry):
        try:
            resp = dashscope.Generation.call(
                model='qwen-turbo',
                prompt=prompt,
                result_format='message',
                temperature=0.3,
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output.choices[0].message.content
            print(f"\n⚠️ LLM 状态码: {resp.status_code} - {resp.message}")
            time.sleep(2 ** i)
        except Exception as e:
            print(f"\n⚠️ 异常: {e}, 重试 {i + 1}/{max_retry}")
            time.sleep(2 ** i)
    return ""


def parse_json(text: str) -> Dict:
    """解析 LLM 返回的 JSON，容错"""
    if not text:
        return {}
    text = text.strip()
    # 去 markdown 包裹
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 提取 {...}
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {}


def load_products() -> List[Dict]:
    """自动找商品数据文件"""
    candidates = [
        ROOT / "data" / "products.json",
        ROOT / "data" / "items.json",
        ROOT / "data" / "products.jsonl",
        ROOT / "data" / "items.jsonl",
    ]
    for path in candidates:
        if path.exists():
            print(f"📂 加载商品数据: {path}")
            if path.suffix == ".jsonl":
                with open(path, encoding="utf-8") as f:
                    return [json.loads(line) for line in f if line.strip()]
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                if isinstance(data, dict):
                    return data.get("products") or data.get("items") or []
    return []


def get_field(product: Dict, *keys: str, default: str = "") -> str:
    """容错取字段（多个候选 key）"""
    for k in keys:
        if k in product and product[k]:
            return str(product[k])
    return default


def main():
    print("🚀 L1: LLM 抽取商品属性")
    print("=" * 50)

    if not dashscope.api_key:
        print("❌ 没有 DASHSCOPE_API_KEY，检查 .env 文件")
        return

    products = load_products()
    if not products:
        print("❌ 没找到商品数据文件")
        print("💡 检查 data/ 目录下是否有 products.json / items.json")
        return

    print(f"📦 商品总数: {len(products)}")
    print(f"📋 字段列表: {list(products[0].keys())}")
    print(f"📋 第一条示例:")
    print(json.dumps(products[0], ensure_ascii=False, indent=2)[:400])
    print("=" * 50)

    # 加载缓存
    cache: Dict = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"📌 已缓存 {len(cache)} 条，自动跳过")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    success, failed = 0, 0

    for product in tqdm(products, desc="LLM 抽取中"):
        pid = get_field(product, "id", "product_id", "sku", "item_id")
        if not pid:
            continue
        if pid in cache:
            continue

        name = get_field(product, "name", "title", "product_name")
        desc = get_field(product, "description", "desc", "detail", "summary", default=name)
        cat = get_field(product, "category", "cat", "category_name")
        brand = get_field(product, "brand", "brand_name")

        prompt = EXTRACT_PROMPT.format(
            name=name, description=desc, category=cat, brand=brand
        )
        result = call_llm(prompt)
        attrs = parse_json(result)

        if attrs:
            cache[pid] = {"name": name, "attributes": attrs}
            success += 1
            # 实时落盘，防中断
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        else:
            failed += 1
            print(f"\n⚠️ {pid} 抽取失败，原始返回: {result[:100]}")

    # 统计
    total_attrs = 0
    for item in cache.values():
        for v in item["attributes"].values():
            if isinstance(v, list):
                total_attrs += len(v)

    print("\n" + "=" * 50)
    print("✅ L1 抽取完成")
    print("=" * 50)
    print(f"📊 本次新增: {success} | 失败: {failed}")
    print(f"📊 累计商品: {len(cache)}")
    print(f"📊 总属性边: {total_attrs}")
    print(f"📊 平均每商品: {total_attrs / max(len(cache), 1):.1f} 个属性")
    print(f"📁 结果文件: {OUTPUT_PATH}")
    print("\n👉 下一步: 用这些属性重建 KG")


if __name__ == "__main__":
    main()
