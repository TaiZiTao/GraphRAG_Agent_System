"""
VLM 图片打标 → 追加到 products.json
=========================================================
对齐用户原有格式:
  {id, price, image_path, description}

特性:
- 自动从最大 ID 续号 (010 → 011, 012, ...)
- VLM 生成电商标题风格 description (关键词密集，对 L1 友好)
- VLM 估算价格 (合理区间)
- 跨目录智能采样 (避免同款连续)
- 实时落盘 + 断点续跑
"""
import os
import sys
import json
import time
import random
import glob
from pathlib import Path
from http import HTTPStatus
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from dotenv import load_dotenv
from tqdm import tqdm
import dashscope
from dashscope import MultiModalConversation

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

# ============================================
# 配置
# ============================================
IMAGE_DIR = ROOT / "data" / "tianchi_fm_img2_1"
OUTPUT_PATH = ROOT / "data" / "products.json"
PROGRESS_PATH = ROOT / "data" / ".vlm_progress.json"

MAX_NEW_IMAGES = 1000         # 本次新增多少张（试水 20，全量改 1000）
CONCURRENCY = 5             # 并发，限流就降到 3
MODEL = "qwen-vl-max-latest"
SEED = 42

SHUFFLE_ACROSS_DIRS = True  # 跨目录打散，避免同款连续
MAX_PER_DIR = 920            # 每个子目录最多取 N 张

# ============================================
# Prompt（输出对齐淘宝标题风格）
# ============================================
CAPTION_PROMPT = """你是淘宝/京东资深商品标题撰写人。仔细观察这张服饰商品图，输出严格 JSON（不要 markdown、不要任何额外文字）：

{
  "description": "电商商品标题（35-60字，关键词密集，仿淘宝京东风格）",
  "price": 估算价格（数字，元）
}

【description 撰写规范】
模仿这种风格："梵先生Mr.Vanset2026新款夏季凉感短袖t恤男潮牌宽松嘻哈潮流复古"

必须包含的关键词类型（按出现频率排序）：
1. 品牌词：能看到 logo 写出，看不到编一个有英文感的潮牌名（如：MOZE/STKR/Avenir/NUOVA）
2. 品类词：T恤/衬衫/连衣裙/牛仔裤/休闲裤/板鞋/运动鞋/外套
3. 性别：男/女/男女款
4. 季节年份：2026新款/夏季/春秋
5. 风格：潮牌/美式/港风/法式/复古/嘻哈/学院/通勤
6. 工艺/版型：宽松/修身/oversize/扎染/拼接/印花/纯色
7. 场景：休闲/通勤/约会/度假/日常

【price 估算规范】
- T恤/短袖：60-200
- 衬衫/POLO：100-400
- 牛仔裤/休闲裤：150-500
- 连衣裙：150-600
- 外套：300-1500
- 板鞋/运动鞋：100-500
- 看材质和精致度上下浮动

直接输出 JSON，不要解释。
"""


# ============================================
# 工具
# ============================================
output_lock = Lock()


def load_progress():
    if PROGRESS_PATH.exists():
        try:
            return set(json.load(open(PROGRESS_PATH))["done"])
        except Exception:
            return set()
    return set()


def save_progress(done_set):
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w") as f:
        json.dump({"done": list(done_set)}, f)


def parse_json(text):
    """容错解析 JSON"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return None


def collect_images_smart(image_dir, max_images, max_per_dir, shuffle, exclude_paths):
    """跨目录均匀采样，避免同款连续"""
    dir_to_imgs = {}
    for ext in ["jpg", "jpeg", "png", "webp", "JPG", "JPEG", "PNG"]:
        for img in glob.glob(str(image_dir / f"**/*.{ext}"), recursive=True):
            if img in exclude_paths:
                continue
            parent = str(Path(img).parent)
            dir_to_imgs.setdefault(parent, []).append(img)
    
    if not dir_to_imgs:
        return []
    
    print(f"📂 发现 {len(dir_to_imgs)} 个子目录（已排除已处理）")
    for d, imgs in list(dir_to_imgs.items())[:5]:
        try:
            rel = Path(d).relative_to(image_dir)
        except ValueError:
            rel = Path(d).name
        print(f"   {rel}: {len(imgs)} 张")
    if len(dir_to_imgs) > 5:
        print(f"   ... 还有 {len(dir_to_imgs) - 5} 个目录")
    
    rng = random.Random(SEED)
    
    # 每个目录内打散 + 截断
    for d in dir_to_imgs:
        if shuffle:
            rng.shuffle(dir_to_imgs[d])
        dir_to_imgs[d] = dir_to_imgs[d][:max_per_dir]
    
    # 跨目录轮询取
    selected = []
    dirs = list(dir_to_imgs.keys())
    if shuffle:
        rng.shuffle(dirs)
    
    idx = 0
    while len(selected) < max_images:
        any_added = False
        for d in dirs:
            if idx < len(dir_to_imgs[d]):
                selected.append(dir_to_imgs[d][idx])
                any_added = True
                if len(selected) >= max_images:
                    break
        if not any_added:
            break
        idx += 1
    
    return selected


def caption_one(image_path):
    """单张打标"""
    try:
        messages = [{
            "role": "user",
            "content": [
                {"image": f"file://{os.path.abspath(image_path)}"},
                {"text": CAPTION_PROMPT},
            ],
        }]
        
        resp = MultiModalConversation.call(model=MODEL, messages=messages)
        
        if resp.status_code != HTTPStatus.OK:
            return image_path, None, f"API {resp.code}: {resp.message}"
        
        content = resp.output.choices[0].message.content
        if isinstance(content, list):
            text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
        else:
            text = str(content)
        
        result = parse_json(text)
        if not result:
            return image_path, None, f"JSON 解析失败: {text[:80]}"
        
        if "description" not in result or "price" not in result:
            return image_path, None, f"字段缺失: {list(result.keys())}"
        
        return image_path, result, None
    except Exception as e:
        return image_path, None, f"{type(e).__name__}: {e}"


def get_next_id(products):
    """从已有 products 中找最大 ID，返回下一个"""
    max_id = 0
    for p in products:
        try:
            pid = int(str(p.get("id", "0")).lstrip("p"))
            max_id = max(max_id, pid)
        except (ValueError, TypeError):
            pass
    return max_id + 1


def main():
    print("🚀 VLM 图片打标 → products.json")
    print("=" * 60)
    
    if not dashscope.api_key:
        print("❌ 没有 DASHSCOPE_API_KEY，检查 .env 文件")
        return
    
    if not IMAGE_DIR.exists():
        print(f"❌ 图片目录不存在: {IMAGE_DIR}")
        return
    
    # 1. 加载已有 products
    products = []
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            products = json.load(f)
        print(f"📌 已有 products.json: {len(products)} 条")
    
    existing_paths = {p.get("image_path") for p in products if p.get("image_path")}
    progress_paths = load_progress()
    exclude_paths = existing_paths | progress_paths
    
    next_id = get_next_id(products)
    print(f"   下一个 ID: {next_id:03d}\n")
    
    # 2. 智能采样
    print(f"📦 采样图片（max={MAX_NEW_IMAGES}, 跨目录打散）...")
    image_files = collect_images_smart(
        IMAGE_DIR, MAX_NEW_IMAGES, MAX_PER_DIR, SHUFFLE_ACROSS_DIRS, exclude_paths
    )
    print(f"✅ 选中 {len(image_files)} 张待处理\n")
    
    if not image_files:
        print("✅ 没有新图片需要处理")
        return
    
    # 3. 成本估算
    cost = len(image_files) * 0.02
    print(f"💰 预估成本: ¥{cost:.2f}")
    print(f"⚙️  并发: {CONCURRENCY} | 模型: {MODEL}\n")
    
    # 4. 并发打标
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    success, failed = 0, 0
    t_start = time.time()
    done_set = set(progress_paths)
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {executor.submit(caption_one, img): img for img in image_files}
        
        pbar = tqdm(as_completed(futures), total=len(futures), desc="VLM 打标")
        for future in pbar:
            img_path, result, err = future.result()
            
            if err or not result:
                failed += 1
                pbar.write(f"❌ {Path(img_path).name[:25]:25s} | {err}")
                continue
            
            with output_lock:
                # 严格对齐你的格式: {id, price, image_path, description}
                record = {
                    "id": f"{next_id + success:03d}",
                    "price": float(result.get("price", 0)),
                    "image_path": img_path,
                    "description": result.get("description", ""),
                }
                products.append(record)
                done_set.add(img_path)
                success += 1
                
                # 每 5 条落盘
                if success % 5 == 0:
                    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                        json.dump(products, f, ensure_ascii=False, indent=2)
                    save_progress(done_set)
                
                pbar.set_postfix({
                    "ok": success,
                    "fail": failed,
                    "id": record["id"],
                    "desc": record["description"][:12],
                })
    
    # 5. 最终保存
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=2)
    save_progress(done_set)
    
    elapsed = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print(f"✅ 完成 | 成功 {success} | 失败 {failed}")
    print(f"⏱️  耗时 {elapsed:.1f}min")
    print(f"💰 实际成本: ¥{success * 0.02:.2f}")
    print(f"📁 输出: {OUTPUT_PATH}")
    print(f"📊 累计商品: {len(products)} 条")
    print(f"\n👉 下一步: python scripts/l1_extract_attrs.py")


if __name__ == "__main__":
    main()
