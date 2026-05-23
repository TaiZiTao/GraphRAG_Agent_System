"""CLIP 编码器（单例懒加载）"""
import base64
import torch
from PIL import Image
from config import Config

_model = None
_proc = None
_device = None

def _init():
    global _model, _proc, _device
    if _model is not None: return
    
    print("📦 加载 Chinese-CLIP（本地离线）...")
    import os
    from pathlib import Path
    from transformers import ChineseCLIPProcessor, ChineseCLIPModel
    
    # 强制离线，不联网检查
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    # 直接指向本地路径
    ROOT = Path(__file__).resolve().parent.parent
    model_dir = ROOT / "models" / "AI-ModelScope" / "chinese-clip-vit-base-patch16"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"❌ Chinese-CLIP 不存在: {model_dir}")
    
    model_dir = str(model_dir)
    _model = ChineseCLIPModel.from_pretrained(model_dir, local_files_only=True)
    _proc = ChineseCLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    _model.eval()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(_device)
    print(f"✅ CLIP 加载完成 (device={_device})")


def encode_image(image):
    _init()
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    inputs = _proc(images=image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.vision_model(**inputs)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        feat = _model.visual_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().float().numpy().tolist()

def encode_text(text):
    _init()
    inputs = _proc(text=text, return_tensors="pt", padding=True, truncation=True, max_length=52)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        out = _model.text_model(**inputs)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        feat = _model.text_projection(pooled)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().float().numpy().tolist()

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        b = base64.b64encode(f.read()).decode("utf-8")
    ext = image_path.split(".")[-1].lower()
    mime_map = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}
    mime = mime_map.get(ext, "image/jpeg")
    return f"data:{mime};base64,{b}"
