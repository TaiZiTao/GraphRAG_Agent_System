"""安全门卫"""
import re
from typing import Tuple, Optional

class Guardrails:
    BLOCKED_PLATFORMS = ["拼多多", "京东", "亚马逊", "唯品会"]
    SENSITIVE_PATTERNS = [
        re.compile(r"(辱骂|歧视|色情|暴力|赌博|毒品)"),
    ]
    
    @classmethod
    def check(cls, text: str) -> Tuple[bool, Optional[str]]:
        if not text: return True, None
        for p in cls.BLOCKED_PLATFORMS:
            if p in text:
                return False, f"我们不能推荐第三方平台 ({p}) 商品"
        for pat in cls.SENSITIVE_PATTERNS:
            if pat.search(text):
                return False, "您的输入包含不当内容"
        return True, None
