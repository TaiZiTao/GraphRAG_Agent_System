"""
画像存储层 - 抽象接口 + 多种实现
默认内存版，未来通过 REDIS_URL 环境变量自动切换
"""
import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ProfileStorage(ABC):
    """存储抽象接口"""
    
    @abstractmethod
    def get(self, session_id: str) -> Optional[Dict]: ...
    
    @abstractmethod
    def set(self, session_id: str, profile: Dict): ...
    
    @abstractmethod
    def delete(self, session_id: str): ...


class MemoryStorage(ProfileStorage):
    """内存版 - 开发默认，重启清空"""
    
    def __init__(self):
        self._store: Dict[str, Dict] = {}
    
    def get(self, session_id: str) -> Optional[Dict]:
        return self._store.get(session_id)
    
    def set(self, session_id: str, profile: Dict):
        self._store[session_id] = profile
    
    def delete(self, session_id: str):
        self._store.pop(session_id, None)


class RedisStorage(ProfileStorage):
    """Redis 版 - 生产可选，需要 redis 包"""
    
    def __init__(self, url: str, ttl: int = 3600 * 24):
        import redis
        self.client = redis.from_url(url, decode_responses=True)
        self.ttl = ttl
        self.client.ping()  # 启动时探活
    
    def get(self, session_id: str) -> Optional[Dict]:
        raw = self.client.get(f"profile:{session_id}")
        return json.loads(raw) if raw else None
    
    def set(self, session_id: str, profile: Dict):
        self.client.setex(
            f"profile:{session_id}",
            self.ttl,
            json.dumps(profile, ensure_ascii=False)
        )
    
    def delete(self, session_id: str):
        self.client.delete(f"profile:{session_id}")


def get_storage() -> ProfileStorage:
    """根据环境自动选择存储"""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            print(f"✅ [Profile] 使用 Redis 存储: {redis_url}")
            return RedisStorage(redis_url)
        except Exception as e:
            print(f"⚠️ [Profile] Redis 不可用 ({e})，降级到内存")
    return MemoryStorage()
