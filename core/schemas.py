"""结构化 DSL"""
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from enum import Enum

class IntentType(str, Enum):
    SEARCH = "search"
    COMPARE = "compare"
    CONSULT = "consult"
    ADD_CART = "add_cart"
    AFTER_SALE = "after_sale"
    CHITCHAT = "chitchat"
    REJECT = "reject"

class QueryDSL(BaseModel):
    intent: IntentType = IntentType.SEARCH
    query_text: str = ""
    use_image: bool = False
    
    brand: Optional[str] = None
    category: Optional[str] = None
    color: Optional[List[str]] = None
    size: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    
    scene: Optional[str] = None
    style: Optional[List[str]] = None
    
    sort_by: Literal["relevance","price_asc","price_desc","sales"] = "relevance"
    
    blocked: bool = False
    block_reason: Optional[str] = None
    
    def to_chroma_where(self):
        """转 ChromaDB 过滤器"""
        conds = []
        if self.brand: conds.append({"brand": self.brand})
        if self.category: conds.append({"category": self.category})
        if self.price_min is not None:
            conds.append({"price": {"$gte": float(self.price_min)}})
        if self.price_max is not None:
            conds.append({"price": {"$lte": float(self.price_max)}})
        # vec_type 是必要的，避免图文向量混淆
        conds.append({"vec_type": "text"})
        
        if len(conds) == 1: return conds[0]
        return {"$and": conds}
    
    def to_chroma_where_image(self):
        conds = [{"vec_type": "image"}]
        if self.brand: conds.append({"brand": self.brand})
        if self.category: conds.append({"category": self.category})
        if self.price_min is not None:
            conds.append({"price": {"$gte": float(self.price_min)}})
        if self.price_max is not None:
            conds.append({"price": {"$lte": float(self.price_max)}})
        return {"$and": conds} if len(conds) > 1 else conds[0]
