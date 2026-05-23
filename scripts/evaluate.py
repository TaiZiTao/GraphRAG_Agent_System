"""检索效果评估"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import chromadb
from langchain_community.chat_models import ChatTongyi
from config import Config
from core.kg import get_kg
from core.retriever import HybridRetriever
from core.query_parser import QueryParser

# 业务定制评估集
EVAL_SET = [
    {"query":"便宜的白色运动鞋", "expected":["p001"]},
    {"query":"适合大理旅游穿的连衣裙", "expected":["p003"]},
    {"query":"商务通勤的男士衬衫", "expected":["p005"]},
]

def main():
    with open(Config.DATA_FILE, encoding="utf-8") as f:
        products = json.load(f)
    col = chromadb.PersistentClient(path=Config.DB_PATH).get_collection("products")
    
    retriever = HybridRetriever(products, col, get_kg())
    llm = ChatTongyi(model_name=Config.LLM_TURBO, temperature=0)
    parser = QueryParser(llm)
    
    recall5 = mrr = 0
    for case in EVAL_SET:
        dsl = parser.parse(case["query"], has_image=False)
        results = retriever.retrieve(dsl, top_k=5)
        got = [r["id"] for r in results]
        print(f"\n[Q] {case['query']}\n[期望] {case['expected']}\n[实际] {got}")
        
        hit = any(e in got for e in case["expected"])
        recall5 += hit
        for i, gid in enumerate(got):
            if gid in case["expected"]:
                mrr += 1/(i+1); break
    
    n = len(EVAL_SET)
    print(f"\n{'='*40}")
    print(f"Recall@5: {recall5/n:.3f}  ({recall5}/{n})")
    print(f"MRR: {mrr/n:.3f}")

if __name__ == "__main__":
    main()
