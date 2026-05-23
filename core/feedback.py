"""反馈埋点 → 离线微调闭环"""
import sqlite3
import json
import uuid
from datetime import datetime
from typing import List, Optional
from config import Config

class FeedbackLogger:
    _db = None
    
    @classmethod
    def _get_db(cls):
        if cls._db is None:
            cls._db = sqlite3.connect(Config.FEEDBACK_DB_PATH, check_same_thread=False)
            cls._db.execute("""CREATE TABLE IF NOT EXISTS rec_log(
                log_id TEXT PRIMARY KEY, user_id TEXT,
                query_text TEXT, dsl_json TEXT,
                recommended_ids TEXT, response TEXT,
                clicked_ids TEXT, feedback TEXT, ts TEXT
            )""")
            cls._db.commit()
        return cls._db
    
    @classmethod
    def log_recommendation(cls, user_id, dsl, candidates, response) -> str:
        log_id = str(uuid.uuid4())
        db = cls._get_db()
        db.execute("INSERT INTO rec_log VALUES (?,?,?,?,?,?,?,?,?)", (
            log_id, user_id, dsl.query_text, dsl.model_dump_json(),
            json.dumps([p["id"] for p in candidates]),
            (response or "")[:1000], "[]", "", datetime.now().isoformat()
        ))
        db.commit()
        return log_id
    
    @classmethod
    def record_feedback(cls, log_id: str, feedback: str, clicked_ids: Optional[List[str]]=None):
        db = cls._get_db()
        db.execute("UPDATE rec_log SET feedback=?, clicked_ids=? WHERE log_id=?",
                   (feedback, json.dumps(clicked_ids or []), log_id))
        db.commit()
    
    @classmethod
    def export_training_data(cls, products_map, output_path="rerank_train.jsonl"):
        """导出用于 Reranker 微调的训练数据"""
        db = cls._get_db()
        rows = db.execute("""SELECT query_text, recommended_ids, clicked_ids, feedback 
                              FROM rec_log WHERE feedback != ''""").fetchall()
        
        n = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for query, rec_ids, clk_ids, fb in rows:
                rec_ids = json.loads(rec_ids)
                clk_ids = json.loads(clk_ids or "[]")
                pos, neg = [], []
                for rid in rec_ids:
                    if rid not in products_map: continue
                    doc = products_map[rid].get("dense_caption","")
                    if rid in clk_ids: pos.append(doc)
                    elif fb in ("换一批","dislike"): neg.append(doc)
                if pos or neg:
                    f.write(json.dumps({"query":query,"pos":pos,"neg":neg}, 
                                       ensure_ascii=False) + "\n")
                    n += 1
        print(f"✅ 导出 {n} 条训练样本 → {output_path}")
