"""统一启动入口"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("""
🛒 智能导购 Agent 启动

用法:
  python run.py enrich    # 数据增强（首次必跑）
  python run.py build     # 构建向量索引（首次必跑）
  python run.py eval      # 评估检索效果
  python run.py ui        # 启动 Web UI
  python run.py cli       # 命令行模式
""")
        return
    
    cmd = sys.argv[1]
    
    if cmd == "enrich":
        from scripts.enrich_data import enrich
        enrich()
    elif cmd == "build":
        from scripts.build_index import build
        build()
    elif cmd == "eval":
        from scripts.evaluate import evaluate
        evaluate()
    elif cmd == "ui":
        # 直接执行，避免重复初始化
        os.system(f"{sys.executable} ui/gradio_app.py")

    elif cmd == "cli":
        cli_chat()
    else:
        print(f"❌ 未知命令: {cmd}")

def cli_chat():
    """命令行版本（无 Gradio）"""
    import json, chromadb
    from langchain_community.chat_models import ChatTongyi
    from config import Config
    from core.kg import get_kg
    from core.retriever import HybridRetriever
    from agent.graph import build_graph
    
    with open(Config.DATA_FILE, encoding="utf-8") as f:
        products = json.load(f)
    col = chromadb.PersistentClient(path=Config.DB_PATH).get_collection("products")
    
    retriever = HybridRetriever(products, col, get_kg())
    llm_turbo = ChatTongyi(model_name=Config.LLM_TURBO, temperature=0)
    vlm_max = ChatTongyi(model_name=Config.LLM_VL_MAX, temperature=0.3, streaming=True)
    
    # === L5: 注入 LLM 重排器 ===
    if Config.USE_LLM_RERANK:
        from core.llm_reranker import LLMReranker
        retriever.llm_reranker = LLMReranker(llm_turbo, max_candidates=Config.LLM_RERANK_TOPN)
        print(f"✅ LLM 重排器已启用 (TopN={Config.LLM_RERANK_TOPN})")
    
    agent = build_graph(llm_turbo, vlm_max, retriever, use_react=True)


    
    print("=" * 50)
    print("🛒 智能导购小美 (CLI)")
    print("输入 'q' 退出")
    print("=" * 50)
    
    user_id = "cli_user"
    while True:
        print("\n" + "-" * 40)
        text = input("👤 你: ").strip()
        if text.lower() in ['q','quit','exit']: break
        if not text: continue
        
        img_path = input("🖼️ 图片路径 (回车跳过): ").strip()
        if img_path and not os.path.exists(img_path):
            print(f"⚠️ 路径不存在: {img_path}")
            img_path = None
        
        state = {
            "user_id": user_id, "user_text": text,
            "user_image_path": img_path or None,
            "dsl": None, "candidates": [], "profile_ctx": "",
            "final_answer": "", "feedback_log_id": None
        }
        
        print("\n🤖 小美: ", end="")
        try:
            agent.invoke(state)
        except Exception as e:
            print(f"\n❌ {e}")

if __name__ == "__main__":
    main()
