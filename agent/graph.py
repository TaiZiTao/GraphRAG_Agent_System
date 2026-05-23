"""LangGraph 编排（ReAct 增强版）"""
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Optional
from PIL import Image

from core.schemas import QueryDSL, IntentType
from core.query_parser import QueryParser
from core.memory import TieredMemory
from core.feedback import FeedbackLogger
from core.encoders import image_to_base64
from agent.prompts import GENERATION_PROMPT, REJECT_PROMPT
from agent.profile_integrator import ProfileIntegrator
from agent.tools import ToolRegistry              # 🆕
from agent.react_agent import ReactAgent          # 🆕


class AgentState(TypedDict):
    user_id: str
    user_text: str
    user_image_path: Optional[str]
    dsl: Optional[QueryDSL]
    candidates: List[dict]
    profile_ctx: str
    final_answer: str
    feedback_log_id: Optional[str]
    react_trace: List[dict]                       # 🆕


def build_graph(llm_turbo, vlm_max, retriever, use_react: bool = True):
    parser = QueryParser(llm_turbo)
    integrator = ProfileIntegrator(llm_turbo)
    
    # 🆕 ReAct 组件（消融时可关闭 use_react）
    tool_registry = ToolRegistry(retriever) if use_react else None
    react_agent = ReactAgent(llm_turbo, tool_registry, max_iter=4) if use_react else None

    def node_parse(state):
        integrator.flush_pending_logs(state["user_id"])
        dsl = parser.parse(state["user_text"], bool(state.get("user_image_path")))
        print(f"\n[🎯 DSL] {dsl.model_dump_json(exclude_none=True, exclude_defaults=True)}")
        return {"dsl": dsl}

    def node_memory_load(state):
        sid = state["user_id"]
        dsl = state.get("dsl")
        if dsl and not dsl.blocked and dsl.intent != IntentType.CHITCHAT:
            integrator.enrich_dsl(state["dsl"], state["user_id"], query=state["user_text"])
            integrator.sync_merge_from_dsl(sid, dsl, state["user_text"])
        
        mem = TieredMemory(sid)
        ctx_old = mem.render_profile_context()
        ctx_new = integrator.render_context(sid)
        ctx = "\n".join(c for c in [ctx_old, ctx_new] if c)
        if ctx: print(f"[👤 画像]\n{ctx}")
        return {"profile_ctx": ctx, "dsl": dsl}

    # ============================================================
    # 🆕 ReAct 检索节点（替代原 node_retrieve）
    # ============================================================
    def node_react_retrieve(state):
        dsl = state["dsl"]
        image = None
        if state.get("user_image_path"):
            try: image = Image.open(state["user_image_path"]).convert("RGB")
            except: pass

        # Baseline 兜底检索（确保有结果）
        baseline = retriever.retrieve(dsl, image=image, top_k=5)
        print(f"[🔍 Baseline] 初始检索 {len(baseline)} 个候选")

        # 闲聊或关闭 ReAct → 直接用 baseline
        if dsl.intent == IntentType.CHITCHAT or not use_react:
            return {"candidates": baseline, "react_trace": []}

        # ReAct 决策是否优化
        result = react_agent.run(
            query=state["user_text"],
            dsl=dsl,
            baseline_results=baseline,
            profile_ctx=state.get("profile_ctx", ""),
            session_id=state["user_id"],
            image=image,
        )
        candidates = result["candidates"] or baseline   # 双兜底
        print(f"[🎯 ReAct] {len(result['trace'])} 步推理 → 最终 {len(candidates)} 个商品")
        return {"candidates": candidates, "react_trace": result["trace"]}

    def node_generate(state):
        candidates = state["candidates"]
        if not candidates:
            return {"final_answer": "抱歉，没找到符合您条件的商品。要不要放宽一下条件试试？😊"}

        def fmt_explain(p):
            ex = p.get("explain") or {}
            scenes = ex.get("scenes", [])
            attrs = ex.get("matched_attrs", [])
            if scenes and attrs:
                return f"🌐 KG路径: 场景【{'/'.join(scenes)}】→ 属性[{', '.join(attrs)}]"
            elif attrs:
                return f"🌐 商品属性: [{', '.join(attrs[:5])}]"
            return "🌐 KG路径: （未命中场景）"

        candidates_text = "\n\n".join([
            f"""【{i+1}】ID={p['id']} | {p.get('name', p.get('description','-')[:30])}
    - 价格: ¥{p['price']} | 品牌: {p.get('brand','-')} | 类目: {p.get('category','-')}
    - 描述: {p.get('dense_caption') or p.get('description','')}
    - {fmt_explain(p)}"""
            for i, p in enumerate(candidates)
        ])

        text_prompt = GENERATION_PROMPT.format(
            profile_ctx=state.get("profile_ctx", ""),
            user_text=state["user_text"],
            dsl_json=state["dsl"].model_dump_json(exclude_none=True, exclude_defaults=True),
            candidates_text=candidates_text
        )

        content = []
        if state.get("user_image_path"):
            try:
                content.append({"image": image_to_base64(state["user_image_path"])})
            except Exception as e:
                print(f"[图片编码失败] {e}")
        content.append({"text": text_prompt})

        if len(content) == 1 and "text" in content[0]:
            message_content = content[0]["text"]
        else:
            message_content = content

        print("\n💬 小美:\n", end="", flush=True)
        full = ""
        try:
            for chunk in vlm_max.stream([HumanMessage(content=message_content)]):
                txt = chunk.content if isinstance(chunk.content, str) else \
                      "".join(c.get("text","") for c in chunk.content if isinstance(c, dict))
                print(txt, end="", flush=True)
                full += txt
            print()
        except Exception as e:
            print(f"\n[生成失败] {e}")
            full = "抱歉，生成回复时出错，请稍后再试。"

        log_id = FeedbackLogger.log_recommendation(
            state["user_id"], state["dsl"], candidates, full
        )
        return {"final_answer": full, "feedback_log_id": log_id}

    def node_memory_save(state):
        mem = TieredMemory(state["user_id"])
        mem.add_message("user", state["user_text"])
        mem.add_message("assistant", state["final_answer"])
        if state.get("dsl"):
            mem.update_preference(state["dsl"])
        dsl = state.get("dsl")
        if dsl and not dsl.blocked and dsl.intent != IntentType.CHITCHAT:
            integrator.async_llm_enrich(state["user_id"], state["user_text"])
        return {}

    def node_reject(state):
        return {"final_answer": REJECT_PROMPT.format(reason=state["dsl"].block_reason)}

    def route_after_parse(state):
        if state["dsl"].blocked: return "reject"
        if state["dsl"].intent == IntentType.CHITCHAT:
            return "memory_load"
        return "memory_load"

    g = StateGraph(AgentState)
    g.add_node("parse", node_parse)
    g.add_node("memory_load", node_memory_load)
    g.add_node("react_retrieve", node_react_retrieve)   # 🆕 替换原 retrieve
    g.add_node("generate", node_generate)
    g.add_node("memory_save", node_memory_save)
    g.add_node("reject", node_reject)

    g.set_entry_point("parse")
    g.add_conditional_edges("parse", route_after_parse,
                            {"reject": "reject", "memory_load": "memory_load"})
    g.add_edge("memory_load", "react_retrieve")          # 🆕
    g.add_edge("react_retrieve", "generate")             # 🆕
    g.add_edge("generate", "memory_save")
    g.add_edge("memory_save", END)
    g.add_edge("reject", END)

    return g.compile()
