"""ReAct Agent 主循环"""
import json
import re


REACT_PROMPT = """你是导购小美的检索决策大脑，使用 ReAct 模式优化商品检索。

【用户需求】{query}
【初始 DSL】{dsl_json}
【用户画像摘要】{profile_summary}

【你的任务】
系统已基于初步理解做了 baseline 检索（见下方）。判断 baseline 是否够好：
- 如果合适 → 直接 finalize
- 如果偏差大 → 用 search_products 重搜
- 如果只是预算/颜色不对 → 用 filter_last_results
- 不确定用户偏好 → 先 get_user_profile

{tools}

【输出格式】严格按以下格式，每轮只输出 1 个 Thought + 1 个 Action：

Thought: <一句话思考>
Action: <工具名>
Action Input: <严格 JSON>

【硬规则】
- 最多 4 步，必须以 finalize 结束
- 不要自己编 Observation，等系统返回
- 简单 query 不要过度调用工具，1~2 步就 finalize

【Baseline 检索结果】
{baseline}

开始：
"""


class ReactAgent:
    def __init__(self, llm, tools, max_iter=4, verbose=True):
        self.llm = llm
        self.tools = tools
        self.max_iter = max_iter
        self.verbose = verbose

    def run(self, query, dsl, baseline_results, profile_ctx, session_id, image=None):
        self.tools.reset(session_id, image=image,
                         initial_results=baseline_results, profile_ctx=profile_ctx)
        baseline_text = self.tools._fmt(baseline_results)

        prompt = REACT_PROMPT.format(
            query=query,
            profile_summary=(profile_ctx[:200] if profile_ctx else "（无）"),
            dsl_json=dsl.model_dump_json(exclude_none=True, exclude_defaults=True),
            tools=self.tools.get_schemas(),
            baseline=baseline_text,
        )

        scratchpad = ""
        trace = []
        finalized = False

        for step in range(self.max_iter):
            try:
                resp = self.llm.invoke(prompt + "\n" + scratchpad).content
            except Exception as e:
                if self.verbose: print(f"[ReAct LLM 失败] {e}")
                break

            resp = resp.split("Observation:")[0].strip()  # 防 LLM 自答 Observation

            thought = self._extract(resp, r'Thought:\s*(.+?)(?=Action:|$)')
            action_name = self._extract(resp, r'Action:\s*(\w+)')
            action_input = self._parse_input(resp)

            if self.verbose:
                print(f"\n🤔 [ReAct 第 {step+1} 步]")
                if thought: print(f"   💭 {thought[:120]}")
                print(f"   🔧 {action_name}({json.dumps(action_input, ensure_ascii=False)[:120]})")

            trace.append({"step": step+1, "thought": thought,
                          "action": action_name, "input": action_input})

            if not action_name:
                if self.verbose: print("   ⚠️ 解析不出 Action，结束")
                break

            obs = self.tools.execute(action_name, action_input)
            if self.verbose: print(f"   👁️  {obs[:160]}")
            trace[-1]["observation"] = obs

            if action_name == "finalize":
                finalized = True
                break

            scratchpad += f"\n{resp}\nObservation: {obs}\n"

        if not finalized and self.verbose:
            print(f"   ⚠️  未显式 finalize，使用 last_results ({len(self.tools.last_results)} 个)")

        return {"candidates": self.tools.last_results, "trace": trace}

    def _extract(self, text, pattern):
        m = re.search(pattern, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _parse_input(self, text):
        m = re.search(r'Action Input:\s*(\{.*?\})\s*(?:\n|$)', text, re.DOTALL)
        if not m:
            m = re.search(r'Action Input:\s*(\{.*\})', text, re.DOTALL)
        if not m: return {}
        raw = m.group(1).strip()
        try: return json.loads(raw)
        except Exception:
            try:
                fixed = (raw.replace("'", '"')
                            .replace("None", "null")
                            .replace("True", "true")
                            .replace("False", "false"))
                return json.loads(fixed)
            except: return {}
