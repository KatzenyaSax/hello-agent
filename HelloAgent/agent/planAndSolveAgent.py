"""
Plan-and-Solve Agent 实现
基于 拆分子任务 → 顺序执行子任务 的两阶段模式

继承自 Agent 抽象类，无工具调用，规划器分解问题，执行器逐步求解。
"""
import ast
import re
from typing import Optional, List, Dict
from core.agent import Agent
from core.llm import LLM
from core.config import Config
from core.message import Message
from agent.prompts.planAndSolvePrompt import DEFAULT_PLANNER_PROMPT, DEFAULT_EXECUTOR_PROMPT


class PlanAndSolveAgent(Agent):
    """Plan-and-Solve 模式 Agent。

    阶段 1（Plan）：将复杂问题分解为有序的子任务列表。
    阶段 2（Solve）：按顺序执行每个子任务，累积历史上下文。
    run() 始终返回 str；不支持流式输出。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        verbose: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.verbose = verbose
        self._step_results: List[Dict[str, str]] = []

        print(f"✅ {name} (PlanAndSolve) 初始化完成")

    # ================================================================
    # 公共入口
    # ================================================================

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Plan-and-Solve Agent。

        流程: 规划（分解子任务） → 执行（逐步求解） → 返回最终答案
        """
        self._step_results = []

        # 阶段 1: 规划
        plan = self._plan(input_text, **kwargs)
        if plan is None:
            return "❌ 规划阶段 LLM 调用失败，请检查 API 配置。"
        if not plan:
            return "❌ 规划阶段出错：未能生成有效的子任务计划。"

        if self.verbose:
            print(f"\n📋 计划 ({len(plan)} 步):")
            for j, step in enumerate(plan, 1):
                print(f"  {j}. {step}")

        # 保存用户输入和计划到 history
        self._save_to_history(input_text, self._format_plan(plan))

        # 阶段 2: 逐步执行
        history_text = ""

        for i, step in enumerate(plan):
            if self.verbose:
                print(f"\n--- 执行步骤 {i + 1}/{len(plan)} ---")

            result = self._execute_step(input_text, plan, history_text, step, **kwargs)
            if result is None:
                if self.verbose:
                    print(f"⚠️ 步骤 {i + 1} LLM 调用失败，返回已累积结果")
                break

            self._step_results.append({"step": step, "result": result})

            if self.verbose:
                print(f"✅ 步骤 {i + 1} 完成: {result[:120]}{'...' if len(result) > 120 else ''}")

            # 更新历史
            history_text += f"步骤 {i + 1}: {step}\n结果 {i + 1}: {result}\n\n"

            # 保存每步结果到 history
            self.add_message(Message(
                f"步骤 {i + 1}: {step}\n结果: {result}", "assistant"
            ))

        # 返回最后一步的结果
        if self._step_results:
            final = self._step_results[-1]["result"]
            if self.verbose:
                print(f"\n🏁 最终答案: {final[:120]}{'...' if len(final) > 120 else ''}")
            return final

        return "❌ 执行阶段未能产出任何结果。"

    # ================================================================
    # 阶段 1: 规划
    # ================================================================

    def _plan(self, question: str, **kwargs) -> Optional[List[str]]:
        """调用规划器 LLM，生成子任务列表。"""
        messages = [
            {"role": "system", "content": DEFAULT_PLANNER_PROMPT.format(question=question)},
            {"role": "user", "content": question},
        ]
        response = self.llm.think(messages, **kwargs)
        if response is None:
            return None
        return self._parse_plan(response)

    def _parse_plan(self, text: str) -> List[str]:
        """解析 LLM 输出的计划。

        优先使用 ast.literal_eval，失败后用正则提取引号内字符串兜底。
        """
        # 尝试定位 ```python ... ``` 代码块中的内容
        code_block = text
        m = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
        if m:
            code_block = m.group(1).strip()

        # 1) ast.literal_eval
        try:
            result = ast.literal_eval(code_block)
            if isinstance(result, list) and all(isinstance(s, str) for s in result):
                return result
        except (ValueError, SyntaxError):
            pass

        # 2) 正则兜底：匹配所有被双引号或单引号包裹的字符串
        steps = re.findall(r"""["']([^"']+)["']""", code_block)
        if steps:
            return steps

        return []

    # ================================================================
    # 阶段 2: 执行
    # ================================================================

    def _execute_step(
        self,
        question: str,
        plan: List[str],
        history: str,
        current_step: str,
        **kwargs,
    ) -> Optional[str]:
        """执行单个子任务，返回该步骤的结果。"""
        plan_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(plan))
        history_text = history if history else "（尚无已完成的步骤）"

        messages = [
            {
                "role": "system",
                "content": DEFAULT_EXECUTOR_PROMPT.format(
                    question=question,
                    plan=plan_text,
                    history=history_text,
                    current_step=current_step,
                ),
            },
            {"role": "user", "content": f"请执行当前步骤: {current_step}"},
        ]
        return self.llm.think(messages, **kwargs)

    # ================================================================
    # 辅助方法
    # ================================================================

    def _format_plan(self, plan: List[str]) -> str:
        """格式化计划为可读字符串，用于存储到 history。"""
        lines = [f"{i + 1}. {s}" for i, s in enumerate(plan)]
        return "📋 执行计划:\n" + "\n".join(lines)

    def _save_to_history(self, input_text: str, response: str):
        """保存用户输入和回答到 Agent 历史。"""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

    # ================================================================
    # 便利方法
    # ================================================================

    def get_step_results(self) -> List[Dict[str, str]]:
        """返回本轮每个步骤的名称和结果。"""
        return self._step_results.copy()
