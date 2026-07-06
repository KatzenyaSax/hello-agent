"""
ReAct (Reasoning + Acting) Agent 实现
基于 Thought → Action → Observation 循环的推理-行动模式

继承自 Agent 抽象类（非 SimpleAgent），独立实现完整的 ReAct 循环。
"""
import re
from typing import Dict, Any, Optional, List
from core.agent import Agent
from core.llm import LLM
from core.config import Config
from core.message import Message
from tool.toolRegistry import ToolRegistry
from agent.prompts.reActPrompt import REACT_PROMPT


class ReActAgent(Agent):
    """ReAct 模式 Agent。

    通过 Thought → Action → Observation 循环进行推理和工具调用。
    run() 始终返回 str；is_stream=True 时内部走流式路径，
    边输出 chunk 边收集，对外签名不变。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional[ToolRegistry] = None,
        max_iterations: int = 5,
        is_stream: bool = False,
        verbose: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations
        self.is_stream = is_stream
        self.verbose = verbose
        self._execution_trace: List[str] = []   # 本轮执行轨迹

        # 统计工具数量
        tool_count = 0
        if tool_registry:
            desc = tool_registry.get_tools_description()
            if desc and desc != "暂无可用工具":
                tool_count = len([l for l in desc.split("\n") if l.strip().startswith("-")])

        print(
            f"✅ {name} (ReAct) 初始化完成 | "
            f"流式: {'开' if is_stream else '关'} | "
            f"最大迭代: {max_iterations} | "
            f"工具: {tool_count}个"
        )

    # ================================================================
    # 公共入口
    # ================================================================

    def run(self, input_text: str, **kwargs) -> str:
        """运行 ReAct Agent。

        is_stream=False → 批量模式，完整接收每轮响应后解析
        is_stream=True  → 流式模式，实时打印 chunk，行为对调用方透明
        """
        self._execution_trace = []

        if self.is_stream:
            return self._run_stream(input_text, **kwargs)
        else:
            return self._run_batch(input_text, **kwargs)

    # ================================================================
    # 批量模式
    # ================================================================

    def _run_batch(self, input_text: str, **kwargs) -> str:
        """批量模式：每轮用 llm.think() 获取完整响应。"""
        messages = self._build_initial_messages(input_text)

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- ReAct 第 {i + 1}/{self.max_iterations} 轮 ---")

            response = self.llm.think(messages, **kwargs)
            if response is None:
                return "❌ LLM 调用失败，请检查 API 配置。"

            result = self._process_response(response, messages)
            if result is not None:
                self._save_to_history(input_text, result)
                return result

        return self._force_final_answer(messages, input_text, **kwargs)

    # ================================================================
    # 流式模式
    # ================================================================

    def _run_stream(self, input_text: str, **kwargs) -> str:
        """流式模式：每轮用 llm.think_stream() 实时输出 chunk。

        等本轮全部收完后再解析（不做增量解析），保证解析正确性。
        """
        messages = self._build_initial_messages(input_text)

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n--- ReAct 第 {i + 1}/{self.max_iterations} 轮 ---")

            full_response = ""
            for chunk in self.llm.think_stream(messages, **kwargs):
                if chunk:
                    print(chunk, end="", flush=True)
                    full_response += chunk
            print()  # 本轮流式结束，换行

            if not full_response:
                return "❌ LLM 流式调用失败，请检查 API 配置。"

            result = self._process_response(full_response, messages)
            if result is not None:
                self._save_to_history(input_text, result)
                return result

        return self._force_final_answer(messages, input_text, **kwargs)

    # ================================================================
    # 消息构建
    # ================================================================

    def _build_initial_messages(self, input_text: str) -> List[Dict[str, str]]:
        """构建初始消息列表：system（含 ReAct 格式说明 + 工具列表）
        + 历史消息 + 当前用户消息。"""
        messages: List[Dict[str, str]] = []

        # 1. 系统消息
        tools_desc = (
            self.tool_registry.get_tools_description()
            if self.tool_registry
            else "暂无可用工具"
        )
        system_content = REACT_PROMPT.format(
            tools=tools_desc,
            question=input_text,
            history="（尚未执行任何操作）",
        )
        messages.append({"role": "system", "content": system_content})

        # 2. 历史消息（来自父类 self._history）
        for msg in self._history:
            messages.append(msg.to_dict())

        # 3. 当前用户消息
        messages.append({"role": "user", "content": input_text})

        return messages

    # ================================================================
    # 响应处理（批量 / 流式共用）
    # ================================================================

    def _process_response(
        self, response: str, messages: List[Dict]
    ) -> Optional[str]:
        """处理 LLM 的一轮完整响应。

        Returns:
            None  → 已执行工具并将 Observation 追加到 messages，继续循环
            str   → 这是最终答案，调用方应立刻返回
        """
        parsed = self._parse_react_output(response)

        # 打印 Thought
        if self.verbose and parsed["thought"]:
            t = parsed["thought"]
            print(f"🧠 Thought: {t[:120]}{'...' if len(t) > 120 else ''}")

        # —— 情况 1：检测到 Finish ——
        if parsed["finish"]:
            if self.verbose:
                f = parsed["finish"]
                print(f"🎯 最终答案: {f[:100]}{'...' if len(f) > 100 else ''}")
            return parsed["finish"]

        # —— 情况 2：检测到 Action ——
        if parsed["action"]:
            observation = self._execute_action(
                parsed["action"], parsed["action_input"] or ""
            )

            if self.verbose:
                print(f"🔧 Action: {parsed['action']}[{parsed['action_input']}]")
                o = observation
                print(f"👁️  Observation: {o[:150]}{'...' if len(o) > 150 else ''}")

            # 记录本轮执行轨迹
            trace = (
                f"Thought: {parsed['thought']}\n"
                f"Action: {parsed['action']}[{parsed['action_input']}]\n"
                f"Observation: {observation}"
            )
            self._execution_trace.append(trace)

            # 将本轮输出追加到对话上下文
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            return None

        # —— 情况 3：格式异常，提示 LLM 重新输出 ——
        if self.verbose:
            print("⚠️  响应格式异常（无 Action / Finish），提示 LLM 重试")
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": (
                "请严格遵循 ReAct 格式。每次回复必须包含 Thought 和以下两者之一：\n"
                "  - Action: 工具名[参数]  （调用工具）\n"
                "  - Finish[你的最终答案]   （结束任务）"
            ),
        })
        return None

    # ================================================================
    # 解析器
    # ================================================================

    def _parse_react_output(self, text: str) -> Dict[str, Optional[str]]:
        """解析 ReAct 格式的 LLM 输出。

        支持的格式:
          Thought: 推理内容...
          Action: tool_name[input]
          Finish[最终答案]

        Returns:
          {"thought": ..., "action": ..., "action_input": ..., "finish": ...}
          finish 与 action 互斥；未匹配到的字段为 None。
        """
        result: Dict[str, Optional[str]] = {
            "thought": None,
            "action": None,
            "action_input": None,
            "finish": None,
        }

        # Thought —— 从 "Thought:" 到下一个 "Action" / "Finish" 或文本末尾
        m = re.search(
            r"Thought:\s*(.+?)(?=\n\s*(?:Action|Finish)|\Z)",
            text, re.DOTALL | re.IGNORECASE,
        )
        if m:
            result["thought"] = m.group(1).strip()

        # Finish —— 优先检测，一旦命中就不再解析 Action
        m = re.search(r"Finish\s*\[(.+)\]", text, re.DOTALL | re.IGNORECASE)
        if m:
            result["finish"] = m.group(1).strip()
            return result

        # Action
        m = re.search(r"Action:\s*(\w+)\s*\[([^\]]+)\]", text, re.IGNORECASE)
        if m:
            result["action"] = m.group(1).strip()
            result["action_input"] = m.group(2).strip()

        return result

    # ================================================================
    # 工具执行
    # ================================================================

    def _execute_action(self, tool_name: str, tool_input: str) -> str:
        """执行工具，自动判断工具类型（函数 / Tool 对象）。"""
        if not self.tool_registry:
            return "❌ 错误: 未配置工具注册表"

        try:
            if self.tool_registry.is_function(tool_name):
                result = self.tool_registry.execute_function(tool_name, tool_input)
            elif self.tool_registry.is_tool(tool_name):
                result = self.tool_registry.execute_tool(
                    tool_name, {"input": tool_input}
                )
            else:
                return f"❌ 错误: 工具 '{tool_name}' 未在注册表中找到"

            return str(result) if result is not None else "（工具执行完成，无返回内容）"

        except Exception as e:
            return f"❌ 工具调用失败: {str(e)}"

    # ================================================================
    # 兜底 & 辅助
    # ================================================================

    def _force_final_answer(
        self, messages: List[Dict], input_text: str, **kwargs
    ) -> str:
        """达到最大迭代次数后，强制 LLM 输出最终答案。"""
        if self.verbose:
            print(f"⚠️  达到最大迭代次数 ({self.max_iterations})，强制要求最终答案")

        messages.append({
            "role": "user",
            "content": (
                "你已达到最大执行轮数。"
                "请立即用 Finish[你的答案] 格式给出最终回答，不要再调用工具。"
            ),
        })

        if self.is_stream:
            full = ""
            for chunk in self.llm.think_stream(messages, **kwargs):
                if chunk:
                    print(chunk, end="", flush=True)
                    full += chunk
            print()
        else:
            full = self.llm.think(messages, **kwargs)

        if full:
            parsed = self._parse_react_output(full)
            if parsed["finish"]:
                self._save_to_history(input_text, parsed["finish"])
                return parsed["finish"]

        # 最后兜底
        result = full or "❌ 无法获取有效回答"
        self._save_to_history(input_text, result)
        return result

    def _save_to_history(self, input_text: str, response: str):
        """保存对话到 Agent 历史。"""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))

    # ================================================================
    # 便利方法
    # ================================================================

    def get_execution_trace(self) -> List[str]:
        """返回本轮完整的执行轨迹（每轮一条 Thought/Action/Observation）。"""
        return self._execution_trace.copy()
