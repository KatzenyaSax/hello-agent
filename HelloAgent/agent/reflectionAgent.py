"""
Reflection Agent 实现
基于 生成 → 反思 → 优化 循环的自我改进模式

继承自 Agent 抽象类，无工具调用，通过内部反思循环优化答案质量。
"""
import re
from typing import Dict, Optional, List
from core.agent import Agent
from core.llm import LLM
from core.config import Config
from core.message import Message
from agent.prompts.reflectionPrompt import DEFAULT_PROMPTS


class ReflectionAgent(Agent):
    """Reflection 模式 Agent。

    通过 生成 → 反思 → 优化 循环进行自我改进。
    run() 始终返回 str；不支持流式输出。
    """

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_reflections: int = 2,
        auto_finish: bool = False,
        prompts: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.max_reflections = max_reflections
        self.auto_finish = auto_finish
        self.verbose = verbose

        # 合并提示词：用户传入的覆盖默认值
        self.prompts = {**DEFAULT_PROMPTS, **(prompts or {})}

        print(
            f"✅ {name} (Reflection) 初始化完成 | "
            f"最大反思轮数: {max_reflections} | "
            f"智能终止: {'开' if auto_finish else '关'}"
        )

    # ================================================================
    # 公共入口
    # ================================================================

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Reflection Agent。

        流程: 初始生成 → 循环(反思 → 优化) → 返回最终答案
        """
        # 阶段 1: 初始生成
        initial_answer = self._generate_initial(input_text, **kwargs)
        if initial_answer is None:
            return "❌ LLM 调用失败，请检查 API 配置。"

        if self.verbose:
            print(f"\n📝 初始答案: {initial_answer[:120]}{'...' if len(initial_answer) > 120 else ''}")

        # 保存用户输入 + 初始答案到 history
        self._save_to_history(input_text, initial_answer)

        current_answer = initial_answer

        # 阶段 2: 反思-优化循环
        for i in range(self.max_reflections):
            if self.verbose:
                print(f"\n--- Reflection 第 {i + 1}/{self.max_reflections} 轮 ---")

            # 反思
            feedback = self._reflect(input_text, current_answer, **kwargs)
            if feedback is None:
                if self.verbose:
                    print("⚠️ 反思阶段 LLM 调用失败，返回当前答案")
                return current_answer

            if self.verbose:
                print(f"🔍 反思反馈: {feedback[:120]}{'...' if len(feedback) > 120 else ''}")

            # 智能终止检测
            if self._check_no_improvement(feedback):
                if self.auto_finish:
                    if self.verbose:
                        print("✅ 模型判定无需改进，提前终止")
                    self.add_message(Message(feedback, "assistant"))
                    return current_answer
                # auto_finish=False: 忽略标记，继续执行 refine

            # 优化
            refined_answer = self._refine(input_text, current_answer, feedback, **kwargs)
            if refined_answer is None:
                if self.verbose:
                    print("⚠️ 优化阶段 LLM 调用失败，返回当前答案")
                return current_answer

            if self.verbose:
                print(f"✨ 优化结果: {refined_answer[:120]}{'...' if len(refined_answer) > 120 else ''}")

            # 保存本轮反思产物到 history
            self.add_message(Message(feedback, "assistant"))
            self.add_message(Message(refined_answer, "assistant"))

            current_answer = refined_answer

        return current_answer

    # ================================================================
    # 三个阶段
    # ================================================================

    def _generate_initial(self, task: str, **kwargs) -> Optional[str]:
        """第一阶段：生成初始答案。"""
        messages = [
            {"role": "system", "content": self.prompts["initial"].format(task=task)},
            {"role": "user", "content": task},
        ]
        return self.llm.think(messages, **kwargs)

    def _reflect(self, task: str, content: str, **kwargs) -> Optional[str]:
        """第二阶段：反思当前答案，给出改进反馈。"""
        prompt = self.prompts["reflect"].format(task=task, content=content)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "请审查以上回答并给出反馈意见。"},
        ]
        return self.llm.think(messages, **kwargs)

    def _refine(self, task: str, last_attempt: str, feedback: str, **kwargs) -> Optional[str]:
        """第三阶段：根据反馈优化答案。"""
        messages = [
            {"role": "system", "content": self.prompts["refine"].format(
                task=task, last_attempt=last_attempt, feedback=feedback
            )},
            {"role": "user", "content": "请根据反馈意见提供改进后的回答。"},
        ]
        return self.llm.think(messages, **kwargs)

    # ================================================================
    # 辅助方法
    # ================================================================

    def _check_no_improvement(self, feedback: str) -> bool:
        """检测反馈中是否包含无需改进的标记。"""
        return bool(re.search(r'\[NO_IMPROVEMENT_NEEDED\]', feedback))

    def _save_to_history(self, input_text: str, response: str):
        """保存用户输入和回答到 Agent 历史。"""
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(response, "assistant"))
