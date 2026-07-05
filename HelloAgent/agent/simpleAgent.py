from core import LLM, Agent, Message, Config
from tool import ToolRegistry
from typing import Optional, Iterator
import re


class SimpleAgent(Agent):
    """一个简单的Agent实现，使用LLM进行对话。"""

    def __init__(
        self,
        name: str,
        llm: LLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        tool_registry: Optional['ToolRegistry'] = None,
        enable_tool_calling: bool = True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f"✅ {name} 初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行Agent，处理输入文本并返回LLM的响应。"""
        # 将用户输入添加到历史记录
        user_message = Message(role="user", content=input_text)
        self.add_message(user_message)

        # 构建消息列表，包括系统提示和历史记录
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        for msg in self.get_history():
            messages.append({"role": msg.role, "content": msg.content})

        # 调用LLM进行思考
        response_content = self.llm.think(messages, temperature=self.config.temperature)

        # 将LLM的响应添加到历史记录
        assistant_message = Message(role="assistant", content=response_content)
        self.add_message(assistant_message)

        return response_content



