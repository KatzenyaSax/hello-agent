from typing import Callable, Any, Dict
from .tool import Tool

class ToolRegistry:
    """HelloAgents工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, dict[str, Any]] = {}

    def register_tool(self, tool: Tool):
        """注册Tool对象"""
        if tool.name in self._tools:
            print(f"⚠️ 警告:工具 '{tool.name}' 已存在，将被覆盖。")
        self._tools[tool.name] = tool
        print(f"✅ 工具 '{tool.name}' 已注册。")
        
    def register_function(self, name: str, description: str, func: Callable[[str], str]):
        """
        直接注册函数作为工具（简便方式）

        Args:
            name: 工具名称
            description: 工具描述
            func: 工具函数，接受字符串参数，返回字符串结果
        """
        if name in self._functions:
            print(f"⚠️ 警告:工具 '{name}' 已存在，将被覆盖。")

        self._functions[name] = {
            "description": description,
            "func": func
        }
        print(f"✅ 工具 '{name}' 已注册。")

    def execute_function(self, name: str, *args, **kwargs) -> Any:
        """执行注册的函数工具"""
        if name not in self._functions:
            raise ValueError(f"工具 '{name}' 未注册。")
        func = self._functions[name]["func"]
        return func(*args, **kwargs)
    
    def get_function(self, name: str) -> Callable[[str], str]:
        """获取注册的函数工具"""
        if name not in self._functions:
            raise ValueError(f"工具 '{name}' 未注册。")
        return self._functions[name]["func"]
    
    def execute_tool(self, name: str, *args, **kwargs) -> Any:
        """执行注册的Tool对象"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册。")
        tool = self._tools[name]
        return tool.run(*args, **kwargs)
    
    def get_tool(self, name: str) -> Tool:
        """获取注册的Tool对象"""
        if name not in self._tools:
            raise ValueError(f"工具 '{name}' 未注册。")
        return self._tools[name]

    def get_tools_description(self) -> str:
        """获取所有可用工具的格式化描述字符串"""
        descriptions = []

        # Tool对象描述
        for tool in self._tools.values():
            descriptions.append(f"- {tool.name}: {tool.description}")

        # 函数工具描述
        for name, info in self._functions.items():
            descriptions.append(f"- {name}: {info['description']}")

        return "\n".join(descriptions) if descriptions else "暂无可用工具"