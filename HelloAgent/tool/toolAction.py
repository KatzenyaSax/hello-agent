from typing import Callable
def tool_action(name: str = None, description: str = None):
    """装饰器：标记一个方法为可展开的工具 action

    用法:
        @tool_action("memory_add", "添加新记忆")
        def _add_memory(self, content: str, importance: float = 0.5) -> str:
            '''添加记忆

            Args:
                content: 记忆内容
                importance: 重要性分数
            '''
            ...

    Args:
        name: 工具名称（如果不提供，从方法名自动生成）
        description: 工具描述（如果不提供，从 docstring 提取）
    """
    def decorator(func: Callable):
        func._is_tool_action = True
        func._tool_name = name
        func._tool_description = description
        return func
    return decorator