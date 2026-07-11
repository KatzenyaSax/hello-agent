from tool.toolRegistry import ToolRegistry
from tool.builtin.calculator import calculator
from tool.builtin.search import SearchTool
from tool.asyncToolExecutor import AsyncToolExecutor

# 使用示例
async def test_parallel_execution():
    """测试并行工具执行"""

    
    registry = ToolRegistry()
    registry.register_function(
        name="calculator",
        description="简单的数学计算工具，支持基本运算(+,-,*,/)和sqrt函数",
        func=calculator
    )
    registry.register_tool(SearchTool())

    executor = AsyncToolExecutor(registry)

    # 定义并行任务
    tasks = [
        {"tool_name": "search", "input_data": "Python编程"},
        {"tool_name": "search", "input_data": "机器学习"},
        {"tool_name": "calculator", "input_data": "2 + 2"},
        {"tool_name": "calculator", "input_data": "sqrt(16)"},
        
    ]

    # 并行执行
    results = await executor.execute_tools_parallel(tasks)

    for i, result in enumerate(results):
        print(f"任务 {i+1} 结果: {result[:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_parallel_execution())