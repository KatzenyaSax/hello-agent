import asyncio
import concurrent.futures
from typing import Dict, Any, List, Callable
from tool.toolRegistry import ToolRegistry

class AsyncToolExecutor:
    """异步工具执行器"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具，能够判断工具是函数还是Tool对象，并调用相应的执行方法"""
        loop = asyncio.get_event_loop()

        if self.registry.is_function(tool_name):
            def _execute():
                return self.registry.execute_function(tool_name, input_data)
        elif self.registry.is_tool(tool_name):
            def _execute():
                return self.registry.execute_tool(tool_name, input_data)
        else:
            raise ValueError(f"工具 '{tool_name}' 未在注册表中找到（既非函数也非Tool对象）")

        result = await loop.run_in_executor(self.executor, _execute)
        return result


    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[str]:
        """并行执行多个工具"""
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 创建异步任务
        async_tasks = []
        for task in tasks:
            tool_name = task["tool_name"]
            input_data = task["input_data"]
            async_task = self.execute_tool_async(tool_name, input_data)
            async_tasks.append(async_task)

        # 等待所有任务完成
        results = await asyncio.gather(*async_tasks)

        print(f"✅ 所有工具任务执行完成")
        return results

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


