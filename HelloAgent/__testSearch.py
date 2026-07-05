from tool.builtin.search import SearchTool
from tool.toolRegistry import ToolRegistry

if __name__ == "__main__":
    # 测试 SearchTool 的初始化和搜索功能

    registry = ToolRegistry()
    # 注册 SearchTool
    search_tool = SearchTool()
    registry.register_tool(search_tool)
    query = "antropic风评为什么这么差？"
    print(f"正在搜索: {query}")
    result = registry.execute_tool("search", query=query)
    print(f"搜索结果:\n{result}")
