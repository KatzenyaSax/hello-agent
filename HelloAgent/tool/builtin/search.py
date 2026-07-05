from typing import Any, Callable, Dict, Optional
from serpapi import SerpApiClient, GoogleSearch
from tavily import TavilyClient
import os
from ..tool import Tool
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


class SearchTool(Tool):
    """
    智能混合搜索工具，使用 SerpApi 和 Tavily API 进行网页搜索。
    """

    def __init__(self):
        super().__init__(
            name="search",
            description="一个智能网页搜索引擎。支持混合搜索模式，自动选择最佳搜索源。"
        )
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_API_KEY")
        self.available_backends = []
        self._setup_backends()
        print(f"已选择搜索引擎: {self.available_backends}")


    def _setup_backends(self):
        """
        设置可用的搜索后端。
        """
        if self.tavily_key:
            self.available_backends.append("tavily")
        elif self.serpapi_key:
            self.available_backends.append("serpapi")
        if not self.available_backends:
            raise ValueError("No available search backends. Please set TAVILY_API_KEY or SERPAPI_API_KEY.")
        
    def run(self, query: str) -> Dict[str, Any]:
        """
        执行搜索操作，根据可用的后端选择最佳搜索源。
        """
        if "tavily" in self.available_backends:
            return self._search_tavily(query)
        elif "serpapi" in self.available_backends:
            return self._search_serpapi(query)
        else:
            raise ValueError("No available search backends to perform the search.")
        
    def _search_tavily(self, query: str) -> Dict[str, Any]:
        """使用Tavily搜索"""
        self.tavily_client = TavilyClient(api_key=self.tavily_key)

        response = self.tavily_client.search(query=query, max_results=3)

        if response.get('answer'):
            result = f"💡 AI直接答案:{response['answer']}\n\n"
        else:
            result = ""

        result += "🔗 相关结果:\n"
        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"    {item.get('content', '')[:150]}...\n\n"

        return result

    def _search_serpapi(self, query: str) -> Dict[str, Any]:
        """
        使用 SerpApi 执行搜索。
        """
        search = GoogleSearch({
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "num": 3
        })

        results = search.get_dict()

        result = "🔗 Google搜索结果:\n"
        if "organic_results" in results:
            for i, res in enumerate(results["organic_results"][:3], 1):
                result += f"[{i}] {res.get('title', '')}\n"
                result += f"    {res.get('snippet', '')}\n\n"

        return result
    

    def get_parameters(self) -> Dict[str, Any]:
        """
        获取工具参数定义。
        """
        return {
            "query": {
                "type": "string",
                "description": "搜索查询字符串",
                "required": True
            }
        }