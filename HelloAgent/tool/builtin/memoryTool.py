"""
MemoryTool - 记忆系统的统一工具接口

设计原则：统一入口，分发处理（Unified Entry, Dispatch to Handlers）

该工具将底层的 MemoryManager 封装为 LLM 可调用的 Tool，
使 Agent 具备持久化记忆和检索能力。
"""

from typing import List, Dict, Any, Optional
import json

from ..tool import Tool
from ..toolParameter import ToolParameter
from memory.memoryManager import MemoryManager
from memory.memoryConfg import MemoryConfig


class MemoryTool(Tool):
    """记忆工具 —— 让 LLM 能够存取和管理记忆

    作为记忆系统的统一接口，遵循"统一入口，分发处理"的架构模式。
    MemoryTool 专注于参数解析和结果格式化，核心逻辑委托给 MemoryManager。
    """

    def __init__(
        self,
        user_id: str = "default_user",
        config: Optional[MemoryConfig] = None
    ):
        super().__init__(
            name="memory",
            description=(
                "记忆管理工具，用于存储、检索、更新和删除记忆。"
                "支持以下操作：\n"
                "- add: 添加一条新记忆。需要 content（记忆内容），可选 memory_type "
                "（working/episodic/semantic，默认自动推断）、importance（0-1，默认自动计算）\n"
                "- search: 按关键词检索记忆。需要 query（查询内容），可选 limit（返回数量，默认5）\n"
                "- update: 更新已有记忆。需要 memory_id，可选 content/importance\n"
                "- remove: 删除指定记忆。需要 memory_id\n"
                "- summary: 查看记忆系统的统计摘要（总数、各类型容量等）\n"
                "- forget: 触发记忆遗忘机制。可选 strategy（importance_based/time_based/capacity_based）、"
                "threshold（遗忘阈值，默认0.1）\n"
                "- consolidate: 将重要的短期记忆整合为长期记忆。"
                "可选 from_type（默认working）、to_type（默认episodic）、importance_threshold（默认0.7）\n\n"
                "记忆类型说明：\n"
                "- working: 工作记忆（当前会话上下文，容量有限，过期自动清理）\n"
                "- episodic: 情景记忆（重要交互记录，长期保留）\n"
                "- semantic: 语义记忆（抽象规则和用户偏好，永久保留）\n\n"
                "使用建议：\n"
                "- 用户说'记住XXX'时，根据内容类型选择合适的 memory_type\n"
                "- 用户偏好/规范类信息 → semantic，重要性 0.8+\n"
                "- 临时上下文/当前话题 → working，重要性 0.5\n"
                "- 重要事件/决定 → episodic，重要性 0.7+\n"
                "- 在回答用户问题前，先用 search 检索相关记忆"
            )
        )
        self.user_id = user_id
        self._manager = MemoryManager(
            config=config,
            user_id=user_id,
            enable_working=True,
            enable_episodic=True,   # 暂未实现
            enable_semantic=True    # 暂未实现
        )

    # ------------------------------------------------------------------
    # Tool 接口实现
    # ------------------------------------------------------------------

    def run(self, parameters: Dict[str, Any]) -> str:
        """统一入口：根据 action 分发给具体处理方法

        Args:
            parameters: 包含 action 及其他所需参数的字典

        Returns:
            格式化后的结果字符串
        """
        action = parameters.get("action", "search")

        dispatch = {
            "add":         self._handle_add,
            "search":      self._handle_search,
            "update":      self._handle_update,
            "remove":      self._handle_remove,
            "summary":     self._handle_summary,
            "forget":      self._handle_forget,
            "consolidate": self._handle_consolidate,
        }

        handler = dispatch.get(action)
        if handler is None:
            return (
                f"❌ 不支持的操作: '{action}'。"
                f"可用操作: {', '.join(dispatch.keys())}"
            )

        try:
            return handler(parameters)
        except ValueError as e:
            return f"❌ 参数错误: {e}"
        except Exception as e:
            return f"❌ 操作失败: {e}"

    def get_parameters(self) -> List[ToolParameter]:
        """获取工具参数定义（用于 OpenAI function calling schema）"""
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型: add, search, update, remove, summary, forget, consolidate",
                required=True
            ),
            ToolParameter(
                name="content",
                type="string",
                description="记忆内容文本（add/update 时使用）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="检索查询关键词（search 时使用）",
                required=False
            ),
            ToolParameter(
                name="memory_type",
                type="string",
                description="记忆类型: working, episodic, semantic（add 时可选，默认自动推断）",
                required=False,
                default="working"
            ),
            ToolParameter(
                name="memory_id",
                type="string",
                description="记忆ID（update/remove 时使用）",
                required=False
            ),
            ToolParameter(
                name="importance",
                type="number",
                description="重要性分数 0-1（add/update 时可选，默认自动计算）",
                required=False
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回数量限制（search 时使用，默认5）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="strategy",
                type="string",
                description="遗忘策略: importance_based, time_based, capacity_based（forget 时使用，默认 importance_based）",
                required=False,
                default="importance_based"
            ),
            ToolParameter(
                name="threshold",
                type="number",
                description="遗忘阈值（forget 时使用，默认0.1）",
                required=False,
                default=0.1
            ),
        ]

    # ------------------------------------------------------------------
    # Action 处理器
    # ------------------------------------------------------------------

    def _handle_add(self, params: Dict[str, Any]) -> str:
        """处理 add 操作"""
        content = params.get("content")
        if not content:
            raise ValueError("add 操作需要 'content' 参数")

        memory_type = params.get("memory_type", "working")
        importance = params.get("importance")
        # 将 memory_type 注入 metadata，使 MemoryManager 优先采纳 LLM 的判断
        metadata = params.get("metadata", {})
        metadata["type"] = memory_type

        memory_id = self._manager.add_memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            metadata=metadata,
            auto_classify=True
        )

        return (
            f"✅ 记忆已添加\n"
            f"   ID: {memory_id}\n"
            f"   类型: {memory_type}\n"
            f"   内容: {content[:200]}{'...' if len(content) > 200 else ''}"
        )

    def _handle_search(self, params: Dict[str, Any]) -> str:
        """处理 search 操作"""
        query = params.get("query")
        if not query:
            raise ValueError("search 操作需要 'query' 参数")

        limit = params.get("limit", 5)
        memory_types = params.get("memory_types")  # 可选：限定检索的记忆类型列表

        results = self._manager.retrieve_memories(
            query=query,
            memory_types=memory_types,
            limit=limit
        )

        if not results:
            return f"🔍 未找到与 '{query}' 相关的记忆。"

        lines = [f"🔍 搜索 '{query}' 的结果（共 {len(results)} 条）:"]
        for i, item in enumerate(results, 1):
            lines.append(
                f"  [{i}] [{item.memory_type}] {item.content[:150]}"
                f"{'...' if len(item.content) > 150 else ''}"
                f"  (importance={item.importance:.2f}, id={item.id[:8]}...)"
            )
        return "\n".join(lines)

    def _handle_update(self, params: Dict[str, Any]) -> str:
        """处理 update 操作"""
        memory_id = params.get("memory_id")
        if not memory_id:
            raise ValueError("update 操作需要 'memory_id' 参数")

        content = params.get("content")
        importance = params.get("importance")
        metadata = params.get("metadata")

        success = self._manager.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            metadata=metadata
        )

        if success:
            return f"✅ 记忆 {memory_id} 已更新"
        else:
            return f"❌ 未找到记忆: {memory_id}"

    def _handle_remove(self, params: Dict[str, Any]) -> str:
        """处理 remove 操作"""
        memory_id = params.get("memory_id")
        if not memory_id:
            raise ValueError("remove 操作需要 'memory_id' 参数")

        success = self._manager.remove_memory(memory_id)

        if success:
            return f"✅ 记忆 {memory_id} 已删除"
        else:
            return f"❌ 未找到记忆: {memory_id}"

    def _handle_summary(self, params: Dict[str, Any]) -> str:
        """处理 summary 操作"""
        stats = self._manager.get_memory_stats()

        lines = ["📊 记忆系统摘要:"]
        lines.append(f"   用户: {stats['user_id']}")
        lines.append(f"   总记忆数: {stats['total_memories']}")
        lines.append(f"   启用的记忆类型: {', '.join(stats['enabled_types'])}")
        lines.append(f"   配置: 最大容量={stats['config']['max_capacity']}, "
                     f"重要性阈值={stats['config']['importance_threshold']}, "
                     f"衰减因子={stats['config']['decay_factor']}")

        for mem_type, type_stats in stats.get("memories_by_type", {}).items():
            lines.append(f"   [{mem_type}] 活跃={type_stats.get('count', 0)}, "
                         f"容量使用率={type_stats.get('capacity_usage', 0):.1%}")

        return "\n".join(lines)

    def _handle_forget(self, params: Dict[str, Any]) -> str:
        """处理 forget 操作"""
        strategy = params.get("strategy", "importance_based")
        threshold = params.get("threshold", 0.1)
        max_age_days = params.get("max_age_days", 30)

        forgotten = self._manager.forget_memories(
            strategy=strategy,
            threshold=threshold,
            max_age_days=max_age_days
        )

        return f"🧹 记忆遗忘完成: {forgotten} 条记忆被清理（策略: {strategy}）"

    def _handle_consolidate(self, params: Dict[str, Any]) -> str:
        """处理 consolidate 操作"""
        from_type = params.get("from_type", "working")
        to_type = params.get("to_type", "episodic")
        importance_threshold = params.get("importance_threshold", 0.7)

        # 检查目标记忆类型是否已启用
        if to_type not in self._manager.memory_types:
            return (
                f"⚠️ 目标记忆类型 '{to_type}' 尚未实现。"
                f"当前只启用了工作记忆（working），情景记忆和语义记忆有待开发。"
            )

        count = self._manager.consolidate_memories(
            from_type=from_type,
            to_type=to_type,
            importance_threshold=importance_threshold
        )

        return (
            f"📦 记忆整合完成: {count} 条记忆从 {from_type} 转移到 {to_type} "
            f"（重要性阈值: {importance_threshold}）"
        )

    # ------------------------------------------------------------------
    # 便捷方法（供程序化调用，非 LLM 路径）
    # ------------------------------------------------------------------

    @property
    def manager(self) -> MemoryManager:
        """暴露底层 MemoryManager，便于程序化访问"""
        return self._manager
