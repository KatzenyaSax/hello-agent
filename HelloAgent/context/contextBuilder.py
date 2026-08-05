from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import math

from context.contextItem import ContextItem
from context.contextConfig import ContextConfig
from context.countToken import count_tokens

from core.message import Message
from tool.builtin.memoryTool import MemoryTool
from tool.builtin.ragTool import RAGTool

import tiktoken


class ContextBuilder:
    """上下文构建器 - GSSC流水线
    
    用法示例：
    ```python
    builder = ContextBuilder(
        memory_tool=memory_tool,
        rag_tool=rag_tool,
        config=ContextConfig(max_tokens=8000)
    )
    
    context = builder.build(
        user_query="用户问题",
        conversation_history=[...],
        system_instructions="系统指令"
    )
    ```
    """
    
    def __init__(
        self,
        memory_tool: Optional[MemoryTool] = None,
        rag_tool: Optional[RAGTool] = None,
        config: Optional[ContextConfig] = None
    ):
        self.memory_tool = memory_tool
        self.rag_tool = rag_tool
        self.config = config or ContextConfig()
        self._encoding = tiktoken.get_encoding("cl100k_base")
    
    def build(
        self,
        user_query: str,
        conversation_history: Optional[List[Message]] = None,
        system_instructions: Optional[str] = None,
        additional_items: Optional[List[ContextItem]] = None
    ) -> str:
        """构建完整上下文
        
        Args:
            user_query: 用户查询
            conversation_history: 对话历史
            system_instructions: 系统指令
            additional_items: 额外的上下文包
            
        Returns:
            结构化上下文字符串
        """
        # 1. Gather: 收集候选信息
        items = self._gather(
            user_query=user_query,
            conversation_history=conversation_history or [],
            system_instructions=system_instructions,
            additional_items=additional_items or []
        )
        
        # 2. Select: 筛选与排序
        selected_items = self._select(items, user_query)
        
        # 3. Structure: 组织成结构化模板
        structured_context = self._structure(
            selected_items=selected_items,
            user_query=user_query,
            system_instructions=system_instructions
        )
        
        # 4. Compress: 压缩与规范化（如果超预算）
        final_context = self._compress(structured_context)
        
        return final_context
    
    def _gather(
        self,
        user_query: str,
        conversation_history: List[Message],
        system_instructions: Optional[str],
        additional_items: List[ContextItem]
    ) -> List[ContextItem]:
        """Gather: 收集候选信息"""
        items = []
        
        # P0: 系统指令（强约束）
        if system_instructions:
            items.append(ContextItem(
                content=system_instructions,
                relevance_score = 1.0,
                metadata={"type": "instructions"}
            ))
        
        # P1: 从记忆中获取任务状态与关键结论
        if self.memory_tool:
            try:
                # 搜索任务状态相关记忆
                state_results = self.memory_tool.run({
                        "action": "search",
                        "query": "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                        "min_importance": 0.7,
                        "limit": 5
                    })
                if state_results and "未找到" not in state_results:
                    items.append(ContextItem(
                        content=state_results,
                        metadata={"type": "task_state", "importance": "high"}
                    ))
                
                # 搜索与当前查询相关的记忆
                related_results = self.memory_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if related_results and "未找到" not in related_results:
                    items.append(ContextItem(
                        content=related_results,
                        metadata={"type": "related_memory"}
                    ))
            except Exception as e:
                print(f"⚠️ 记忆检索失败: {e}")
        
        # P2: 从RAG中获取事实证据
        if self.rag_tool:
            try:
                rag_results = self.rag_tool.run({
                    "action": "search",
                    "query": user_query,
                    "limit": 5
                })
                if rag_results and "未找到" not in rag_results and "错误" not in rag_results:
                    items.append(ContextItem(
                        content=rag_results,
                        metadata={"type": "knowledge_base"}
                    ))
            except Exception as e:
                print(f"⚠️ RAG检索失败: {e}")
        
        # P3: 对话历史（辅助材料）
        if conversation_history:
            # 只保留最近N条
            recent_history = conversation_history[-10:]
            history_text = "\n".join([
                f"[{msg.role}] {msg.content}"
                for msg in recent_history
            ])
            items.append(ContextItem(
                content=history_text,
                metadata={"type": "history", "count": len(recent_history)}
            ))
        
        # 添加额外包
        items.extend(additional_items)
        
        return items
    
    def _select(
        self,
        items: List[ContextItem],
        user_query: str
    ) -> List[ContextItem]:
        """Select: 基于分数与预算的筛选"""
        # 1) 计算相关性（jieba 中文分词后的关键词重叠）
        import jieba
        query_tokens = set(jieba.cut(user_query.lower()))
        # 去掉停用词和标点，只保留有意义的词
        import re
        stopwords = {"了", "的", "是", "吗", "呢", "怎么", "如何", "什么", "在", "我", "你",
                     "一个", "这个", "那个", "与", "和", "及", "或", "？", "?", "，", ",", "。",
                     ".", "！", "!", "：", ":", "；", ";"}
        query_tokens = {t for t in query_tokens if t.strip() and t not in stopwords}
        
        for item in items:
            content_tokens = set(jieba.cut(item.content.lower()))
            content_tokens = {t for t in content_tokens if t.strip() and t not in stopwords}
            if len(query_tokens) > 0:
                overlap = len(query_tokens & content_tokens)
                item.relevance_score = overlap / len(query_tokens)
            else:
                item.relevance_score = 0.0
        
        # 2) 计算新近性（指数衰减）
        def recency_score(ts: datetime) -> float:
            delta = max((datetime.now() - ts).total_seconds(), 0)
            tau = 3600  # 1小时时间尺度，可暴露到配置
            return math.exp(-delta / tau)
        
        # 3) 计算复合分：0.7*相关性 + 0.3*新近性
        scored_items: List[Tuple[float, ContextItem]] = []
        for p in items:
            rec = recency_score(p.timestamp)
            score = 0.7 * p.relevance_score + 0.3 * rec
            scored_items.append((score, p))
        
        # 4) 系统指令单独拿出，固定纳入
        system_items = [p for (_, p) in scored_items if p.metadata.get("type") == "instructions"]
        remaining = [p for (s, p) in sorted(scored_items, key=lambda x: x[0], reverse=True)
                     if p.metadata.get("type") != "instructions"]
        
        # 5) 依据 min_relevance 过滤（对非系统包）
        filtered = [p for p in remaining if p.relevance_score >= self.config.min_relevance]
        
        # 6) 按预算填充
        available_tokens = self.config.get_available_tokens()
        selected: List[ContextItem] = []
        used_tokens = 0
        
        # 先放入系统指令（不排序）
        for p in system_items:
            if used_tokens + p.token_count <= available_tokens:
                selected.append(p)
                used_tokens += p.token_count
        
        # 再按分数加入其余
        for p in filtered:
            if used_tokens + p.token_count > available_tokens:
                continue
            selected.append(p)
            used_tokens += p.token_count
        
        return selected
    
    def _structure(
        self,
        selected_items: List[ContextItem],
        user_query: str,
        system_instructions: Optional[str]
    ) -> str:
        """Structure: 组织成结构化上下文模板"""
        sections = []
        
        # [Role & Policies] - 系统指令
        p0_items = [p for p in selected_items if p.metadata.get("type") == "instructions"]
        if p0_items:
            role_section = "[Role & Policies]\n"
            role_section += "\n".join([p.content for p in p0_items])
            sections.append(role_section)
        
        # [Task] - 当前任务
        sections.append(f"[Task]\n用户问题：{user_query}")
        
        # [State] - 任务状态
        p1_items = [p for p in selected_items if p.metadata.get("type") == "task_state"]
        if p1_items:
            state_section = "[State]\n关键进展与未决问题：\n"
            state_section += "\n".join([p.content for p in p1_items])
            sections.append(state_section)
        
        # [Evidence] - 事实证据
        p2_items = [
            p for p in selected_items
            if p.metadata.get("type") in {"related_memory", "knowledge_base", "retrieval", "tool_result", "note"}
        ]
        if p2_items:
            evidence_section = "[Evidence]\n事实与引用：\n"
            for p in p2_items:
                evidence_section += f"\n{p.content}\n"
            sections.append(evidence_section)
        
        # [Context] - 辅助材料（历史等）
        p3_items = [p for p in selected_items if p.metadata.get("type") == "history"]
        if p3_items:
            context_section = "[Context]\n对话历史与背景：\n"
            context_section += "\n".join([p.content for p in p3_items])
            sections.append(context_section)
        
        # [Output] - 输出约束
        output_section = """[Output]
                            请按以下格式回答：
                            1. 结论（简洁明确）
                            2. 依据（列出支撑证据及来源）
                            3. 风险与假设（如有）
                            4. 下一步行动建议（如适用）"""
        sections.append(output_section)
        
        return "\n\n".join(sections)
    
    def _compress(self, context: str) -> str:
        """Compress: 压缩与规范化"""
        if not self.config.enable_compression:
            return context
        
        current_tokens = count_tokens(context)
        available_tokens = self.config.get_available_tokens()
        
        if current_tokens <= available_tokens:
            return context
        
        # 简单截断策略（保留前N个token）
        # 实际应用中可用LLM做高保真摘要
        print(f"⚠️ 上下文超预算 ({current_tokens} > {available_tokens})，执行截断")
        
        # 按段落截断，保留结构
        lines = context.split("\n")
        compressed_lines = []
        used_tokens = 0
        
        for line in lines:
            line_tokens = count_tokens(line)
            if used_tokens + line_tokens > available_tokens:
                break
            compressed_lines.append(line)
            used_tokens += line_tokens
        
        return "\n".join(compressed_lines)
