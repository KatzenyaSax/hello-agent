from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict

class MemoryItem(BaseModel):
    """记忆项数据结构"""
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}
    arbitrary_types_allowed: bool = True