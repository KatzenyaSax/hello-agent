from pydantic import BaseModel
from typing import List

class MemoryConfig(BaseModel):
    """记忆系统配置"""
    
    # 存储路径
    storage_path: str = "./memory_data"
    
    # 统计显示用的基础配置（仅用于展示）
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95

    # 工作记忆特定配置
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120

    # 感知记忆特定配置
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]