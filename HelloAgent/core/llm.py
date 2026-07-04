# 继承 HelloAgentLLM
import HelloAgentLLM import HelloAgentLLM


import os
from typing import Optional
from openai import OpenAI



class llm(HelloAgentLLM):
    def __init__(
        self,
        model: Optional[str]=None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = "auto",
        **kwargs
    ):
        



