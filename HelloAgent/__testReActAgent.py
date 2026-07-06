# __testReActAgent.py
# ReActAgent 功能测试 —— 覆盖批量/流式模式、函数工具/Tool对象、执行轨迹、历史管理
from dotenv import load_dotenv
from core.llm import LLM
from agent.reActAgent import ReActAgent
from tool.toolRegistry import ToolRegistry
from tool.tool import Tool
from tool.builtin.calculator import calculator

load_dotenv()

llm = LLM()

# ================================================================
# 测试1: 批量模式 + 函数工具（calculator 通过 register_function 注册）
# ================================================================
print("=" * 60)
print("测试1: 批量模式 + 函数工具")
print("=" * 60)

registry1 = ToolRegistry()
registry1.register_function(
    name="calculator",
    description="数学计算工具，支持加减乘除和 sqrt 函数",
    func=calculator,
)

agent1 = ReActAgent(
    name="ReAct数学助手",
    llm=llm,
    system_prompt="你是一个擅长数学的AI助手。",
    tool_registry=registry1,
    max_iterations=5,
    is_stream=False,
    verbose=True,
)

response1 = agent1.run("请帮我计算 15 * 8 + 32 的结果")
print(f"\n📌 最终响应: {response1}\n")

# ================================================================
# 测试2: 批量模式 + Tool 对象（自定义 EchoTool 验证 is_tool 分支）
# ================================================================
print("=" * 60)
print("测试2: 批量模式 + Tool 对象")
print("=" * 60)


class EchoTool(Tool):
    """回显工具：把输入原样返回，用于测试 Tool 对象执行路径"""

    def __init__(self):
        super().__init__(
            name="echo",
            description="回显工具，将输入内容原样返回。可用于确认信息。",
        )

    def run(self, parameters: dict) -> str:
        return f"[回显] {parameters.get('input', '')}"

    def get_parameters(self):
        return []


registry2 = ToolRegistry()
registry2.register_tool(EchoTool())

agent2 = ReActAgent(
    name="ReAct回显助手",
    llm=llm,
    system_prompt="你是一个测试助手，需要回显确认用户输入。",
    tool_registry=registry2,
    max_iterations=3,
    is_stream=False,
    verbose=True,
)

response2 = agent2.run("请用 echo 工具帮我确认: Hello ReAct!")
print(f"\n📌 最终响应: {response2}\n")

# ================================================================
# 测试3: 流式模式（is_stream=True，边输出边收集）
# ================================================================
print("=" * 60)
print("测试3: 流式模式")
print("=" * 60)

registry3 = ToolRegistry()
registry3.register_function(
    name="calculator",
    description="数学计算工具",
    func=calculator,
)

agent3 = ReActAgent(
    name="ReAct流式助手",
    llm=llm,
    system_prompt="你是一个擅长数学的AI助手。",
    tool_registry=registry3,
    max_iterations=5,
    is_stream=True,   # 开启流式
    verbose=True,
)

response3 = agent3.run("帮我算一下 100 / 4 + 7")
print(f"\n📌 最终响应: {response3}\n")

# ================================================================
# 测试4: 执行轨迹（get_execution_trace）
# ================================================================
print("=" * 60)
print("测试4: 执行轨迹")
print("=" * 60)

# 复用 agent1（已有历史），再跑一个问题
trace_agent = ReActAgent(
    name="ReAct轨迹助手",
    llm=llm,
    tool_registry=registry1,  # 用 calculator 的那个 registry
    max_iterations=5,
    is_stream=False,
    verbose=False,
)

trace_agent.run("计算 (3 + 5) * 2")
trace = trace_agent.get_execution_trace()
print(f"执行轨迹共 {len(trace)} 步:")
for i, step in enumerate(trace, 1):
    print(f"  --- 步骤 {i} ---")
    print(f"  {step[:200]}{'...' if len(step) > 200 else ''}")

# ================================================================
# 测试5: 对话历史管理（get_history / clear_history）
# ================================================================
print("\n" + "=" * 60)
print("测试5: 对话历史管理")
print("=" * 60)

hist_agent = ReActAgent(
    name="ReAct历史助手",
    llm=llm,
    tool_registry=registry1,
    max_iterations=5,
    is_stream=False,
    verbose=False,
)

# 先跑一轮
hist_agent.run("1 + 1 等于几？")
print(f"第一轮后历史条数: {len(hist_agent.get_history())}")

# 再跑一轮
hist_agent.run("2 + 2 等于几？")
print(f"第二轮后历史条数: {len(hist_agent.get_history())}")

# 清空
hist_agent.clear_history()
print(f"清空后历史条数: {len(hist_agent.get_history())}")

# 再跑一轮确认清空后正常工作
hist_agent.run("3 + 3 等于几？")
print(f"清空后重新对话，历史条数: {len(hist_agent.get_history())}")

print("\n" + "=" * 60)
print("✅ 所有测试完成")
print("=" * 60)
