# __testPlanAndSolveAgent.py
# PlanAndSolveAgent 功能测试 —— 覆盖多步规划、单步执行、历史管理、步骤结果追踪
from dotenv import load_dotenv
from core.llm import LLM
from agent.planAndSolveAgent import PlanAndSolveAgent

load_dotenv()

llm = LLM()

# ================================================================
# 测试1: 多步规划与执行（复杂问题 → 多子任务）
# ================================================================
print("=" * 60)
print("测试1: 多步规划与执行")
print("=" * 60)

agent1 = PlanAndSolveAgent(
    name="PlanSolve多步助手",
    llm=llm,
    verbose=True,
)

response1 = agent1.run(
    "请帮我分析：Python和JavaScript在异步编程方面的主要区别是什么？"
    "它们的解决方案各有什么优劣？"
)
print(f"\n📌 最终响应: {response1}\n")

# ================================================================
# 测试2: 单步问题（简单问题，验证单步计划正常执行）
# ================================================================
print("=" * 60)
print("测试2: 单步问题")
print("=" * 60)

agent2 = PlanAndSolveAgent(
    name="PlanSolve单步助手",
    llm=llm,
    verbose=True,
)

response2 = agent2.run("请用一句话解释什么是Docker")
print(f"\n📌 最终响应: {response2}\n")

# ================================================================
# 测试3: 步骤结果追踪（get_step_results）
# ================================================================
print("=" * 60)
print("测试3: 步骤结果追踪")
print("=" * 60)

agent3 = PlanAndSolveAgent(
    name="PlanSolve追踪助手",
    llm=llm,
    verbose=False,
)

agent3.run("比较TCP和UDP协议的主要区别，并说明各自适用场景")
step_results = agent3.get_step_results()
print(f"计划共 {len(step_results)} 步:")
for i, sr in enumerate(step_results, 1):
    print(f"  步骤 {i}: {sr['step']}")
    print(f"  结果 {i}: {sr['result'][:100]}{'...' if len(sr['result']) > 100 else ''}")
    print()

# ================================================================
# 测试4: 对话历史管理（get_history / clear_history）
# ================================================================
print("=" * 60)
print("测试4: 对话历史管理")
print("=" * 60)

hist_agent = PlanAndSolveAgent(
    name="PlanSolve历史助手",
    llm=llm,
    verbose=False,
)

# 第一轮
hist_agent.run("什么是Git？它有什么用？")
print(f"第一轮后历史条数: {len(hist_agent.get_history())}")
for i, msg in enumerate(hist_agent.get_history()):
    print(f"  [{i}] {msg.role}: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")

# 第二轮
hist_agent.run("什么是GitHub？它和Git的关系是什么？")
print(f"第二轮后历史条数: {len(hist_agent.get_history())}")

# 清空
hist_agent.clear_history()
print(f"清空后历史条数: {len(hist_agent.get_history())}")

# 清空后正常工作
hist_agent.run("什么是GitLab？")
print(f"清空后重新对话，历史条数: {len(hist_agent.get_history())}")

print()

# ================================================================
# 测试5: 复杂推理问题（验证多步依赖的正确性）
# ================================================================
print("=" * 60)
print("测试5: 复杂推理问题")
print("=" * 60)

agent5 = PlanAndSolveAgent(
    name="PlanSolve推理助手",
    llm=llm,
    verbose=True,
)

response5 = agent5.run(
    "某公司有3个部门：研发部30人、市场部20人、财务部10人。"
    "现在要组建一个6人的跨部门项目组，要求每个部门至少1人。"
    "请计算有多少种不同的选人方案。"
)
print(f"\n📌 最终响应: {response5}\n")

# ================================================================
print("=" * 60)
print("✅ 所有测试完成")
print("=" * 60)
