# __testReflectionAgent.py
# ReflectionAgent 功能测试 —— 覆盖反思循环、智能终止、历史管理、自定义提示词
from dotenv import load_dotenv
from core.llm import LLM
from agent.reflectionAgent import ReflectionAgent

load_dotenv()

llm = LLM()

# ================================================================
# 测试1: 基础反思模式（auto_finish=False，走满 max_reflections 轮）
# ================================================================
print("=" * 60)
print("测试1: 基础反思模式（max_reflections=2, 纯计数）")
print("=" * 60)

agent1 = ReflectionAgent(
    name="Reflection基础助手",
    llm=llm,
    max_reflections=2,
    auto_finish=False,
    verbose=True,
)

response1 = agent1.run("请解释什么是递归，并给出一个简单的Python示例")
print(f"\n📌 最终响应: {response1}\n")

# ================================================================
# 测试2: 智能终止模式（auto_finish=True）
# ================================================================
print("=" * 60)
print("测试2: 智能终止模式（auto_finish=True）")
print("=" * 60)

agent2 = ReflectionAgent(
    name="Reflection智能终止助手",
    llm=llm,
    max_reflections=3,
    auto_finish=True,
    verbose=True,
)

response2 = agent2.run("1 + 1 等于几？")
print(f"\n📌 最终响应: {response2}\n")

# ================================================================
# 测试3: 单轮反思（max_reflections=1）
# ================================================================
print("=" * 60)
print("测试3: 单轮反思（max_reflections=1）")
print("=" * 60)

agent3 = ReflectionAgent(
    name="Reflection单轮助手",
    llm=llm,
    max_reflections=1,
    auto_finish=False,
    verbose=True,
)

response3 = agent3.run("用一句话概括机器学习的定义")
print(f"\n📌 最终响应: {response3}\n")

# ================================================================
# 测试4: 对话历史管理（get_history / clear_history）
# ================================================================
print("=" * 60)
print("测试4: 对话历史管理")
print("=" * 60)

hist_agent = ReflectionAgent(
    name="Reflection历史助手",
    llm=llm,
    max_reflections=1,
    auto_finish=False,
    verbose=False,
)

# 第一轮
hist_agent.run("Python 是什么？")
# 历史应包含: 1条user + 1条assistant(初始) + 1条assistant(反馈) + 1条assistant(优化)
print(f"第一轮后历史条数: {len(hist_agent.get_history())}")
for i, msg in enumerate(hist_agent.get_history()):
    print(f"  [{i}] {msg.role}: {msg.content[:80]}{'...' if len(msg.content) > 80 else ''}")

# 第二轮
hist_agent.run("Java 是什么？")
print(f"第二轮后历史条数: {len(hist_agent.get_history())}")

# 清空
hist_agent.clear_history()
print(f"清空后历史条数: {len(hist_agent.get_history())}")

# 清空后正常工作
hist_agent.run("C++ 是什么？")
print(f"清空后重新对话，历史条数: {len(hist_agent.get_history())}")

print()

# ================================================================
# 测试5: 自定义提示词（合并策略）
# ================================================================
print("=" * 60)
print("测试5: 自定义提示词")
print("=" * 60)

custom_prompts = {
    "initial": "请用专业的语气完成任务: {task}",
}

agent5 = ReflectionAgent(
    name="Reflection自定义助手",
    llm=llm,
    max_reflections=1,
    auto_finish=False,
    prompts=custom_prompts,
    verbose=False,
)

# 验证合并: initial 被覆盖，reflect 和 refine 保持默认
assert agent5.prompts["initial"] == custom_prompts["initial"], "initial 应被覆盖"
assert "[NO_IMPROVEMENT_NEEDED]" in agent5.prompts["reflect"], "reflect 应保留默认"
print("✅ 提示词合并验证通过")
print(f"  initial: {'已覆盖' if agent5.prompts['initial'] == custom_prompts['initial'] else '未覆盖'}")
print(f"  reflect: {'保持默认' if '[NO_IMPROVEMENT_NEEDED]' in agent5.prompts['reflect'] else '异常'}")
print(f"  refine:  {'保持默认' if agent5.prompts['refine'] != '' else '异常'}")
print()

response5 = agent5.run("请用一句话总结 Docker 的作用")
print(f"\n📌 最终响应: {response5}\n")

# ================================================================
print("=" * 60)
print("✅ 所有测试完成")
print("=" * 60)
