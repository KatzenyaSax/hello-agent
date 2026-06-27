# Hello Agent Cli 构建日志

## 1. Agent 经典构建范式

本章任务：

- Cli：构建一个基本的客户端，能够成功调用 LLM

- ReAct(Reasoning and Action)：着重于实现如何让模型在执行过程不断根据当前状况动态地更新行动计划；

- Plan-and-Solve：着重于如何将模型的规划能力更加专业、严谨；

- Reflection：模型的反思能力。


### 1.1 Cli

前置库：

```bash
pip install openai python-dotenv
```

仔细阅读一下 `HelloAgentLLM.py` 具体干了什么。

#### 初始化

```py
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        # 1.初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))
        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        # 2.构建一个 OpenAI 规范的 LLM 调用接口
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)
```

没什么好说的，直接根据 `.env` 的参数构建一个 OpenAI 规范的 LLM 接口。

#### think 函数

`main` 函数中，通过构造函数构建一个接口后，调用了 `think` 函数：

```py
    print("请输入文本...")
    userPrompt = input()

    exampleMessages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": userPrompt}
        ]
        
        print("--- 调用LLM ---")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)
```

传入的 Dict 字典中，包含了系统提示词和用户提示词。

`think` 函数：

```py
    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠 正在调用 {self.model} 模型...")
        try:
            # 1. 通过LLM接口调用大模型，response为流式响应(stream=True)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            # 2. 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                if not chunk.choices:
                    continue
                content = chunk.choices[0].delta.content or ""
                print(content, end="", flush=True)
                collected_content.append(content)
            print()  # 在流式输出结束后换行
            return "".join(collected_content)

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None
```

`stream` 输出模式，每次都只返回一个增量 chunk ，每次都要打印这个 chunk，并拼接到最终输出。


### 1.2 ReAct

本节需要在 Cli 内实现 `Reasoning and Action`，即强制让模型按照 思考-行动-观察 的流程，这样模型更容易激活模型的工具意图，从而主动调用工具。

此外，ReAct 模式最重要的是工具，激活工具意图只是让模型拥有调用工具的能力，而工具调用是否可靠，还是需要自行定义。所以本节还涉及单个工具的定义，多个工具的注册发现、调用。

#### 准备工作

前置库：

```bash
pip install google-search-results
```

`.env` 添加：

```env
SERPAPI_API_KEY = "b4b0893a04b0b31a234dc828a79c336ddfadd7d962a6b6e3196d6b7c258ab111"
```

#### 定义工具 search

`search` 分三步：1.初始化；2.构建 serpapi 接口并调用；3.智能化筛选搜索结果。

第三步中，会检查是否存在 answer_box（Google的答案摘要框）或 knowledge_graph（知识图谱）等信息，如果存在，就直接返回这些最精确的答案。如果不存在，它才会退而求其次，返回前三个常规搜索结果的摘要。这种“智能解析”能为LLM提供质量更高的信息输入。

测试一下：

```py
if __name__ == '__main__':
    print(search("英伟达的历史最高股价是多少"))
```

工具调用结果：

```text
🔍 正在执行 [SerpApi] 网页搜索: 英伟达的历史最高股价是多少
[1] 英偉達(NVDA) 過往股價及數據
成交量. 2026年6月26日, 193.12, 195.55, 191.22, 192.53, 192.53, 178,906,300. 2026年6月25日, 200.08, 200.80, 192.13, 195.74, 195.74, 150,205,600.

[2] 5万亿美元！英伟达市值究竟有多惊人：几组对比告诉你……
①英伟达公司上周进一步创造了历史，成为全球首家市值突破5万亿美元的企业； ②这一里程碑式的壮举，或许也预示着这家全球顶尖的芯片公司，对世界经济和金融市场 ...

[3] NVIDIA股票图表— NASDAQ:NVDA股票价格
NVDA于2026年5月13日达到历史最高价，价格为236.54 USD，其历史最低价为0.03 USD，于1999年4月25日达到。在NVDA图表上查看更多价格动态。 查看其它股票达到的最高和最低价格
```

#### 工具执行类 ToolExecutor

如果存在很多个工具，我们构建一个执行类 `ToolExecutor`，统一注册工具、执行工具。

##### 初始化

构造工具对象时，其内部存入了一个 Dict 对象：

```py
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
```

顾名思义，是用来存储自身工具的注册表。其结构案例为：

```text
  {                                                                                              
      "Search": {                                                                                
          "description": "一个网页搜索引擎。当你需要回答关于时事...",                            
          "func": search.search   # search.py 中的 search 函数                                   
      }
      "other": {
          "description": "another tools needed when...",
          "func": other.other
      }                                                                                          
  }  
```

也就是说，一个注册表应当包含哪些东西：`什么时候用这个工具？`（自然语言描述） 和 `这个工具在哪？`（结构化语言）。

##### 注册

工具注册，本质是采用 callable 传入工具函数的指针：

```py
    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告:工具 '{name}' 已存在，将被覆盖。")
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")
```

##### 暴露工具列表

```py
    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])
```

##### 提取工具

返回 callable

```py
    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")
```

##### 测试

```py
# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search.search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")
```

结果：

```text
工具 'Search' 已注册。

--- 可用的工具 ---
- Search: 一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。

--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---
🔍 正在执行 [SerpApi] 网页搜索: 英伟达最新的GPU型号是什么
--- 观察 (Observation) ---
[1] 比较GeForce 系列最新一代显卡和前代显卡| NVIDIA
比较最新一代RTX 30 系列显卡和前代的RTX 20 系列、GTX 10 和900 系列显卡。查看规格、功能、技术支持等内容。

[2] GeForce RTX 50 系列显卡| NVIDIA
GeForce RTX™ 50 系列GPU 搭载NVIDIA Blackwell 架构，为游戏玩家和创作者带来全新玩法。RTX 50 系列具备强大的AI 算力，带来升级体验和更逼真的画面。

[3] 英伟达
GeForce RTX 50 系列GPU 为游戏玩家和创作者带来全新玩法。借助NVIDIA DLSS 4 实现性能倍增，以更快速度生成图像，并通过NVIDIA Studio 平台释放你的创造力。
```

#### ReActAgent

##### prompt

我们刚刚定义好了工具，也通过调用 LLM 跑通了工具调用，但这只能证明模型自身拥有工具意图和调用能力。而 ReAct 最重要的流程：`Thought => Action => Observation` 模型本身并没有强制遵守。  

我们需要在系统提示词中进行激活（ReActPrompt.py）：

```py
# ReAct 提示词模板
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""
```

##### ReActAgent 指导思想

我们之前定义的 `HelloAgentLLM` 类，通过 `think` 函数就可以进行单一提问单一回答。而ReAct 本质上也只是一个从 Thought 到 Obervation 的三状态机，每个状态也只是单一地调用 HelloAgentLLM 的 think 函数。

所以 ReActAgent 和 HelloAgentLLM 的关系是什么呢？ReActAgent 复杂循环状态机，内部包含 HelloAgentLLM 对象，用于运行最原子化 LLM 调用流程。

另外，ReActAgent 类也整合了前面定义的 ToolExecutor 和所有 Tool。

然后，仔细阅读 ReActAgent 怎么写的。

##### 初始化

```py
    def __init__(self, llm_client: HelloAgentLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []
```

构造函数参数：一个 HelloAgentLLM 对象、ToolExecutor 对象（需要注册工具）、步数用来防止无限返回

`history = []` 用来存储上下文。

##### run

第一步：拼接提示词。从 ToolExecutor 获取工具列表、自身上下文历史、用户输入的问题，一同加入 ReActPrompt，完成拼接。

```py
            # 1. 格式化提示词
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str
            )
```

第二步：调用 LLM 执行。使用 HelloAgentLLM 的 think 函数调用 LLM 输入拼接后提示词。

```py
            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break
```

第三步：使用自身成员函数 _parse_output 解析出 thought 和 action。这个函数就是简单地匹配关键字。

```py
            # 3. 解析LLM的输出
            thought, action = self._parse_output(response_text)
            
            if thought:
                print(f"思考: {thought}")

            if not action:
                print("警告:未能解析出有效的Action，流程终止。")
                break
```

第四步：

```py

```

第五步：

```py

```



##### 测试

```py
if __name__ == '__main__':
    
    # 1.构建 agent
    toolExecutor = ToolExecutor()
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search.search)

    agent = ReActAgent(
        HelloAgentLLM(),
        toolExecutor,
        5
    )

    # 2.提问
    print("请输入文本: ")
    userPrompt = input()

    agent.run(userPrompt)
```

结果：

```text
工具 'Search' 已注册。
请输入文本: 
英伟达股价多少
--- 第 1 步 ---
🧠 正在调用 deepseek-v4-pro 模型...
✅ 大语言模型响应成功:

思考: 用户询问英伟达的股价。我需要获取最新的股价信息。我将使用搜索引擎搜索“英伟达 股价”或“NVIDIA stock price”。
🎬 行动: Search[NVIDIA stock price today]
🔍 正在执行 [SerpApi] 网页搜索: NVIDIA stock price today
👀 观察: 英伟达，创立于1993年1月，是美国一家以设计和销售图形处理器为主的无厂半导体公司，总部设在加利福尼亚州的圣克拉拉，位于硅谷的中心位置。NVIDIA亦会设计游戏机图形处理器，例如第一代Xbox、PlayStation 3及任天堂Switch和任天堂Switch 2。
--- 第 2 步 ---
🧠 正在调用 deepseek-v4-pro 模型...
✅ 大语言模型响应成功:

思考: 用户询问英伟达的股价，属于实时信息。历史中的搜索未返回股价，需要重新搜索。我将直接用股票代码“NVDA”搜索最新价格。
🎬 行动: Search[NVDA 股价 今天]
🔍 正在执行 [SerpApi] 网页搜索: NVDA 股价 今天
👀 观察: 英伟达，创立于1993年1月，是美国一家以设计和销售图形处理器为主的无厂半导体公司，总部设在加利福尼亚州的圣克拉拉，位于硅谷的中心位置。NVIDIA亦会设计游戏机图形处理器，例如第一代Xbox、PlayStation 3及任天堂Switch和任天堂Switch 2。
--- 第 3 步 ---
🧠 正在调用 deepseek-v4-pro 模型...
✅ 大语言模型响应成功:

思考: 之前的搜索只返回了公司简介，没有提供股价。我需要尝试更精确的查询来获取英伟达今天的股票价格，例如包括股票代码和“today”等关键词。
🎬 行动: Search[NVDA stock price today per share]
🔍 正在执行 [SerpApi] 网页搜索: NVDA stock price today per share
👀 观察: 英伟达，创立于1993年1月，是美国一家以设计和销售图形处理器为主的无厂半导体公司，总部设在加利福尼亚州的圣克拉拉，位于硅谷的中心位置。NVIDIA亦会设计游戏机图形处理器，例如第一代Xbox、PlayStation 3及任天堂Switch和任天堂Switch 2。
--- 第 4 步 ---
🧠 正在调用 deepseek-v4-pro 模型...
✅ 大语言模型响应成功:

思考: 前三次搜索返回的都是英伟达公司的简介，没有得到具体的股价信息。可能是搜索引擎返回的结果被截断了，或者当前时间（2025年4月9日）的实时股价数据没有被正确抓取。我需要尝试更精确地搜索，比如直接搜索股票代码和具体日期，或者搜索财经网站。
🎬 行动: Search[NVDA 股价 2025-04-09]
🔍 正在执行 [SerpApi] 网页搜索: NVDA 股价 2025-04-09
👀 观察: [1] 英偉達(NVDA) 過往股價及數據
英偉達(NVDA) ; 2025年9月9日, 169.09, 170.98, 166.74, 170.76 ; 2025年9月8日, 167.55, 170.96, 167.35, 168.31 ...

[2] 英伟达公司(NVDA)_美股行情_今日股价与走势图_新浪财经
新浪财经为您提供英伟达公司(NVDA)实时行情数据。包括最新股价、今日涨跌幅、技术走势图、历史数据查询及最新相关新闻资讯，助力您把握美股市场投资机会。

[3] 英伟达(NVDA)股票股价_股价行情_讨论
雪球为您提供英伟达(NVDA)股票股价,实时行情,热门讨论,最新资讯,最新公告,公司资料,财务数据,财务指标分析等与英伟达(NVDA)股票相关的信息与服务.
--- 第 5 步 ---
🧠 正在调用 deepseek-v4-pro 模型...
✅ 大语言模型响应成功:

思考: 我需要获取英伟达（NVDA）的最新股价。虽然之前的搜索没有得到实时数据，但最后一条搜索返回了新浪财经和雪球的链接，可能包含行情。为了提供准确信息，我将再次搜索“NVDA 最新股价”或直接点击链接。由于无法直接点击链接，我将尝试搜索“NVDA 股价 2025-04-09 实时”来获取当日价格。
🎬 行动: Search[NVDA 股价 2025-04-09 实时]
🔍 正在执行 [SerpApi] 网页搜索: NVDA 股价 2025-04-09 实时
👀 观察: [1] 英伟达公司(NVDA)_美股行情_今日股价与走势图_新浪财经
新浪财经为您提供英伟达公司(NVDA)实时行情数据。包括最新股价、今日涨跌幅、技术走势图、历史数据查询及最新相关新闻资讯，助力您把握美股市场投资机会。

[2] 英伟达(NVDA)最新价格_行情_走势图
手机东方财富网是东方财富网的手机门户网站，为亿万用户打造一个手机联通世界的超级平台，提供7*24小时全面及时的财经中文资讯，内容覆盖国内外突发事件、股市资讯、全球新闻

[3] 英偉達(NVDA) 股價、新聞、報價和記錄
前收市價 195.74 · 開市 193.10 · 買入價 192.00 x 100 · 賣出價 191.85 x 200 · 今日波幅 191.22 - 195.55 · 52 週波幅 151.49 - 236.54 · 成交量 177,757,849 · 平均成交量 ...
已达到最大步数，流程终止。
```