# 这里定义 get_weather(city: str) 工具，查询天气
# 这里有公共的 http api: wttr.in（ http://wttr.in/重庆）


import requests

def get_weather(city: str) -> str:

    # format参数代表让服务器响应的格式，j1就是JSON
    url = f"https://wttr.in/{city}?format=j1"

    try:
        # 使用 requests 包发起网络请求
        response = requests.get(url)
        # 检查是否成功，即状态码为200，若非200，则异常退出
        response.raise_for_status()
        # 解析返回 json 数据
        data = response.json()

        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"















