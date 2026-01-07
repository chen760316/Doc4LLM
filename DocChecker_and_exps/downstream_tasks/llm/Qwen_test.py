# qwen_client_test_revised.py

from dashscope import Generation
# ❌ 不再需要导入 Common
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any

# --- 1. 配置和 SDK 初始化 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 使用 DASHSCOPE 相关的环境变量
QWEN_API_KEY = os.getenv("DASHSCOPE_API_KEY")

if not QWEN_API_KEY:
    print("❌ 错误: DASHSCOPE_API_KEY 未在 .env 文件或环境变量中找到！")
    exit()

# ⚠️ 关键修正：直接设置 dashscope 模块的全局 api_key 属性
# 这取代了旧版使用 Common.api_key = QWEN_API_KEY 的方式。
Generation.api_key = QWEN_API_KEY

print(f"✅ API Key 加载成功 (长度: {len(QWEN_API_KEY)})")
print("✅ DASHSCOPE SDK 初始化成功")

# --- 2. 核心：调用 Qwen API ---

# 使用一个稳定且支持聊天的模型
QWEN_MODEL = "qwen-max"
print(f"\n--- 2. 正在调用 Qwen API (模型: {QWEN_MODEL}) ---")

# 模仿您原始脚本的 JSON 响应要求
TEST_SYSTEM_PROMPT = (
    "你是一位专业的助手。请返回一个包含两个键的 JSON 对象：\n"
    '{"test_status": true, "reason": "Connection confirmed."}\n'
    "只回复 JSON 对象，不包含任何解释、Markdown 或额外文本。"
)
USER_MESSAGE = "确认 API 连接是否正常工作。"

print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    # 构造 messages 列表
    messages = [
        {"role": "system", "content": TEST_SYSTEM_PROMPT},
        {"role": "user", "content": USER_MESSAGE}
    ]

    # 原始的 Generation.call 调用方式
    response = Generation.call(
        model=QWEN_MODEL,
        messages=messages,
        result_format='message', # 要求返回 message 格式
        # Qwen 模型的参数设置
        temperature=0.1,
        max_tokens=500
    )

    # --- 3. API 响应解析 ---
    # Qwen 的回复文本在 response.output.choices[0].message.content 中
    response_text = response.output.choices[0].message.content

    print("\n--- 3. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 尝试解析 JSON
    cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        response_json: Dict[str, Any] = json.loads(cleaned_response)

        # 检查特定的键值
        if response_json.get("test_status") is True:
            print("\n🎉 连接测试成功！🎉")
            print("API 返回了预期的 JSON 格式和值。")
        else:
            print("\n⚠️ 警告: 连接成功，但 JSON 格式或内容不完全符合预期。")
            print(f"   解析到的 JSON: {response_json}")

    except json.JSONDecodeError:
        print("\n⚠️ 警告: 连接成功，但响应内容不是有效的 JSON 格式。")
        print("   原始响应可能需要额外的清理。")

except Exception as e:
    # 捕获所有可能的 DashScope 异常
    print("\n❌ 致命错误: Qwen API 调用失败！")
    print(f"   错误信息: {e}")
    print("   请检查您的 API Key、网络连接和模型名称。")