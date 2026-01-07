"""
使用 Groq API 来测试 Llama 3 连接性是非常高效和便捷的选择。Groq 提供了两种调用方式：

Groq 官方 SDK (groq)：最推荐的方式，因为它支持所有 Groq 特性。
OpenAI 兼容模式 (openai)：如果您更习惯使用 openai 库的接口，也可以使用。

我将为您提供 Groq 官方 SDK 的测试脚本，这是最标准的做法。
"""
# llama3_groq_test.py

from groq import Groq, APIError
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any

# --- 1. 配置和客户端初始化 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 使用 Groq 相关的环境变量
GROQ_API_KEY = "sk-Zu7OAYiAH1sEEXoAi6hyzH3oMjvlEtXFCvwRcSOHZNQQd2Wk"

if not GROQ_API_KEY:
    print("❌ 错误: GROQ_API_KEY 未在 .env 文件或环境变量中找到！")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(GROQ_API_KEY)})")

# --- 2. 初始化 Groq Client ---
# 使用 Groq 托管的 Llama 3 模型
LLAMA3_MODEL = "llama-3-70b"
print(f"\n--- 2. 正在初始化 Groq Client (模型: {LLAMA3_MODEL}) ---")

try:
    # Groq Client 会自动从 GROQ_API_KEY 环境变量中读取密钥
    client = Groq()
    print(f"✅ Groq Client 初始化成功")

except Exception as e:
    print(f"❌ 错误: Groq Client 初始化失败: {e}")
    exit()

# --- 3. 核心：调用 Groq API ---

# 系统提示：要求模型严格返回 JSON
TEST_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond ONLY with a JSON object that has one key 'test_status' "
    "with the boolean value 'true'. Do not include any markdown, extra text, or explanations."
)
USER_MESSAGE = "Confirm the API connection is working."

print("\n--- 3. 正在调用 Llama 3 API 进行连接测试 ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    # 原始的 chat.completions.create 调用方式 (与 OpenAI 兼容)
    response = client.chat.completions.create(
        model=LLAMA3_MODEL,
        messages=[
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE}
        ],
        temperature=0.1,
        max_tokens=200,
        # Groq (Llama 3) 也支持 JSON 模式，但这里我们依赖 system prompt 来简化测试
    )

    # --- 4. API 响应解析 ---
    response_text = response.choices[0].message.content

    print("\n--- 4. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 尝试解析 JSON
    cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        response_json: Dict[str, Any] = json.loads(cleaned_response)

        # 检查特定的键值
        if response_json.get("test_status") is True:
            print("\n🎉 连接测试成功！🎉")
            print(f"API 返回了预期的 JSON 格式和值，延迟：{response.usage.total_time}s")
        else:
            print("\n⚠️ 警告: 连接成功，但 JSON 格式或内容不完全符合预期。")
            print(f"   解析到的 JSON: {response_json}")

    except json.JSONDecodeError:
        print("\n⚠️ 警告: 连接成功，但响应内容不是有效的 JSON 格式。")
        print("   原始响应可能需要额外的清理。")

except APIError as e:
    print("\n❌ 致命错误: Groq API 调用失败 (APIError)！")
    print(f"   错误信息: {e.status_code}: {e.message}")
    print("   请检查您的 GROQ_API_KEY 是否有效，以及模型名称是否拼写正确。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {e}")