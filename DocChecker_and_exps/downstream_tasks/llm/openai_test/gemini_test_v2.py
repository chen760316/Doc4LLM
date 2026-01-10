# openai_proxy_test.py

from openai import OpenAI
from openai import APIError  # 用于捕获 API 错误
import os
from dotenv import load_dotenv
import json

# --- 1. 配置和客户端初始化 ---
print("--- 1. 正在加载配置和检查 API Key/Base URL ---")
load_dotenv()

# 使用 OpenAI 兼容的变量名称
# API_KEY = "sk-g85vPvIOYqokrjMW7kaRVhmLzHPwmkF7q2CSaKdxv8IepaiU"
# BASE_URL = "https://api.vectorengine.ai/v1"
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ 错误: OPENAI_API_KEY 未在环境中设置！")
    exit()

if not BASE_URL:
    print("❌ 错误: OPENAI_BASE_URL 未在环境中设置！")
    exit()

print(f"✅ 找到 API Key (长度: {len(API_KEY)})")
print(f"✅ 找到 Base URL: {BASE_URL}")

# --- 2. 初始化 OpenAI 兼容客户端 ---
# 注意: 您可以使用中转站支持的任何模型，这里继续使用 gemini-2.5-pro
MODEL_NAME = "gemini-2.5-pro"
print(f"\n--- 2. 正在初始化 OpenAI Client (目标模型: {MODEL_NAME}) ---")

try:
    # 初始化 OpenAI Client，连接到中转站
    client = OpenAI(
        api_key=API_KEY
    )
    print("✅ OpenAI Client 初始化成功")

except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    exit()

# --- 3. API 调用和测试 ---
TEST_SYSTEM_PROMPT = (
    "You are an expert. Respond with a single JSON object with one key 'test_status' "
    "with the value 'connected'."
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)
TEST_USER_MESSAGE = "Confirm the API connection is working."

print(f"\n--- 3. 正在调用 {MODEL_NAME} 进行连接测试 ---")
print(f"👉 提示词: '{TEST_USER_MESSAGE}'")

try:
    # 核心 API 调用，使用 OpenAI 的标准参数
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user", "content": TEST_USER_MESSAGE}
        ],
        temperature=0.1,
        max_tokens=500,
    )

    # --- 4. 结果输出和解析 ---
    print("\n--- 4. API 响应解析 ---")

    # 检查响应结构并提取文本
    if response.choices and response.choices[0].message:
        response_text = response.choices[0].message.content
        print("✅ 原始响应:")
        print(response_text)

        # 尝试解析 JSON
        cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
        try:
            response_json = json.loads(cleaned_response)

            # 检查特定的键值
            if response_json.get("test_status") == "connected":
                print("\n🎉 连接测试成功！🎉")
            else:
                print("\n⚠️ 警告: 连接成功，但 JSON 格式或内容不完全符合预期。")
                print(f"   解析到的 JSON: {response_json}")

        except json.JSONDecodeError:
            print("\n⚠️ 警告: 连接成功，但响应内容不是有效的 JSON 格式。")
            print("   模型未能严格遵守 JSON 格式要求，可能需要调整系统提示。")

    else:
        print("⚠️ 警告: 响应结构不完整或为空。")
        # 打印完整的响应对象以供调试
        print(f"完整响应对象: {response.model_dump_json(indent=2)}")


except APIError as e:
    print("\n❌ 致命错误: API 调用失败 (APIError)！")
    print(f"   错误信息: {e}")
    print("   请特别检查：API Key 是否有效、网络连接和模型名称是否在中转站可用。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {e}")