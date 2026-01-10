# gemini_client_test.py

from google import genai
from google.genai.errors import APIError
from google.genai import types  # 引入 types 以处理配置，如果需要自定义 Base URL
import os
from dotenv import load_dotenv
import json
from typing import Optional


# --- 0. 模拟 generate 函数 (使用 Gemini 官方库实现) ---
def generate_with_gemini(
        client: genai.Client,
        model_name: str,
        user_message: str,
        system_message: Optional[str] = None
) -> str:
    """
    模拟 llm.generate 方法，使用 Gemini 官方库进行内容生成。
    """

    # Gemini API 将系统提示和用户消息都放入 contents 列表
    contents = []

    # 将系统提示作为用户角色的第一条消息（最常见做法）
    if system_message:
        contents.append({"role": "user", "parts": [{"text": system_message}]})
        # 注意：这里可能需要根据中转站的具体要求调整角色。
        # 对于官方 Google API，system_message 最好通过 config 参数传递。

    contents.append({"role": "user", "parts": [{"text": user_message}]})

    # 核心 API 调用
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={
            "temperature": 0.1,
            "max_output_tokens": 500,
            "response_mime_type": "application/json"  # 尝试让模型直接返回 JSON
        }
    )

    # 返回原始文本
    return response.text


# --- 1. 配置和 LLM 客户端 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 使用 Gemini 相关的环境变量
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("❌ 错误: GEMINI_API_KEY 未在 .env 文件或环境变量中找到！")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(api_key)})")

# --- 2. 初始化 Gemini Client ---
GEMINI_MODEL = "gemini-2.5-pro"  # 使用一个标准模型进行测试
print(f"\n--- 2. 正在初始化 Gemini Client ---")

try:
    http_options = None

    # 初始化客户端，API Key 会自动从环境变量获取或通过参数传入
    llm_client = genai.Client(
        api_key=api_key,
        http_options=http_options
    )
    print(f"✅ Gemini Client 初始化成功")

except Exception as e:
    print(f"❌ 错误: Gemini Client 初始化失败: {e}")
    exit()

# --- 3. 测试用的系统提示和用户输入 (模仿 JSON 响应要求) ---
TEST_SYSTEM_PROMPT = (
    "You are an expert in detecting contradictions. Respond with a single JSON object with two keys:\n"
    '{"test_status": true, "reason": "Connection confirmed."}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)

TEST_USER_MESSAGE = "Confirm the connection is working."

# --- 4. 核心：调用 Gemini API ---
print("\n--- 3. 正在调用 Gemini API 进行连接测试 ---")
print(f"👉 发送请求: '{TEST_USER_MESSAGE}'")

try:
    # 调用模拟的 generate 函数
    response_text = generate_with_gemini(
        client=llm_client,
        model_name=GEMINI_MODEL,
        user_message=TEST_USER_MESSAGE,
        system_message=TEST_SYSTEM_PROMPT
    )

    print("\n--- 4. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 尝试解析 JSON (与您的原始脚本逻辑相同)
    cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        response_json = json.loads(cleaned_response)

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


except APIError as e:
    print("\n❌ 致命错误: Gemini API 调用失败 (APIError)！")
    print(f"   错误信息: {e}")
    print("   请检查您的 API Key、Base URL 和模型名称。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {e}")