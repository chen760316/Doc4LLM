# claude_raw_test.py

from anthropic import Anthropic, APIError
import os
from dotenv import load_dotenv
import json

# --- 1. 配置和 LLM 客户端 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 使用 Anthropic 相关的环境变量
# CLAUDE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# CLAUDE_BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None) # 用于自定义中转站或端点
CLAUDE_API_KEY = "sk-LUEnFoNtoPojwa4gUZj5MyA4ahyFlzY7KrboSBUQL8gqAoPn"

if not CLAUDE_API_KEY:
    print("❌ 错误: ANTHROPIC_API_KEY 未在 .env 文件或环境变量中找到！")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(CLAUDE_API_KEY)})")

# --- 2. 初始化 Anthropic Client ---
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
print(f"\n--- 2. 正在初始化 Anthropic Client ---")

try:
    # 原始方式初始化客户端
    client = Anthropic(
        api_key=CLAUDE_API_KEY
    )
    print(f"✅ Anthropic Client 初始化成功 (模型: {CLAUDE_MODEL})")

except Exception as e:
    print(f"❌ 错误: Anthropic Client 初始化失败: {e}")
    exit()

# --- 3. 核心：调用 Claude API ---

# 模仿您原始脚本的 JSON 响应要求
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond with a single JSON object with two keys:\n"
    '{"test_status": true, "reason": "Connection confirmed."}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)
USER_MESSAGE = "Confirm the API connection is working."

print("\n--- 3. 正在调用 Claude API 进行连接测试 ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    # 原始的 messages.create 调用方式
    response = client.messages.create(
        model=CLAUDE_MODEL,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_MESSAGE}
        ],
        temperature=0.1,
        max_tokens=500
    )

    # --- 4. API 响应解析 ---
    # Claude 的回复文本在 response.content[0].text 中
    response_text = response.content[0].text

    print("\n--- 4. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 尝试解析 JSON
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
    print("\n❌ 致命错误: Claude API 调用失败 (APIError)！")
    print(f"   错误信息: {e}")
    print("   请检查您的 API Key、Base URL 和模型名称。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {e}")