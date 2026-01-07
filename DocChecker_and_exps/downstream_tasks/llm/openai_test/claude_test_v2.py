# claude_openai_wrapper_test.py

from openai import OpenAI, APIError
import os
from dotenv import load_dotenv
import json
from typing import Dict, Any

# --- 1. 配置和客户端初始化 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 假设中转站要求使用 Anthropic 的 Key，但通过 OpenAI 的变量名传递
# API_KEY = os.getenv("ANTHROPIC_API_KEY") # 您的 Claude Key
# BASE_URL = os.getenv("ANTHROPIC_BASE_URL") # 您的中转站 Base URL (例如: https://api.yourproxy.com/v1)

# API_KEY = "sk-wkw4pjYrgJnI0dJTtiOcCSST20BBXnZAflWAsJvatGxLbkOD"
# BASE_URL = "https://api.vectorengine.ai/v1"
API_KEY = os.getenv("CLAUDE_API_KEY")
BASE_URL = os.getenv("CLAUDE_BASE_URL", None)
CLAUDE_MODEL_NAME = "claude-3-7-sonnet-20250219"

if not API_KEY:
    print("❌ 错误: ANTHROPIC_API_KEY 未在环境中设置！")
    exit()

if not BASE_URL:
    print("❌ 错误: OPENAI_BASE_URL (中转站地址) 未在环境中设置！")
    exit()

print(f"✅ 找到 API Key (长度: {len(API_KEY)})")
print(f"✅ 找到 Base URL: {BASE_URL}")

# --- 2. 初始化 OpenAI Client (指向中转站) ---
# 使用中转站用来映射到 Claude 模型的名称
print(f"\n--- 2. 正在初始化 OpenAI Client (目标模型: {CLAUDE_MODEL_NAME}) ---")

try:
    # 关键：使用中转站的 Base URL 和 Claude 的 Key 初始化 OpenAI Client
    client = OpenAI(
        base_url=BASE_URL,
        api_key=API_KEY
    )
    print(f"✅ OpenAI Client 初始化成功")

except Exception as e:
    print(f"❌ 错误: Client 初始化失败: {e}")
    exit()

# --- 3. 核心：调用 OpenAI 接口 (通过中转站调用 Claude) ---

# 系统提示：要求模型严格返回 JSON
SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond with a single JSON object with two keys:\n"
    '{"test_status": true, "reason": "Connection confirmed."}\n'
    "Respond with only the JSON object—no explanations, no markdown, no extra text."
)
USER_MESSAGE = "Confirm the API connection is working."

print("\n--- 3. 正在调用 Claude API (OpenAI 兼容模式) 进行连接测试 ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    # 使用 OpenAI 兼容的 chat.completions.create 方法
    response = client.chat.completions.create(
        model=CLAUDE_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT}, # 兼容模式下 system 放在 messages 中
            {"role": "user", "content": USER_MESSAGE}
        ],
        temperature=0.1,
        max_tokens=500
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
        else:
            print("\n⚠️ 警告: 连接成功，但 JSON 格式或内容不完全符合预期。")
            print(f"   解析到的 JSON: {response_json}")

    except json.JSONDecodeError:
        print("\n⚠️ 警告: 连接成功，但响应内容不是有效的 JSON 格式。")

except APIError as e:
    print("\n❌ 致命错误: Claude API (通过中转站) 调用失败 (APIError)！")
    print(f"   错误信息: {e.status_code}: {e.message}")
    print("   请检查您的 BASE_URL (中转站地址) 和 CLAUDE_MODEL_NAME 是否在中转站配置正确。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {e}")