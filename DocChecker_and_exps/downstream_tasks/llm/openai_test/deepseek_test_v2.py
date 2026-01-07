from openai import OpenAI, APIError
import os
from dotenv import load_dotenv
import json

# --- 1. 加载环境变量 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# DEEPSEEK_API_KEY = "sk-FWBZgZHIb9oq8EeI9XPXgJGW1D9szMLftDRoKhnvc4u2VDre"
# DEEPSEEK_BASE_URL = "https://api.vectorengine.ai/v1"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", None)
DEEPSEEK_MODEL = "deepseek-chat"  # 或 deepseek-coder、deepseek-llm-67b 等

if not DEEPSEEK_API_KEY:
    print("❌ 错误: DEEPSEEK_API_KEY 未找到！")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(DEEPSEEK_API_KEY)})")
print(f"✅ Base URL 加载成功: {DEEPSEEK_BASE_URL}")

# --- 2. 初始化 OpenAI Client (DeepSeek Endpoint) ---
try:
    client = OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )
    print("✅ OpenAI Client 初始化成功")
except Exception as e:
    print(f"❌ Client 初始化失败: {e}")
    exit()

# --- 3. 构造消息并调用 DeepSeek ---
TEST_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond ONLY with a JSON object that has one key 'test_status' "
    "with the boolean value 'true'. Do not include any markdown, extra text, or explanations."
)
USER_MESSAGE = "Confirm the API connection is working."

print("\n--- 3. 正在调用 DeepSeek API ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": TEST_SYSTEM_PROMPT},
            {"role": "user", "content": USER_MESSAGE}
        ],
        temperature=0.1,
        max_tokens=50,
    )
    response_text = response.choices[0].message.content.strip()
    print("\n--- 4. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 清理响应并解析为 JSON
    cleaned = response_text.replace("```json", "").replace("```", "").strip()
    try:
        resp_json = json.loads(cleaned)
        if resp_json.get("test_status") is True:
            print("\n🎉 连接测试成功！🎉")
        else:
            print("\n⚠️ JSON内容不完全符合预期: ", resp_json)
    except Exception as e:
        print("\n⚠️ 响应不是有效JSON。原始内容：")
        print(cleaned)

except APIError as e:
    print("\n❌ DeepSeek API 调用失败 (APIError)！")
    print("   错误类型:", type(e).__name__)
    print("   错误详情:", e)

except Exception as e:
    print("\n❌ 发生未知错误！")
    print(f"   错误信息: {e}")
