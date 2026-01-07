from openai import OpenAI, APIError
import os
from dotenv import load_dotenv
import json

# --- 1. 加载环境变量 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# GROQ_API_KEY = "sk-Zu7OAYiAH1sEEXoAi6hyzH3oMjvlEtXFCvwRcSOHZNQQd2Wk"
# GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.getenv("Llama_API_KEY")
GROQ_BASE_URL = os.getenv("Llama_BASE_URL", None)
LLAMA3_MODEL = "llama-2-70b"  # Groq支持的模型

if not GROQ_API_KEY:
    print("❌ 错误: GROQ_API_KEY 未找到！")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(GROQ_API_KEY)})")
print(f"✅ Base URL 加载成功: {GROQ_BASE_URL}")

# --- 2. 初始化 OpenAI Client (Groq Endpoint) ---
try:
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url=GROQ_BASE_URL
    )
    print("✅ OpenAI Client 初始化成功")
except Exception as e:
    print(f"❌ Client 初始化失败: {e}")
    exit()

# --- 3. 构造消息并调用 Llama3 ---
TEST_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond ONLY with a JSON object that has one key 'test_status' "
    "with the boolean value 'true'. Do not include any markdown, extra text, or explanations."
)
USER_MESSAGE = "Confirm the API connection is working."

print("\n--- 3. 正在调用 Llama 3 API ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    response = client.chat.completions.create(
        model=LLAMA3_MODEL,
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
        print("\n❌ Groq API 调用失败 (APIError)！")
        print("   错误类型:", type(e).__name__)
        print("   错误详情:", e)

except Exception as e:
    print("\n❌ 发生未知错误！")
    print(f"   错误信息: {e}")
