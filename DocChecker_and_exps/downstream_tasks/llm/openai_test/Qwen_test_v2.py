import openai
from dotenv import load_dotenv
import os

print("--- 1. 正在加载配置 ---")
load_dotenv()

# QWEN_API_KEY = "sk-FWBZgZHIb9oq8EeI9XPXgJGW1D9szMLftDRoKhnvc4u2VDre"
# QWEN_OPENAI_BASE_URL = "https://api.vectorengine.ai/v1"
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_OPENAI_BASE_URL = os.getenv("QWEN_BASE_URL", None)
QWEN_MODEL = "qwen-max"

print(f"✅ API Key 加载成功 (长度: {len(QWEN_API_KEY)})")
print(f"✅ Base URL 加载成功: {QWEN_OPENAI_BASE_URL}")

print(f"\n--- 2. 正在配置 OpenAI 库 ---")
print(f"目标模型: {QWEN_MODEL}")
print(f"Base URL: {QWEN_OPENAI_BASE_URL}")

client = openai.OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_OPENAI_BASE_URL
)

print("✅ OpenAI 库配置成功")

TEST_SYSTEM_PROMPT = "你是一个简单的助手。请回复 'API Connection Successful.'，不要回复任何其他内容。"
USER_MESSAGE = "测试连接。"

print("\n--- 3. 正在调用 Qwen API (简单文本测试) ---")
print(f"👉 发送请求: '{USER_MESSAGE}'")

try:
    response = client.chat.completions.create(
        model=QWEN_MODEL,
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

    if response_text:
        print(response_text)
        print("\n🎉 文本生成测试成功！🎉")
    else:
        print("❌ 模型返回了空内容。")
        print("\n⚠️ 致命警告: 基础文本生成失败。问题可能在于 Base URL 服务端或 API Key。")
except Exception as e:
    print("\n❌ 致命错误: 发生未知错误！")
    print(f"   错误信息: {type(e).__name__}: {e}")
