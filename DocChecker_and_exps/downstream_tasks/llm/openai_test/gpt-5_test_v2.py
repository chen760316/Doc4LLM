# test_gpt_connection.py

from knowledge_graph_maker import OpenAIClient
import os
from dotenv import load_dotenv
import json

# --- 1. 配置和 LLM 客户端 ---
print("--- 1. 正在加载配置 ---")
load_dotenv()

# 请确保您的 .env 文件配置正确
# 这里的 os.environ 是为了确保即使 dotenv 失败，后续的 client 也能尝试使用环境中的变量
# 但在实际环境中，OpenAIClient 应该直接读取 os.getenv 的值。
# 这里我们仅作检查和设置：
try:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", None)
    if not api_key:
        print("❌ 错误: OPENAI_API_KEY 未在 .env 文件或环境变量中找到！")
        exit()
except Exception as e:
    print(f"❌ 错误: 环境变量加载失败: {e}")
    exit()

print(f"✅ API Key 加载成功 (长度: {len(api_key)})")
if base_url:
    print(f"✅ Base URL: {base_url}")
else:
    print("✅ 正在使用默认 OpenAI API Base URL")

# --- 2. 初始化 LLM 客户端 ---
print("\n--- 2. 正在初始化 OpenAIClient ---")
try:
    # 注意: 模型名称 "gpt-4.1-mini" 可能是自定义或占位的，
    # 实际测试中请使用有效的模型，例如 "gpt-3.5-turbo" 或 "gpt-4o-mini"
    llm = OpenAIClient(
        model="gpt-5",
        temperature=0.1,
        top_p=0.9,
        max_tokens=500  # 减少 token 数量，只需要简短的 JSON 响应
    )
    print(f"✅ OpenAIClient 初始化成功")
except Exception as e:
    print(f"❌ 错误: OpenAIClient 初始化失败: {e}")
    exit()

# --- 3. 测试用的系统提示和用户输入 ---
TEST_SYSTEM_PROMPT = (
    "You are a helpful assistant. Respond with a JSON object that has one key 'test_status' "
    "with the value 'connected'."
)

TEST_USER_MESSAGE = "Say 'Hello' and confirm the connection is working."

# --- 4. 核心：调用 GPT API ---
print("\n--- 3. 正在调用 GPT API 进行连接测试 ---")
print(f"👉 发送请求: '{TEST_USER_MESSAGE}'")

try:
    response_text = llm.generate(
        user_message=TEST_USER_MESSAGE,
        system_message=TEST_SYSTEM_PROMPT
    )

    print("\n--- 4. API 响应解析 ---")
    print("✅ 原始响应:")
    print(response_text)

    # 尝试解析 JSON
    cleaned_response = response_text.strip().replace("```json", "").replace("```", "").strip()
    try:
        response_json = json.loads(cleaned_response)

        # 检查特定的键值
        if response_json.get("test_status") == "connected":
            print("\n🎉 连接测试成功！🎉")
            print("API 返回了预期的 JSON 格式和值。")
        else:
            print("\n⚠️ 警告: 连接成功，但 JSON 格式或内容不完全符合预期。")
            print(f"   解析到的 JSON: {response_json}")

    except json.JSONDecodeError:
        print("\n⚠️ 警告: 连接成功，但响应内容不是有效的 JSON 格式。")
        print("   原始响应可能需要额外的清理。")


except Exception as e:
    print("\n❌ 致命错误: GPT API 调用失败！")
    print(f"   错误信息: {e}")
    print("   请检查您的 API Key、网络连接、Base URL (如果使用代理/私有部署) 或模型名称。")