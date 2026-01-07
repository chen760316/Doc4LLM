"""
模型有最大的tokens数限制
"""
from src.kg_gen import KGGen
import os
from dotenv import load_dotenv
from tests.utils.visualize_kg import visualize
from openai import OpenAI

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))

    # Initialize client
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    try:
        # Test API connection
        models = client.models.list()
        print("API connection successful! Available models:")
        for model in models.data:
            print(f"- {model.id}")
    except Exception as e:
        print("API test failed:", str(e))
        if "401" in str(e):
            print("\n🔴 Authentication failed. Please verify:")
            print("1. Your API key is correct and active")
            print("2. Your account has sufficient credits")
            print("3. No network restrictions (proxies/firewalls)")

    with open("book_split.txt", "r", encoding="utf-8") as file:
        text = file.read()

    graph = kg.generate(
        input_data=text,
        model="openai/gpt-4o-mini"
    )

    # 可选：生成 HTML 可视化文件，仅用于人工验证
    visualize(graph, "E:/Document cleaning/kg-gen/tests/test_basic.html")
    print(graph)
    print("Extracted Triples:")
    for head, relation, tail in graph.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    # # 核心断言：检查生成结果非空
    # assert graph is not None
    # assert isinstance(graph, dict) or isinstance(graph, list)  # 根据你实际的返回类型调整
