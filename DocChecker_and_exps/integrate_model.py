from src.kg_gen import KGGen
import os
from dotenv import load_dotenv
from openai import OpenAI

from tests.utils.visualize_kg import visualize

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    # Initialize KGGen
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

    # Test texts
    with open("split1.txt", "r", encoding="utf-8") as file:
        text1 = file.read()
    with open("split2.txt", "r", encoding="utf-8") as file:
        text2 = file.read()

    # Generate individual graphs
    graph1 = kg.generate(
        input_data=text1,
        model="openai/gpt-4o-mini"
    )

    graph2 = kg.generate(
        input_data=text2,
        model="openai/gpt-4o-mini"
    )

    # Cluster the combined graph
    # clustered_graph = kg.cluster(
    #     combined_graph,
    #     model="openai/gpt-4o-mini",
    # )

    # Print results
    print("\nGraph 1:")
    print("Entities:", graph1.entities)
    print("Relations:", graph1.relations)
    print("Edges:", graph1.edges)

    print("\nGraph 2:")
    print("Entities:", graph2.entities)
    print("Relations:", graph2.relations)
    print("Edges:", graph2.edges)

    # # Aggregate the graphs
    combined_graph = kg.aggregate([graph1, graph2])

    print("\nCombined Graph:")
    print("Entities:", combined_graph.entities)
    print("Relations:", combined_graph.relations)
    print("Edges:", combined_graph.edges)

    # print("\nClustered Combined Graph:")
    # print("Entities:", clustered_graph.entities)
    # print("Relations:", clustered_graph.relations)
    # print("Edges:", clustered_graph.edges)
    # print("Entity Clusters:", clustered_graph.entity_clusters)
    # print("Edge Clusters:", clustered_graph.edge_clusters)

    print("*"*100)

    print("Graph1 Extracted Triples:")
    for head, relation, tail in graph1.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    print("*"*100)

    print("Graph2 Extracted Triples:")
    for head, relation, tail in graph2.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    print("*"*100)

    print("Combined Graph Extracted Triples:")
    for head, relation, tail in combined_graph.relations:
        print(f"({head}) -[{relation}]-> ({tail})")

    visualize(combined_graph, "E:/Document cleaning/kg-gen/tests/test_basic_2.html")
