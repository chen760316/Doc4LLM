# worker.py
import os
import sys
import pickle
from dotenv import load_dotenv
from src.kg_gen import KGGen

if __name__ == "__main__":
    load_dotenv()
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"), api_base=os.getenv("OPENAI_BASE_URL"))
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
    graph = kg.generate(text, model="openai/gpt-4o-mini", context="Family relationships")
    with open(output_file, "wb") as f:
        pickle.dump(graph, f)
