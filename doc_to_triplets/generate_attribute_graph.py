"""
暂时useless
"""
from knowledge_graph_maker import GraphMaker, Document
from knowledge_graph_maker import OpenAIClient
from openai import OpenAI
from dotenv import load_dotenv
import os
import json

# 加载环境变量
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_BASE_URL", None)

llm = OpenAIClient(
    model="gpt-4o-mini",
    temperature=0.1,
    top_p=0.9,
    max_tokens=1500
)

graph_maker = GraphMaker(llm_client=llm)

with open("split1.txt", "r", encoding="utf-8") as f:
    text1 = f.read()

docs = [Document(text=text1, metadata={"source": "split1.txt"})]

# text1 = """John Smith, a 45-year-old software engineer from San Francisco, works at TechNova Inc., a leading AI research company founded in 2012. He has been leading the Natural Language Processing (NLP) team there since 2018.
#
# John's wife, Emily Clark, is a 42-year-old physician currently working at St. Mary's Hospital. They got married in 2010 and have two children: Lily, aged 12, and Max, aged 9.
#
# In 2021, John gave a keynote speech at the International AI Conference held in Berlin, where he presented a paper on ethical machine learning. He also collaborates with Professor Alan Cheng from Stanford University on explainable AI research.
#
# During the COVID-19 pandemic in 2020, Emily volunteered in the ICU department for over six months, earning a community award for outstanding medical service.
#
# They currently live in a townhouse near Golden Gate Park and frequently donate to the San Francisco Education Fund, supporting STEM education for underprivileged students."""
#
# docs = [Document(text=text1, metadata={"source": "split1.txt"})]

edges = graph_maker.from_documents(docs)

# 输出识别出的图边
for e in edges:
    print(e.model_dump())

# ✅ Save to JSON file
output_path = "output_knowledge_graph.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(edges, f, ensure_ascii=False, indent=2)

print(f"\nKnowledge graph saved to: {output_path}")