#!/usr/bin/env python3
"""
Knowledge Graph enrichment with Wikipedia
为了在领域中包含专门知识，DocChecker在KG可用时将图G和知识图KG连接起来。
G仅在用户指定集合中的实体上执行联接，即，它只从KG中提取用户感兴趣的相关知识。这就产生了一个丰富的图，也表示为G
先打印出graph1和graph2中的前部实体，由用户选择后从wiki知识库中进行图增强
"""
import os
import re
from dotenv import load_dotenv
import wikipediaapi
from typing import Set, Dict, List, Tuple
from collections import defaultdict
from src.kg_gen import KGGen
from src.kg_gen.models import Graph
from tests.utils.visualize_kg import visualize


# Wikipedia Enricher Class
class WikipediaEnricher:
    def __init__(self, lang: str = "en"):
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent="DocChecker/1.0 (your-email@example.com)"
        )

    def enrich_graph(self, graph: Graph, entities_of_interest: Set[str]) -> Graph:
        """Enrich the graph with Wikipedia knowledge for specified entities"""
        new_relations = set(graph.relations)
        new_edges = set(graph.edges)
        new_entities = set(graph.entities)

        for entity in entities_of_interest:
            if entity not in graph.entities:
                continue

            page = self.wiki.page(entity)
            if not page.exists():
                continue

            # Add Wikipedia categories as new relations
            for category in page.categories:
                clean_cat = category.replace("Category:", "").strip()
                new_relations.add((entity, "wikipedia_category", clean_cat))
                new_edges.add("wikipedia_category")
                new_entities.add(clean_cat)

            # Add links between entities that exist in our graph
            for linked_entity in page.links:
                if linked_entity in graph.entities:
                    new_relations.add((entity, "related_to", linked_entity))
                    new_edges.add("related_to")

        return Graph(
            entities=new_entities,
            relations=new_relations,
            edges=new_edges,
            entity_clusters=graph.entity_clusters,
            edge_clusters=graph.edge_clusters
        )


# Helper Functions
def locate_entities(text: str, entity: str) -> List[Tuple[int, int]]:
    """Find all occurrences of an entity in text"""
    return [(m.start(), m.end()) for m in re.finditer(re.escape(entity), text)]


def find_matched_entities(entities1: List[str], entities2: List[str]) -> List[Tuple[str, str]]:
    """Find entities with the same name in both graphs"""
    return [(e1, e2) for e1 in entities1 for e2 in entities2 if e1 == e2]


if __name__ == "__main__":
    # Load environment and initialize KGGen
    load_dotenv()
    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"))

    # Read input texts
    with open("split1.txt", "r", encoding="utf-8") as f:
        text1 = f.read()
    with open("split2.txt", "r", encoding="utf-8") as f:
        text2 = f.read()

    # Generate knowledge graphs
    graph1 = kg.generate(text1, model="openai/gpt-5.2", context="Family relationships")
    graph2 = kg.generate(text2, model="openai/gpt-5.2", context="Family relationships")

    # 🔍 打印原始实体
    print("\nGraph1 Entities:")
    print(sorted(graph1.entities))
    print("\nGraph2 Entities:")
    print(sorted(graph2.entities))

    # Combine graphs
    combined_graph = kg.aggregate([graph1, graph2])

    # Add connections between matched entities
    matched_pairs = find_matched_entities(graph1.entities, graph2.entities)
    for head, tail in matched_pairs:
        combined_graph.relations.add((head, "matched_entity", tail))
        combined_graph.edges.add("matched_entity")

    # ✅ 手动指定需要增强的实体（可以从上面的打印结果中挑）
    entities_of_interest = {"Christmas", "Hamlet", "City clocks", "Scrooge", "St. Paul's Churchyard"}  # ← 你感兴趣的实体

    # Enrich with Wikipedia
    enricher = WikipediaEnricher()
    enriched_graph = enricher.enrich_graph(combined_graph, entities_of_interest)

    # Visualize the final graph
    visualize(enriched_graph, "enriched_kg.html")

    # Print summary
    print(f"\nOriginal graph had {len(combined_graph.entities)} entities and {len(combined_graph.relations)} relations")
    print(f"After Wikipedia enrichment: {len(enriched_graph.entities)} entities and {len(enriched_graph.relations)} relations")
    print(f"Visualization saved to enriched_kg.html")
