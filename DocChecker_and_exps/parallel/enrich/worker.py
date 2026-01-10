#!/usr/bin/env python3
import os, sys, pickle
from dotenv import load_dotenv
from src.kg_gen import KGGen
import wikipediaapi
from src.kg_gen.models import Graph
from typing import Set, Dict, List, Tuple

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

def main():
    load_dotenv()
    input_txt, output_pkl = sys.argv[1], sys.argv[2]

    with open(input_txt, "r", encoding="utf-8") as f:
        text = f.read()

    kg = KGGen(api_key=os.getenv("OPENAI_API_KEY"))
    graph = kg.generate(text, model="openai/gpt-5.2", context="Family relationships")

    # 可以选择在 worker 层提前 enrich
    top_entities = set(sorted(graph.entities)[:3])  # 示例前3实体
    enricher = WikipediaEnricher()
    enriched = enricher.enrich_graph(graph, top_entities)

    with open(output_pkl, "wb") as f:
        pickle.dump(enriched, f)
    print(f"✔ Worker finished: {output_pkl}")


if __name__ == "__main__":
    main()
