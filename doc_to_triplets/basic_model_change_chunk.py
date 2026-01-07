import os
from dotenv import load_dotenv
from src.kg_gen import KGGen
from transformers import AutoTokenizer
import tiktoken

# 滑动窗口参数
MAX_TOKENS = 512
STRIDE = 64


def split_text_by_token_limit(text, max_tokens=512, stride=64, model_name="gpt-4o-mini"):
    """Split text into chunks with token limits and overlap."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to a default encoding if the model isn't found
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - stride  # overlap
    return chunks


if __name__ == "__main__":
    load_dotenv()

    # Initialize KGGen with proper error handling
    try:
        kg = KGGen(
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base=os.getenv("OPENAI_BASE_URL")
        )
    except Exception as e:
        print(f"Failed to initialize KGGen: {e}")
        exit(1)

    # Read input file with error handling
    try:
        with open("book_chunk.txt", "r", encoding="utf-8") as file:
            full_text = file.read()
    except FileNotFoundError:
        print("Error: book_chunk.txt not found")
        exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    # Split text into chunks
    chunks = split_text_by_token_limit(full_text, max_tokens=MAX_TOKENS, stride=STRIDE)

    all_relations = []

    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i + 1}/{len(chunks)}...")
        print(f"Chunk length: {len(chunk)} characters")

        try:
            # Use a valid OpenAI model name
            graph = kg.generate(input_data=chunk, model="gpt-4o-mini")
            if graph and hasattr(graph, 'relations'):
                all_relations.extend(graph.relations)
                print(f"Extracted {len(graph.relations)} relations from this chunk")
        except Exception as e:
            print(f"Error processing chunk {i + 1}: {e}")
            continue

    # Print all extracted triples
    print("\nFinal Extracted Triples:")
    for i, (head, relation, tail) in enumerate(all_relations, 1):
        print(f"{i}. ({head}) -[{relation}]-> ({tail})")