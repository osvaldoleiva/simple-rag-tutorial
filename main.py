import ollama
from typing import List, Tuple
import time

# Configuration
EMBEDDING_MODEL = 'nomic-embed-text'  # Good local embedding model
LANGUAGE_MODEL = 'llama3'  # Default local Llama3 model

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB: List[Tuple[str, List[float]]] = []

def load_dataset(file_path: str) -> List[str]:
    """Load and clean the dataset."""
    with open(file_path, 'r', encoding='utf-8') as file:
        # Remove empty lines and strip whitespace
        return [line.strip() for line in file if line.strip()]

def add_chunk_to_database(chunk: str) -> None:
    """Add a text chunk to the vector database with its embedding."""
    try:
        # Generate embedding for the chunk
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=chunk
        )
        embedding = response['embedding']
        VECTOR_DB.append((chunk, embedding))
    except Exception as e:
        print(f"Error processing chunk: {e}")

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def retrieve(query: str, top_n: int = 3) -> List[Tuple[str, float]]:
    """Retrieve top_n most similar chunks to the query."""
    try:
        # Get embedding for the query
        response = ollama.embeddings(
            model=EMBEDDING_MODEL,
            prompt=query
        )
        query_embedding = response['embedding']

        # Calculate similarities
        similarities = []
        for chunk, embedding in VECTOR_DB:
            similarity = cosine_similarity(query_embedding, embedding)
            similarities.append((chunk, similarity))

        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def main():
    print("Loading cat facts dataset...")
    dataset = load_dataset('cat-facts.txt')
    print(f'Loaded {len(dataset)} entries')

    print("Generating embeddings (this may take a while)...")
    for i, chunk in enumerate(dataset, 1):
        add_chunk_to_database(chunk)
        if i % 10 == 0 or i == len(dataset):
            print(f'Processed {i}/{len(dataset)} chunks')

    print("\nCat Facts Chatbot")
    print("Type 'quit' to exit")
    print("-" * 40)

    while True:
        try:
            input_query = input('\nAsk me a question about cats: ').strip()
            if input_query.lower() in ['quit', 'exit', 'q']:
                break

            if not input_query:
                continue

            print("\nSearching for relevant information...")
            retrieved_knowledge = retrieve(input_query)

            if not retrieved_knowledge:
                print("I couldn't find any relevant information. Please try another question.")
                continue

            print("\nGenerating response...")

            # Build the prompt with retrieved context
            context = '\n'.join([f'- {chunk}' for chunk, _ in retrieved_knowledge])
            prompt = f"""You are a helpful cat expert assistant. Answer the question based only on the following context:
            
            {context}
            
            Question: {input_query}
            
            If the context doesn't contain relevant information, say "I don't have enough information to answer that question about cats."
            Answer:"""

            # Generate response
            response = ollama.generate(
                model=LANGUAGE_MODEL,
                prompt=prompt,
                stream=True
            )

            # Stream the response
            print("\nChatbot:")
            for chunk in response:
                print(chunk['response'], end='', flush=True)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
