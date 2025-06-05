import json
import cohere
from dotenv import load_dotenv
import os

# Load environment variables (API key, etc)
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")

# Initialize Cohere client
cohere_client = cohere.Client(cohere_api_key)

# Load the JSON dataset
input_file = "../datasets/dataset.json"
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Extract texts for embedding
texts = [record["text"] for record in dataset]

# Generate embeddings using Cohere
response = cohere_client.embed(
    texts=texts,
    model="embed-english-v3.0",  # atau 'embed-multilingual-v3.0' untuk bahasa Indonesia
)
embeddings = response.embeddings  # List of list of floats

# Add embeddings to the dataset
for record, embedding in zip(dataset, embeddings):
    record["embedding"] = embedding  # Already a list, no need to convert

# Save the updated dataset
output_file = "../datasets/dataset_with_embedding_cohere.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset with embeddings saved to {output_file}")
