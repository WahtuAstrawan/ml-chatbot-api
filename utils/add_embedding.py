import json
from sentence_transformers import SentenceTransformer

# Load the JSON dataset
input_file = "../datasets/dataset.json"
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Extract texts for embedding
texts = [record["text"] for record in dataset]

# Generate embeddings
embeddings = model.encode(texts, convert_to_numpy=True)

# Add embeddings to the dataset
for record, embedding in zip(dataset, embeddings):
    record["embedding"] = embedding.tolist()  # Convert numpy array to list for JSON serialization

# Save the updated dataset
output_file = "../datasets/dataset_with_embedding.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)

print(f"Dataset with embeddings saved to {output_file}")