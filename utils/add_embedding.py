import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Konfigurasi file
input_file = "./datasets/dataset.json"
output_file = "./datasets/dataset_with_embedding.json"
embedding_model_name = "all-MiniLM-L6-v2"

# Load SentenceTransformer model
model = SentenceTransformer(embedding_model_name)

# Load original dataset
with open(input_file, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Cari record yang belum memiliki embedding
records_to_embed = [record for record in dataset if "embedding" not in record]

# Jika semua sudah punya embedding
if not records_to_embed:
    print("All entries already contain embeddings.")
else:
    texts = [record["text"] for record in records_to_embed]
    print(f"Embedding {len(texts)} entries with model '{embedding_model_name}'...")

    # Buat embedding
    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    # Sisipkan embedding ke masing-masing record
    embed_index = 0
    for record in dataset:
        if "embedding" not in record:
            record["embedding"] = embeddings[embed_index]
            embed_index += 1

    # Simpan ke file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Successfully embedded {embed_index} entries and saved to '{output_file}'.")
