import json
import os
import cohere
from dotenv import load_dotenv

# Load API key
load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(cohere_api_key)

# Konfigurasi
sargah_target = 26 # Ganti ini jika ingin memproses sargah_number 2, 3, dst
input_file = "../datasets/dataset.json"
output_file = "../datasets/dataset_with_embedding_cohere.json"

# Load original dataset
with open(input_file, "r", encoding="utf-8") as f:
    original_dataset = json.load(f)

# Load existing output file jika ada, jika tidak gunakan original
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
else:
    dataset = original_dataset

# Cari entri sargah_target yang belum punya embedding
records_to_embed = [
    record for record in dataset
    if record.get("sargah_number") == sargah_target and "embedding" not in record
]

# Kalau semua sudah di-embed, skip
if not records_to_embed:
    print(f"Tidak ada entri baru pada sargah_number {sargah_target} yang perlu diproses.")
else:
    texts = [record["text"] for record in records_to_embed]

    # Proses embedding
    response = cohere_client.embed(
        texts=texts,
        model="embed-v4.0"
    )
    embeddings = response.embeddings

    # Sisipkan kembali ke record
    embed_index = 0
    for record in dataset:
        if record.get("sargah_number") == sargah_target and "embedding" not in record:
            record["embedding"] = embeddings[embed_index]
            embed_index += 1

    # Simpan kembali
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Embedding untuk sargah_number {sargah_target} berhasil ditambahkan ke '{output_file}'.")
