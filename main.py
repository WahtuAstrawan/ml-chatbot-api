from typing import List
from google import genai
from fastapi import FastAPI
from fastapi import HTTPException
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import json
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Load dataset JSON
try:
    with open('dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not data:
        raise ValueError("Dataset is empty")
except FileNotFoundError:
    raise ValueError("dataset.json not found")
except json.JSONDecodeError:
    raise ValueError("Invalid JSON format in dataset.json")

app = FastAPI()
client = genai.Client(api_key=gemini_api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Dataset 1 dokumen = 1 bait
corpus = [entry['text'] for entry in data]

# Encode teks dari corpus
corpus_embeddings = embedding_model.encode(corpus)

# Buat FAISS index
dimension = corpus_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))


# Retrieval function with FAISS
def retrieve_with_faiss(query: str, top_k=3, context_size=10):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    retrieved_entries = []
    seen_baits = set()

    for idx in indices[0]:
        sargah_num = data[idx]['sargah_number']
        bait_num = data[idx]['bait']

        # Ambil semua bait dalam sargah yang sama
        same_sargah_entries = [e for e in data if e['sargah_number'] == sargah_num]
        same_sargah_entries.sort(key=lambda x: x['bait'])

        # Temukan indeks bait terpilih dalam sargah
        selected_entry_idx = next(i for i, e in enumerate(same_sargah_entries) if e['bait'] == bait_num)
        total_baits = len(same_sargah_entries)

        # Ambil context_size bait sebelum dan sesudah
        start_idx = max(0, selected_entry_idx - context_size)
        end_idx = min(total_baits, selected_entry_idx + context_size + 1)  # +1 karena inclusive

        # Tambahkan bait dari start_idx ke end_idx, hindari duplikasi
        for entry in same_sargah_entries[start_idx:end_idx]:
            bait_id = (entry['sargah_number'], entry['bait'])
            if bait_id not in seen_baits:
                retrieved_entries.append(entry)
                seen_baits.add(bait_id)

        # Jika kurang dari 2*context_size + 1 bait, tambahkan bait sekitar
        current_count = sum(1 for e in retrieved_entries if e['sargah_number'] == sargah_num)
        expected_count = 2 * context_size + 1
        if current_count < expected_count:
            remaining_needed = expected_count - current_count
            # Coba tambahkan bait sebelum start_idx, tapi tidak lintas sargah
            if sargah_num > 1:  # Hanya ambil jika bukan Sargah 1
                extra_start_idx = max(0, start_idx - remaining_needed)
                for entry in same_sargah_entries[extra_start_idx:start_idx]:
                    bait_id = (entry['sargah_number'], entry['bait'])
                    if bait_id not in seen_baits:
                        retrieved_entries.append(entry)
                        seen_baits.add(bait_id)
                        current_count += 1
                        if current_count >= expected_count:
                            break

            # Jika masih kurang, tambahkan bait setelah end_idx
            if current_count < expected_count:
                extra_end_idx = min(total_baits, end_idx + (expected_count - current_count))
                for entry in same_sargah_entries[end_idx:extra_end_idx]:
                    bait_id = (entry['sargah_number'], entry['bait'])
                    if bait_id not in seen_baits:
                        retrieved_entries.append(entry)
                        seen_baits.add(bait_id)
                        current_count += 1
                        if current_count >= expected_count:
                            break

    # Urutkan berdasarkan sargah dan nomor bait
    retrieved_entries.sort(key=lambda x: (x['sargah_number'], x['bait']))

    print(query)
    print(retrieved_entries[0])

    return retrieved_entries


# Prompt builder untuk Gemini AI
def build_prompt(query: str, contexts: List[dict]):
    # Urutkan konteks berdasarkan sargah dan nomor bait
    sorted_contexts = sorted(contexts, key=lambda x: (x['sargah_number'], x['bait']))

    # Buat teks konteks dengan urutan yang benar
    context_text = "\n".join(
        [f"(Sargah {c['sargah_number']} - {c['sargah_name']}, Bait {c['bait']}): {c['text']}" for c in sorted_contexts])

    return f"""
KONTEKS TERKAIT PERTANYAAN DI KAKAWIN RAMAYANA:
{context_text}

PERTANYAAN:
{query}

INSTRUKSI PENTING:
1. Jawab pertanyaan hanya berdasarkan konteks di atas, dengan akurat, spesifik, dan menggunakan bahasa Indonesia yang jelas serta mudah dipahami.
2. Sertakan nama lengkap tokoh atau istilah yang disebutkan dalam konteks, hindari ambiguitas (misalnya, ganti "he" atau "she" dengan nama tokoh yang jelas).
3. Ikuti alur narasi sesuai urutan bait, pastikan semua peristiwa relevan hingga konflik selesai (jika berlaku) dijelaskan secara kronologis.
4. Jika pertanyaan meminta motivasi, reaksi, atau tindakan karakter, jelaskan dengan lengkap termasuk pemicu tindakan, konsekuensi, dan nilai budaya (misalnya, dharma) jika relevan.
5. Jawaban harus singkat, tepat, dan langsung ke inti tanpa kalimat pengantar atau penutup seperti "berdasarkan konteks di atas" atau "semoga membantu."
6. Gunakan format teks murni (paragraf) tanpa penomoran, bullet, atau format lain.
7. Harap sertakan referensi sarggah dan bait dalam jawaban (seperti, "Desaratha adalah seorang raja (Prathamas Sarggah, bait 34-38)") dalam jawaban; pastikan jawaban akurat dan mencerminkan konteks.
8. Jawaban yang diberikan harap jangan berlebihan dan bertele-tele, cukup jawab apa yang ditanyakan oleh pertanyaan saja sesuai konteks.
9. Jika pertanyaan sama sekali tidak relevan dengan Kakawin Ramayana. berikan jawaban HANYA SEPERTI BERIKUT "Maaf, pertanyaan Anda tidak relevan dengan Kakawin Ramayana."
""".strip()

# Mendapatkan query yang lebih baik dengan Gemini
def get_query(query: str):
    prompt = f"""
    KONTEKS:
    Query berikut akan digunakan untuk retrieval augmented generation (RAG) dengan FAISS pada dataset Kakawin Ramayana. Dataset berisi teks dalam bahasa Inggris tentang narasi Kakawin Ramayana. Query awal kemungkinan dalam bahasa Indonesia.

    QUERY AWAL:
    {query}

    INSTRUKSI:
    1. Terjemahkan query ke bahasa Inggris jika belum dalam bahasa Inggris.
    2. Perjelas query agar lebih spesifik dan relevan dengan Kakawin Ramayana, tanpa mengubah makna aslinya.
    3. Pertahankan nama tokoh (misalnya, Dasaratha, Triwikrama, Wedha) dan istilah budaya tanpa perubahan.
    4. Jika query ambigu, tambahkan konteks untuk memperjelas query agar hasil pencarian dengan FAISS lebih relevan.
    5. Kembalikan hanya query yang telah diolah dalam bahasa Inggris, tanpa penjelasan atau teks tambahan.
    """

    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text

    except Exception:
        return query

# Endpoint utama untuk informasi aplikasi
@app.get("/")
def get_app_info():
    return {"App Name": "Chatbot ML App", "Description": "API untuk tanya jawab tentang Kakawin Ramayana"}

class ChatRequest(BaseModel):
    query: str
    top_k: int = 3
    context_size: int = 10

# Endpoint untuk menangani pertanyaan mengenai Kakawin Ramayana
@app.post("/chat")
async def chat_with_kakawin_ramayana(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")
    if request.context_size < 0:
        raise HTTPException(status_code=400, detail="context_size must be non-negative")

    # Translate query ke English agar relevan dengan FAISS
    enhance_query = get_query(request.query)

    # Retrieve konteks yang relevan
    contexts = retrieve_with_faiss(enhance_query, request.top_k, request.context_size)

    # Membangun prompt dengan konteks yang diambil
    prompt = build_prompt(request.query, contexts)

    # Menampilkan konteks yang relevan
    context_details = [
        {
            "sargah_number": c["sargah_number"],
            "sargah_name": c["sargah_name"],
            "bait": c["bait"],
            "sanskrit_text": c["sanskrit_text"],
            "text": c["text"]
        }
        for c in contexts
    ]

    # Gemini API request
    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)

        if "pertanyaan anda tidak relevan" in response.text.strip().lower():
            return {
                "response": response.text,
                "context": []
            }

        return {
            "response": response.text,
            "context": context_details
        }
    except Exception as e:
        return {"error": f"Failed to generate response: {str(e)}"}