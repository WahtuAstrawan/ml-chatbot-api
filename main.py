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

# Kelompokkan dataset 1 dokumen = 1 sargah
sargah_texts = {}
corpus = []
sargah_mapping = {}

for entry in data:
    sargah_num = entry['sargah_number']
    if sargah_num not in sargah_texts:
        sargah_texts[sargah_num] = []
    sargah_texts[sargah_num].append(entry['text'])

for sargah_num, texts in sorted(sargah_texts.items()):
    combined_text = " ".join(texts)
    corpus.append(combined_text)
    sargah_mapping[len(corpus) - 1] = sargah_num

# Encode teks dari corpus
corpus_embeddings = embedding_model.encode(corpus)

# Buat FAISS index
dimension = corpus_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# Retrieval function with FAISS
def retrieve_with_faiss(query: str, top_k=1):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Ambil semua bait dari sargah yang terpilih
    retrieved_entries = []
    for idx in indices[0]:
        sargah_num = sargah_mapping[idx]
        same_sargah_entries = [e for e in data if e['sargah_number'] == sargah_num]
        retrieved_entries.extend(same_sargah_entries)

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
7. Harap sertakan referensi bait dalam jawaban (seperti, "Desaratha adalah seorang raja (bait 34-38)") dalam jawaban; pastikan jawaban akurat dan mencerminkan konteks.
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

# Endpoint untuk menangani pertanyaan mengenai Kakawin Ramayana
@app.post("/chat")
async def chat_with_kakawin_ramayana(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Translate query ke English agar relevan dengan FAISS
    enhance_query = get_query(request.query)

    # Retrieve konteks yang relevan
    contexts = retrieve_with_faiss(enhance_query)

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