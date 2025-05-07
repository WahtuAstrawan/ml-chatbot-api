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
from deep_translator import GoogleTranslator

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

class ChatRequest(BaseModel):
    query: str
    top_k: int = 1
    context_window: int = 3

# Encode teks dari dataset
corpus = [entry['text'] for entry in data]
corpus_embeddings = embedding_model.encode(corpus)

# Buat FAISS index
dimension = corpus_embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(corpus_embeddings))

# Fungsi untuk translate ke English
def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text


# Fungsi untuk mendapatkan konteks yang lebih luas (bait-bait sekitar)
def get_surrounding_context(selected_entries, window_size=3):
    expanded_context = []

    for entry in selected_entries:
        sargah_num = entry['sargah_number']
        bait_num = entry['bait']

        if entry not in expanded_context:
            expanded_context.append(entry)

        same_sargah_entries = [e for e in data if e['sargah_number'] == sargah_num]
        for nearby_entry in same_sargah_entries:
            if 0 < nearby_entry['bait'] - bait_num <= window_size:
                if nearby_entry not in expanded_context:
                    expanded_context.append(nearby_entry)

        for nearby_entry in same_sargah_entries:
            if 0 < bait_num - nearby_entry['bait'] <= window_size:
                if nearby_entry not in expanded_context:
                    expanded_context.append(nearby_entry)

        if bait_num <= window_size and sargah_num > 1:
            prev_sargah = sargah_num - 1
            prev_sargah_entries = [e for e in data if e['sargah_number'] == prev_sargah]
            if prev_sargah_entries:
                prev_sargah_entries.sort(key=lambda x: x['bait'], reverse=True)
                for nearby_entry in prev_sargah_entries[:window_size]:
                    if nearby_entry not in expanded_context:
                        expanded_context.append(nearby_entry)

        lower_baits = [e for e in same_sargah_entries if 0 < e['bait'] - bait_num <= window_size]
        if len(lower_baits) < window_size:
            next_sargah = sargah_num + 1
            next_sargah_entries = [e for e in data if e['sargah_number'] == next_sargah]
            if next_sargah_entries:
                next_sargah_entries.sort(key=lambda x: x['bait'])
                for nearby_entry in next_sargah_entries[:window_size - len(lower_baits)]:
                    if nearby_entry not in expanded_context:
                        expanded_context.append(nearby_entry)

    expanded_context.sort(key=lambda x: (x['sargah_number'], x['bait']))

    return expanded_context


# Retrieval function yang ditingkatkan dengan FAISS dan konteks tambahan
def retrieve_with_faiss_enhanced(query: str, top_k=1, context_window=3):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)

    # Dapatkan entri dari indeks yang ditemukan
    retrieved_entries = [data[i] for i in indices[0]]
    print(retrieved_entries)

    # Perluas dengan konteks sekitarnya
    expanded_entries = get_surrounding_context(retrieved_entries, context_window)

    return expanded_entries


# Prompt builder untuk Gemini AI
def build_prompt(query: str, contexts: List[dict]):
    # Urutkan konteks berdasarkan sargah dan nomor bait
    sorted_contexts = sorted(contexts, key=lambda x: (x['sargah_number'], x['bait']))

    # Buat teks konteks dengan urutan yang benar
    context_text = "\n".join(
        [f"(Sargah {c['sargah_number']} - {c['sargah_name']}, Bait {c['bait']}): {c['text']}" for c in sorted_contexts])

    return f"""
KONTEKS KAKAWIN RAMAYANA:
{context_text}

PERTANYAAN:
{query}

INSTRUKSI PENTING:
1. Berdasarkan HANYA konteks di atas, jawab pertanyaan dengan akurat dan spesifik (sesuaikan kata katanya agar mudah dimengerti dan jelas dalam bahasa indonesia).
2. Jika ada nama tokoh atau istilah yang disebutkan dalam konteks, sertakan namanya dengan lengkap dalam jawaban.
3. Ikuti alur narasi sesuai urutan bait.
4. Jika ada referensi ke tokoh sebelumnya (dengan kata "he", "she", dll), pastikan mengidentifikasi siapa tokohnya dengan nama dan konteks sebenarnya.
5. Jawaban harus singkat, tepat dan to the point. Tidak usah isi kalimat atau kata kata pengantar/tambahan seperti "berdasarkan konteks diatas", "semoga membantu", DLL (HANYA JAWABAN).
6. Jawaban harus berupa plain teks murni (paragraf) dan tidak memiliki format atau tambahan lain.
7. Jika pertanyaan sama sekali tidak relevan dengan konteks kakawin ramayananya. berikan jawaban "Maaf, pertanyaan Anda tidak relevan dengan Kakawin Ramayana."
""".strip()


# Endpoint utama untuk informasi aplikasi
@app.get("/")
def get_app_info():
    return {"App Name": "Chatbot ML App", "Description": "API untuk tanya jawab tentang Kakawin Ramayana"}


# Endpoint untuk menangani pertanyaan mengenai Kakawin Ramayana
@app.post("/chat")
async def chat_with_kakawin_ramayana(request: ChatRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")
    if request.context_window < 0:
        raise HTTPException(status_code=400, detail="context_window cannot be negative")

    # Translate query ke English agar relevan dengan FAISS
    translated_query = translate_to_english(request.query)

    # Retrieve konteks yang relevan
    contexts = retrieve_with_faiss_enhanced(translated_query, request.top_k, request.context_window)

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

        if "Maaf, pertanyaan Anda tidak relevan dengan Kakawin Ramayana" in response.text.strip():
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