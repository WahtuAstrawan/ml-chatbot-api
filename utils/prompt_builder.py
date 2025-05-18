from typing import List, Dict, Any


def build_query_enhancement_prompt(query: str) -> str:
    """
    Build prompt for enhancing the original query.

    Args:
        query: Original query string

    Returns:
        Complete prompt for query enhancement
    """
    return f"""
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
    """.strip()


def build_chat_prompt(query: str, contexts: List[Dict[str, Any]]) -> str:
    """
    Build prompt for generating the final response.

    Args:
        query: Original query string
        contexts: List of context entries

    Returns:
        Complete prompt for response generation
    """
    # Sort contexts by sargah and bait number
    sorted_contexts = sorted(contexts, key=lambda x: (x['sargah_number'], x['bait']))

    # Create context text in the correct order
    context_text = "\n".join(
        [f"(Sargah {c['sargah_number']} - {c['sargah_name']}, Bait {c['bait']}): {c['text']}" for c in sorted_contexts]
    )

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