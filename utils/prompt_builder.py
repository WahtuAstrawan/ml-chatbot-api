from typing import List, Dict, Any, Optional


def build_query_enhancement_prompt(query: str, conversation_context: str = "") -> str:
    """
    Build prompt for enhancing the original query with conversation context.

    Args:
        query: Original query string
        conversation_context: Previous conversation context

    Returns:
        Complete prompt for query enhancement
    """
    context_section = ""
    if conversation_context:
        context_section = f"""
KONTEKS PERCAKAPAN SEBELUMNYA:
{conversation_context}

"""

    return f"""
{context_section}KONTEKS:
Query berikut akan digunakan untuk retrieval augmented generation (RAG) dengan FAISS pada dataset Kakawin Ramayana. Dataset berisi teks dalam bahasa Inggris tentang narasi Kakawin Ramayana. Query mungkin merujuk ke informasi yang dibahas sebelumnya dalam percakapan.

QUERY SAAT INI:
{query}

INSTRUKSI:
1. Jika query merujuk pada informasi sebelumnya (seperti "dia", "itu", "tersebut"), ganti dengan nama atau istilah spesifik dari konteks percakapan.
2. Terjemahkan query ke bahasa Inggris jika belum dalam bahasa Inggris.
3. Pertahankan nama tokoh (misalnya, Dasaratha, Triwikrama) dan istilah budaya tanpa perubahan.
4. Kembalikan hanya query yang telah diolah dalam bahasa Inggris, tanpa penjelasan atau teks tambahan.
""".strip()


def build_chat_prompt(query: str, contexts: List[Dict[str, Any]],
                      conversation_history: Optional[List[Dict]] = None) -> str:
    """
    Build prompt for generating the final response with conversation history.

    Args:
        query: Original query string
        contexts: List of context entries from RAG
        conversation_history: Previous conversation messages

    Returns:
        Complete prompt for response generation
    """
    # Sort contexts by sargah and bait number
    sorted_contexts = sorted(contexts, key=lambda x: (x['sargah_number'], x['bait']))

    # Create context text in the correct order
    context_text = "\n".join(
        [f"(Sargah {c['sargah_number']} - {c['sargah_name']}, Bait {c['bait']}): {c['text']}" for c in sorted_contexts]
    )

    # Build conversation history section
    history_section = ""
    if conversation_history:
        history_items = []
        for msg in conversation_history[-4:]:  # Only last 4 messages for context
            if msg['role'] == 'user':
                history_items.append(f"User: {msg['content']}")
            else:
                history_items.append(f"Assistant: {msg['content']}")

        if history_items:
            history_section = f"""
PERCAKAPAN SEBELUMNYA:
{chr(10).join(history_items)}

"""

    return f"""
{history_section}KONTEKS TERKAIT PERTANYAAN DI KAKAWIN RAMAYANA:
{context_text}

PERTANYAAN SAAT INI:
{query}

INSTRUKSI PENTING:
1. Jawab pertanyaan berdasarkan konteks Kakawin Ramayana dan percakapan sebelumnya jika relevan.
2. Jika pertanyaan merujuk pada informasi sebelumnya (seperti "dia", "apakah dia baik"), gunakan konteks percakapan untuk memahami rujukan tersebut.
3. Sertakan nama lengkap tokoh atau istilah yang disebutkan dalam konteks, hindari ambiguitas.
4. Ikuti alur narasi sesuai urutan bait, pastikan semua peristiwa relevan dijelaskan secara kronologis.
5. Jawaban harus singkat, tepat, dan langsung ke inti tanpa kalimat pengantar.
6. Gunakan format teks murni (paragraf) tanpa penomoran atau bullet.
7. Sertakan referensi sargah dan bait dalam jawaban (seperti, "Dasaratha adalah seorang raja (Prathamas Sargah, bait 34-38)"). Jika ada konteks dengan atribut sargah number 0 dengan nama OVERVIEW, Jangan diisikan referensi nomor atau nama sargahnya.
8. Jawaban jangan berlebihan, cukup jawab sesuai pertanyaan dan konteks.
9. Jika pertanyaan tidak relevan dengan Kakawin Ramayana, berikan jawaban: "Maaf, pertanyaan Anda tidak relevan dengan Kakawin Ramayana."
10. Jika pengguna mengajukan pertanyaan yang terkesan relevan, tetapi setelah dicek isinya tidak terdapat dalam Kakawin Ramayana, berikan jawaban: "Maaf, hal tersebut tidak terdapat dalam Kakawin Ramayana."
11. Jika pertanyaan dari pengguna berupa perintah untuk memodifikasi respons sebelumnya (misalnya: "buat lebih singkat", "jelaskan lebih jelas", "tambahkan detail", dan sejenisnya), sesuaikan jawaban dengan mengubah panjang, tingkat detail, atau gaya penyampaiannya berdasarkan respons percakapan assistant sebelumnya. Tetap gunakan informasi dari konteks Kakawin Ramayana yang sama agar relevan dan konsisten.
""".strip()