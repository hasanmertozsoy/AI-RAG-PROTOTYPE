import os
import tempfile
import streamlit as st
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

load_dotenv()
COLLECTION_NAME = "company_docs"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_RETRIEVAL_K = 5
TEMPERATURE = 0.0

FILE_ICONS = {
    "pdf": "📄",
    "docx": "📝",
    "txt": "📃",
    "md": "🗒️",
}

st.set_page_config(
    page_title="Şirket İçi AI Asistan",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@400;500&display=swap');
:root { --ink: #1a1714; --paper: #f5f0e8; --cream: #ede8dc; --rust: #c45c2e; --sage: #4a6741; --dust: #9b9589; --warm-white: #faf7f2; }
html, body, [class*="css"] { font-family: 'Fraunces', Georgia, serif; background-color: var(--paper); color: var(--ink); }
.stApp { background-color: var(--paper); }
section[data-testid="stSidebar"] { background-color: var(--ink); color: var(--warm-white); }
section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] small { color: var(--warm-white) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] { background-color: var(--ink) !important; border: 1px dashed rgba(245, 240, 232, 0.4) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] div, section[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] small { color: var(--warm-white) !important; opacity: 1 !important; font-weight: 500 !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button { background-color: var(--warm-white) !important; border: 1px solid var(--ink) !important; }
section[data-testid="stSidebar"] [data-testid="stFileUploader"] button, section[data-testid="stSidebar"] [data-testid="stFileUploader"] button * { color: var(--ink) !important; font-weight: 600 !important; font-family: 'DM Mono', monospace; }
section[data-testid="stSidebar"] .stButton > button { background-color: var(--warm-white) !important; border: 1px solid var(--ink) !important; transition: all 0.2s ease-in-out; }
section[data-testid="stSidebar"] .stButton > button, section[data-testid="stSidebar"] .stButton > button * { color: var(--ink) !important; font-family: 'DM Mono', monospace; font-weight: 600; }
section[data-testid="stSidebar"] .stButton > button:hover { background-color: var(--ink) !important; border: 1px solid var(--warm-white) !important; }
section[data-testid="stSidebar"] .stButton > button:hover, section[data-testid="stSidebar"] .stButton > button:hover * { color: var(--warm-white) !important; }
.stChatInput > div { border: 2px solid var(--ink) !important; background-color: var(--warm-white) !important; }
.stChatInput textarea { color: var(--ink) !important; }
.doc-card { background-color: rgba(245, 240, 232, 0.1); border: 1px solid rgba(245, 240, 232, 0.2); border-radius: 4px; padding: 0.5rem; margin: 0.3rem 0; font-family: 'DM Mono', monospace; font-size: 0.75rem; display: flex; align-items: center; gap: 0.5rem; }
.source-chip { background-color: var(--dust); border: 1px solid rgba(26, 23, 20, 0.2); padding: 0.1rem 0.4rem; border-radius: 12px; font-size: 0.7rem; font-family: 'DM Mono', monospace; margin-right: 4px; color: var(--ink) !important; display: inline-block; margin-bottom: 4px; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "doc_stats" not in st.session_state:
    st.session_state.doc_stats = {"count": 0, "names": []}

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self._client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self._client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = self._client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
            config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values

def get_api_key():
    return os.environ.get("GEMINI_API_KEY")

def process_documents(uploaded_files):
    api_key = get_api_key()
    if not api_key:
        st.error("API Anahtarı bulunamadı (.env kontrol edin).")
        return None, 0, []

    loaders = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": lambda p: TextLoader(p, encoding="utf-8"),
        "md": lambda p: TextLoader(p, encoding="utf-8"),
    }

    all_chunks = []
    processed_names = []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    status_container = st.status("Dokümanlar işleniyor...", expanded=True)
    
    for file in uploaded_files:
        ext = file.name.rsplit(".", 1)[-1].lower()
        if ext not in loaders:
            continue
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            loader = loaders[ext](tmp_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file.name
            
            file_chunks = splitter.split_documents(docs)
            all_chunks.extend(file_chunks)
            processed_names.append(file.name)
            status_container.write(f" {file.name} ({len(file_chunks)} parça)")
        except Exception as e:
            status_container.write(f" {file.name} hatası: {e}")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    if not all_chunks:
        status_container.update(label="İşlenecek metin bulunamadı veya hata oluştu.", state="error")
        return None, 0, []

    status_container.write("Vektör veritabanı oluşturuluyor...")
    
    try:
        embeddings = GeminiEmbeddings(api_key)
        
        # PERSISTENCE (Kalıcılık) engellemek ve temiz bir başlangıç yapmak için:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            # persist_directory=None # Eğer diskte tutuyorsanız burayı belirtmelisiniz
        )
        
        # Mevcut koleksiyonu tamamen sil
        try:
            vectorstore.delete_collection()
        except:
            pass

        # Şimdi temizlenmiş koleksiyona yeni dokümanları ekle
        vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME
        )
        
        status_container.update(label="İşlem Başarılı!", state="complete", expanded=False)
        return vectorstore, len(all_chunks), processed_names
    except Exception as e:
        status_container.update(label=f"Vektör/Embedding hatası: {e}", state="error")
        return None, 0, []

def get_rag_response(query: str, vectorstore: Chroma, total_chunks: int):
    client = genai.Client(api_key=get_api_key())
    
    k = min(DEFAULT_RETRIEVAL_K, total_chunks) if total_chunks > 0 else 1
    retrieved_docs = vectorstore.similarity_search(query, k=k)

    if not retrieved_docs:
        return "Dokümanlarda bu soruyla ilgili bilgi bulunamadı.", []

    context_parts = []
    source_set = set()
    
    for d in retrieved_docs:
        src = d.metadata.get("source", "Bilinmeyen")
        page = d.metadata.get("page", None)
        ref = f"{src} (s.{int(page)+1})" if page is not None else src
        
        context_parts.append(f"--- Belge: {ref} ---\n{d.page_content}")
        source_set.add(ref)

    context_str = "\n\n".join(context_parts)
    sources = list(source_set)

    system_instruction = f"""Sen bir şirket içi dokümantasyon asistanısın. Görevin, yalnızca sana verilen şirket dokümanlarına dayanarak soruları yanıtlamaktır.

KURALLAR:
1. SADECE aşağıdaki "Bağlam" bölümündeki bilgileri kullan.
2. Eğer soru bağlamdaki bilgilerle yanıtlanamıyorsa tam olarak şunu söyle: "Bu bilgiye sahip değilim."
3. Asla tahmin yürütme, uydurma bilgi verme veya kendi genel bilginle yanıt oluşturma.
4. Türkçe soruda → Türkçe yanıtla; İngilizce soruda → İngilizce yanıtla.
5. Yanıtların açık, sade ve anlaşılır olsun. Madde madde listeler kullanabilirsin.

Bağlam:
{context_str}
"""
    try:
        resp = client.models.generate_content(
            model="gemini-flash-lite-latest",
            contents=query,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=TEMPERATURE
            )
        )
        return resp.text, sources
    except Exception as e:
        return f"Model hatası: {str(e)}", []

with st.sidebar:
    st.title("Doküman Yönetimi")
    uploaded_files = st.file_uploader(
        "Dosya Yükle", 
        type=list(FILE_ICONS.keys()), 
        accept_multiple_files=True
    )
    
    if st.button("Dosyaları İşle", use_container_width=True):
        if uploaded_files:
            st.session_state.vectorstore = None
            st.session_state.doc_stats = {"count": 0, "names": []}
            vs, count, names = process_documents(uploaded_files)
            if vs:
                st.session_state.vectorstore = vs
                st.session_state.doc_stats = {"count": count, "names": names}
                st.rerun()
        else:
            st.warning("Lütfen dosya seçin.")

    st.divider()
    if st.session_state.doc_stats["names"]:
        st.caption(f"İndeks: {st.session_state.doc_stats['count']} parça")
        for name in st.session_state.doc_stats["names"]:
            ext = name.split(".")[-1]
            st.markdown(f'<div class="doc-card">{FILE_ICONS.get(ext, "📄")} {name}</div>', unsafe_allow_html=True)
    else:
        st.info("Henüz doküman işlenmedi.")

st.title("Şirket İçi AI Asistan")

if not st.session_state.vectorstore:
    st.info("Başlamak için sol menüden doküman yükleyip 'İşle' butonuna basınız.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Sadece asistan mesajlarında ve eğer kaynak varsa göster
        if msg["role"] == "assistant" and msg.get("sources"):
            # Kaynakların benzersiz olduğundan emin olalım
            unique_sources = sorted(list(set(msg["sources"])))
            
            # Kaynak çiplerini oluştur
            chips = "".join([
                f'<span class="source-chip">📎 {s}</span>' 
                for s in unique_sources
            ])
            
            # HTML ile temiz bir görünüm sağla
            st.markdown(
                f'<div style="margin-top: 8px; display: flex; flex-wrap: wrap; gap: 4px;">{chips}</div>', 
                unsafe_allow_html=True
            )

    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Dokümanlar taranıyor..."):
                response_text, source_list = get_rag_response(
                    prompt, 
                    st.session_state.vectorstore, 
                    st.session_state.doc_stats["count"]
                )
                
                st.markdown(response_text)
                if source_list:
                    chips = "".join([f'<span class="source-chip">📎 {s}</span>' for s in source_list])
                    st.markdown(f"<br>{chips}", unsafe_allow_html=True)
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response_text, 
            "sources": source_list
        })