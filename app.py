# pysqlite3 sadece Streamlit Cloud için gerekli
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Local'de çalışıyoruz, sorun yok

import time
import pandas as pd
import streamlit as st
import os
import shutil
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------- PREPROCESSING -------------------

def preprocess_query(query):
    """Süper basit ama etkili preprocessing"""
    
    # Karşılaştırma
    if any(word in query.lower() for word in ['en yüksek', 'en iyi', 'kimdir']):
        return "**COMPARE:highest_overall**"
    if 'hızlı' in query.lower():
        return "**COMPARE:highest_pace**"
    
    # Büyük harfli isim varsa al
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    if names:
        return names[0]
    
    # Basit temizleme
    result = query.lower()
    
    # Türkçe ekleri manuel temizle
    result = result.replace("'nın", "").replace("'nin", "").replace("nın", "").replace("nin", "")
    result = result.replace("'ın", "").replace("'in", "").replace("ın", "").replace("in", "")
    
    # Gereksiz kelimeleri sil
    for word in ['kartı', 'kart', 'göster', 'oluştur', 'getir', 'bana', 'fifa']:
        result = result.replace(word, " ")
    
    # İlk kelimeyi al ve capitalize
    result = result.strip().split()[0] if result.strip().split() else result
    return result.capitalize()

# ------------------- YAPILANDIRMA -------------------

load_dotenv()

# API key'i çoklu kaynaktan al
GEMINI_KEY = (
    os.environ.get("GEMINI_API_KEY") or
    st.secrets.get("GEMINI_API_KEY", None) or
    None
) 
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fifa-players"

# ------------------- CSV YÜKLEME -------------------

@st.cache_resource(show_spinner=False)
def load_database():
    if not GEMINI_KEY:
        st.error("❌ API Anahtarı bulunamadı.")
        return None
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )
    
    # Schema hatası varsa CSV fallback kullan (DB'siz çalışacak)
    try:
        from langchain_community.vectorstores import Chroma
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        # Test
        vectordb.similarity_search("test", k=1)
        return vectordb
    except Exception as e:
        st.warning(f"⚠️ Veritabanı yüklenemedi: {e}")
        st.info("💡 Sadece CSV fallback modu çalışacak")
        return None


# Global CSV data
csv_df = load_csv_data()

# ------------------- VERITABANI YÜKLEME -------------------

@st.cache_resource(show_spinner=False)
def load_database():
    if not GEMINI_KEY:
        st.error("❌ API Anahtarı bulunamadı.")
        return None
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )
    
    # Veritabanını yükle
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        return vectordb
        
    except Exception as e:
        st.error(f"⚠️ Veritabanı yükleme hatası: {e}")
        return None

# ------------------- RAG ZİNCİRİ KURULUMU -------------------

@st.cache_resource(show_spinner=False)
def setup_rag_chain(_vectordb):
    """RAG zincirini kurar."""
    
    if _vectordb is None:
        return None
    
    try:
        # LLM'i yapılandır
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            google_api_key=GEMINI_KEY
        )
        
        # Prompt template
        prompt_template = """Sen futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.

Aşağıda 'context' kısmında futbolcu bilgileri var. Bu bilgileri kullanarak kullanıcının sorduğu futbolcunun FIFA kartını oluştur.

**ÖNEMLİ TALİMAT:**
- Context'teki futbolculardan, kullanıcının sorgusuna EN UYGUN OLANI seç
- Sadece O futbolcunun kartını oluştur
- Eğer context'te hiç uygun futbolcu yoksa: "Üzgünüm, [futbolcu adı] veritabanında bulunamadı" de

**FIFA Kartı Formatı:**
━━━━━━━━━━━━━━━━━━━━
⚽ **[FUTBOLCU ADI]**
━━━━━━━━━━━━━━━━━━━━
🏆 **OVR:** [Genel Puan]
🏟️ **Kulüp:** [Kulüp Adı]

📊 **İSTATİSTİKLER:**
├─ ⚡ Hız: [PAC]
├─ 🎯 Şut: [SHO]
├─ 🎨 Pas: [PAS]
├─ ⚽ Dribling: [DRI]
├─ 🛡️ Defans: [DEF]
└─ 💪 Fizik: [PHY]
━━━━━━━━━━━━━━━━━━━━

Context:
{context}

Kullanıcı Sorusu: {input}

Cevap (sadece 1 futbolcu kartı):"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # Retriever
        retriever = _vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # RAG zincirini oluştur
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"❌ RAG zinciri kurulum hatası: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(page_title="⚽ FIFA Kartı Chatbot", layout="wide")

# Header
st.title("⚽ FIFA Kartı Oluşturucu")
st.markdown("🔍 Futbolcu adı girin ve FIFA kartını görün!")

# Sidebar
with st.sidebar:
    st.header("📖 Kullanım Kılavuzu")
    st.markdown("""
    **Nasıl Kullanılır?**
    1. Aşağıdaki chat kutusuna futbolcu adı yazın
    2. Enter'a basın
    3. FIFA kartını görüntüleyin!
    
    **Örnek Aramalar:**
    - Lionel Messi
    - Cristiano Ronaldo
    - Kylian Mbappé
    - En yüksek dereceli futbolcu
    """)
    
    st.markdown("---")
    show_debug = st.checkbox("🐛 Debug Modu", value=False)

# Veritabanını yükle
vectordb = load_database()

if vectordb:
    # RAG zincirini kur
    qa_chain = setup_rag_chain(vectordb)
    
    if qa_chain:
        # Session state başlangıç
        if "last_request_time" not in st.session_state:
            st.session_state.last_request_time = 0
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Önceki mesajları göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Kullanıcı inputu
        if prompt := st.chat_input("Örnek: Lionel Messi, Benzema, en yüksek dereceli futbolcu..."):
            # Rate limiting
            current_time = time.time()
            if current_time - st.session_state.last_request_time < 1.5:
                st.warning("⏳ Lütfen 1.5 saniye bekleyin...")
                st.stop()
            st.session_state.last_request_time = current_time
            
            processed_query = preprocess_query(prompt)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("⚽ FIFA Kartı hazırlanıyor..."):
                    try:
                        # KARŞILAŞTIRMA SORGULARİ
                        if processed_query.startswith("**COMPARE:"):
                            compare_type = processed_query.replace("**COMPARE:", "").replace("**", "")
                            
                            # Stat belirleme
                            if compare_type == "highest_overall":
                                stat_name = "Overall"
                                stat_label = "Overall"
                            elif compare_type == "highest_pace":
                                stat_name = "Pace"
                                stat_label = "Hız"
                            elif compare_type == "highest_physicality":
                                stat_name = "Physicality"
                                stat_label = "Fizik"
                            else:
                                stat_name = "Overall"
                                stat_label = "Overall"
                            
                            # CSV'DEN DIREKT SIRALA
                            if csv_df is not None:
                                df_clean = csv_df.dropna(subset=[stat_name])
                                top_df = df_clean.sort_values(by=stat_name, ascending=False).head(10)
                                best = top_df.iloc[0]
                                
                                # DEBUG
                                if show_debug:
                                    with st.expander("🔍 Debug: En İyi 10 Futbolcu"):
                                        st.write(f"**Sıralama Kriteri:** {stat_label}")
                                        st.dataframe(top_df[['Name', 'Club', stat_name]].head(10))
                                
                                # Kartı oluştur
                                full_response = f"""━━━━━━━━━━━━━━━━━━━━
⚽ **{best['Name']}**
━━━━━━━━━━━━━━━━━━━━
🏆 **OVR:** {int(best['Overall'])}
🏟️ **Kulüp:** {best['Club']}

📊 **İSTATİSTİKLER:**
├─ ⚡ Hız: {int(best['Pace'])}
├─ 🎯 Şut: {int(best['Shooting'])}
├─ 🎨 Pas: {int(best['Passing'])}
├─ ⚽ Dribling: {int(best['Dribbling'])}
├─ 🛡️ Defans: {int(best['Defending'])}
└─ 💪 Fizik: {int(best['Physicality'])}
━━━━━━━━━━━━━━━━━━━━

*En yüksek {stat_label}: {int(best[stat_name])}*"""
                            else:
                                full_response = "❌ Üzgünüm, CSV verisi yüklenemedi."
                        
                        # NORMAL İSİM SORGULARİ
                        else:
                            # Embedding search
                            docs_with_scores = vectordb.similarity_search_with_score(processed_query, k=10)
                            
                            best_score = docs_with_scores[0][1] if docs_with_scores else 999
                            
                            # CSV Fallback (skor kötüyse)
                            if best_score > 0.7 or not docs_with_scores:
                                if csv_df is not None:
                                    matching = csv_df[
                                        csv_df['Name'].str.contains(
                                            processed_query, 
                                            case=False, 
                                            na=False, 
                                            regex=False
                                        )
                                    ]
                                    
                                    if len(matching) > 0:
                                        best = matching.iloc[0]
                                        
                                        # DEBUG
                                        if show_debug:
                                            with st.expander("🔍 Debug: CSV Fallback"):
                                                st.write(f"**Aranan:** '{processed_query}'")
                                                st.write(f"**Bulunan:** {best['Name']}")
                                                st.write(f"**Embedding skoru kötü:** {best_score:.3f}")
                                        
                                        # Kartı oluştur
                                        full_response = f"""━━━━━━━━━━━━━━━━━━━━
⚽ **{best['Name']}**
━━━━━━━━━━━━━━━━━━━━
🏆 **OVR:** {int(best['Overall'])}
🏟️ **Kulüp:** {best['Club']}

📊 **İSTATİSTİKLER:**
├─ ⚡ Hız: {int(best['Pace'])}
├─ 🎯 Şut: {int(best['Shooting'])}
├─ 🎨 Pas: {int(best['Passing'])}
├─ ⚽ Dribling: {int(best['Dribbling'])}
├─ 🛡️ Defans: {int(best['Defending'])}
└─ 💪 Fizik: {int(best['Physicality'])}
━━━━━━━━━━━━━━━━━━━━"""
                                    else:
                                        full_response = f"Üzgünüm, '{processed_query}' veritabanında bulunamadı."
                                else:
                                    full_response = "❌ CSV verisi yüklenemedi."
                            
                            # Embedding başarılı
                            else:
                                if show_debug:
                                    with st.expander("🔍 Debug: Embedding Search"):
                                        st.write(f"**Aranan:** '{processed_query}'")
                                        st.write(f"**En İyi Skor:** {best_score:.3f}")
                                
                                best_doc = docs_with_scores[0][0]
                                response = qa_chain.invoke({
                                    "input": processed_query,
                                    "context": best_doc.page_content
                                })
                                full_response = response['answer']
                        
                        st.markdown(full_response)
                        
                    except Exception as e:
                        st.error(f"❌ Hata: {e}")
                        import traceback
                        with st.expander("🐛 Teknik Detaylar"):
                            st.code(traceback.format_exc())
                        full_response = "Üzgünüm, bir hata oluştu."
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    else:
        st.error("❌ RAG zinciri kurulamadı.")
else:
    st.error("❌ Veritabanı yüklenemedi. Lütfen sayfayı yenileyin.")
