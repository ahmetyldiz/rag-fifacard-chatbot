# pysqlite3 sadece Streamlit Cloud için gerekli
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import time
import pandas as pd
import streamlit as st
import os
import shutil
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------- YAPILANDIRMA -------------------

load_dotenv()

GEMINI_KEY = (
    os.environ.get("GEMINI_API_KEY") or
    st.secrets.get("GEMINI_API_KEY", None) or
    None
) 
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fifa-players"

# ⚠️ TOKEN KORUMA AYARLARI
MAX_QUERIES_PER_SESSION = 20  # Her session için maksimum sorgu sayısı
RATE_LIMIT_SECONDS = 2  # Sorgular arası minimum süre

# ------------------- LLM PREPROCESSING (CACHE'LENMİŞ) -------------------

@st.cache_resource(show_spinner=False)
def load_database():
    if not GEMINI_KEY:
        return None  # Sessizce None döndür
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )
    
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        vectordb.similarity_search("test", k=1)
        return vectordb
    except Exception as e:
        # Sessizce None döndür (kullanıcı görmez)
        return None


# ------------------- CSV YÜKLEME -------------------

@st.cache_data(show_spinner=False)
def load_csv_data():
    csv_path = 'male_players.csv'
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            st.error(f"CSV yükleme hatası: {e}")
            return None
    else:
        st.error(f"❌ '{csv_path}' dosyası bulunamadı!")
        return None

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
    
    try:
        vectordb = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        vectordb.similarity_search("test", k=1)
        return vectordb
    except Exception as e:
        st.warning(f"⚠️ Veritabanı yüklenemedi: {e}")
        st.info("💡 Sadece CSV fallback modu çalışacak")
        return None

# ------------------- RAG ZİNCİRİ -------------------

@st.cache_resource(show_spinner=False)
def setup_rag_chain(_vectordb):
    if _vectordb is None:
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.2,
            max_output_tokens=300,  # Token limiti
            google_api_key=GEMINI_KEY
        )
        
        prompt_template = """Sen futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.

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

Context: {context}
Soru: {input}
Cevap:"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = _vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"❌ RAG zinciri kurulum hatası: {e}")
        return None

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(page_title="⚽ FIFA Kartı Chatbot", layout="wide")
st.title("⚽ FIFA Kartı Oluşturucu")
st.markdown("🔍 Futbolcu adı girin ve FIFA kartını görün!")

# Session state başlatma
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

with st.sidebar:
    st.header("📖 Kullanım Kılavuzu")
    st.markdown("""
    **Nasıl Kullanılır?**
    1. Chat kutusuna futbolcu adı yazın
    2. Enter'a basın
    3. FIFA kartını görüntüleyin!
    
    **Örnek Aramalar:**
    - Lionel Messi
    - Messinin kartı
    - En yüksek dereceli futbolcu
    - En iyi defans
    """)
    
    st.markdown("---")
    st.metric("Kalan Sorgu", max(0, MAX_QUERIES_PER_SESSION - st.session_state.query_count))
    show_debug = st.checkbox("🐛 Debug Modu", value=False)

vectordb = load_database()


# QUERY LİMİT KONTROLÜ
if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
    st.error(f"❌ Maksimum sorgu limitine ulaştınız ({MAX_QUERIES_PER_SESSION}). Sayfayı yenileyerek devam edebilirsiniz.")
    st.stop()

if prompt := st.chat_input("Futbolcu adı girin (örn: Messi, Ronaldo)..."):
    # Rate limit kontrolü
    current_time = time.time()
    if current_time - st.session_state.last_request_time < RATE_LIMIT_SECONDS:
        st.warning(f"⏳ Lütfen {RATE_LIMIT_SECONDS} saniye bekleyin...")
        st.stop()
    
    st.session_state.last_request_time = current_time
    st.session_state.query_count += 1
    
    processed_query = preprocess_query(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("⚽ Aranıyor..."):
            try:
                if processed_query.startswith("**COMPARE:"):
                    compare_type = processed_query.replace("**COMPARE:", "").replace("**", "")
                    
                    stat_mapping = {
                        "highest_overall": ("Overall", "Overall"),
                        "highest_pace": ("Pace", "Hız"),
                        "highest_defending": ("Defending", "Defans"),
                        "highest_physicality": ("Physicality", "Fizik"),
                        "highest_shooting": ("Shooting", "Şut"),
                        "highest_passing": ("Passing", "Pas"),
                        "highest_dribbling": ("Dribbling", "Dribling")
                    }
                    
                    stat_name, stat_label = stat_mapping.get(compare_type, ("Overall", "Overall"))
                    
                    if csv_df is not None:
                        df_clean = csv_df.dropna(subset=[stat_name])
                        best = df_clean.sort_values(by=stat_name, ascending=False).iloc[0]
                        
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
                        full_response = "❌ CSV verisi yüklenemedi."
                
                else:
                    if csv_df is not None:
                        matching = csv_df[csv_df['Name'].str.contains(processed_query, case=False, na=False, regex=False)]
                        
                        if len(matching) > 0:
                            best = matching.iloc[0]
                            
                            if show_debug:
                                st.info(f"🔍 LLM preprocessing: '{prompt}' → '{processed_query}'")
                            
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
                            full_response = f"Üzgünüm, '{processed_query}' bulunamadı."
                    else:
                        full_response = "❌ CSV verisi yüklenemedi."
                
                st.markdown(full_response)
                
            except Exception as e:
                st.error(f"❌ Hata: {e}")
                full_response = "Üzgünüm, bir hata oluştu."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
