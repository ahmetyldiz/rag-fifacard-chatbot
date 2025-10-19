# pysqlite3 sadece Streamlit Cloud için gerekli
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass  # Local'de çalışıyoruz, sorun yok

import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from create_database import create_database

# .env dosyasını yükle
load_dotenv()

# ------------------- YAPILANDIRMA -------------------

# API key'i çoklu kaynaktan al
GEMINI_KEY = (
    os.environ.get("GEMINI_API_KEY") or
    st.secrets.get("GEMINI_API_KEY", None) or
    None
) 
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fifa-players"

# ------------------- VERITABANI YÜKLEME -------------------

@st.cache_resource(show_spinner=False)
def load_database():
    """
    Vektör veritabanını yükler.
    Eğer yoksa veya hatalıysa otomatik oluşturur.
    """
    
    if not GEMINI_KEY:
        st.error("❌ API Anahtarı bulunamadı. 'GEMINI_API_KEY' ayarlayın.")
        return None
    
    # Embedding fonksiyonu
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Daha yeni model
        google_api_key=GEMINI_KEY
    )
    
    # Veritabanı var mı kontrol et
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        st.warning("🔄 Vektör veritabanı bulunamadı. Oluşturuluyor...")
        st.info("⏳ **Bu işlem 10-30 dakika sürebilir.** Lütfen sayfayı kapatmayın!")
        
        try:
            # Veritabanını oluştur
            with st.spinner("📊 Futbolcular indeksleniyor..."):
                create_database()
            st.success("✅ Veritabanı başarıyla oluşturuldu!")
            st.rerun()  # Sayfayı yenile
        except Exception as e:
            st.error(f"❌ Veritabanı oluşturma hatası: {e}")
            return None
    
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
        st.warning("🔧 Veritabanı temizleniyor ve yeniden oluşturuluyor...")
        
        try:
            # Bozuk veritabanını sil
            if os.path.exists(PERSIST_DIRECTORY):
                shutil.rmtree(PERSIST_DIRECTORY)
            
            # Yeniden oluştur
            with st.spinner("📊 Yeniden oluşturuluyor..."):
                create_database()
            
            # Tekrar yükle
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_function,
                collection_name=COLLECTION_NAME
            )
            st.success("✅ Veritabanı başarıyla yenilendi!")
            st.rerun()
            return vectordb
            
        except Exception as e2:
            st.error(f"❌ Yeniden oluşturma başarısız: {e2}")
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
        prompt_template = """Sen, futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.

Aşağıdaki 'context' kısmında verilen futbolcu istatistiklerini kullanarak,
SADECE o verilere dayanarak, net ve görsel bir FIFA kartı formatında cevap oluştur.

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

Soru: {input}

FIFA Kartı:
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # RAG zincirini oluştur
        retrieval_chain = create_retrieval_chain(
            _vectordb.as_retriever(search_kwargs={"k": 3}),
            document_chain
        )
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"❌ RAG zinciri kurulum hatası: {e}")
        return None

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(page_title="⚽ FIFA Kartı Chatbot", layout="wide")

# Header
st.title("⚽ FIFA Kartı Oluşturucu")
st.markdown("🔍 Futbolcu adı girin ve FIFA kartını görün!")

# Sidebar bilgi
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
    
    ---
    """)
    
    st.header("⚙️ Sistem Durumu")
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        st.success("✅ Vektör DB Hazır")
    else:
        st.warning("⏳ İlk Kurulum Gerekli")
    
    st.markdown("---")
    st.caption("🔧 **Sorun mu var?**")
    if st.button("🗑️ Veritabanını Sıfırla"):
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            st.success("✅ Veritabanı silindi. Sayfa yenilenecek...")
            st.rerun()

# Veritabanını yükle
vectordb = load_database()

if vectordb:
    # RAG zincirini kur
    qa_chain = setup_rag_chain(vectordb)
    
    if qa_chain:
        # Chat geçmişi
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Önceki mesajları göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Kullanıcı inputu
        if prompt := st.chat_input("Örnek: Lionel Messi, Cristiano Ronaldo..."):
            # Kullanıcı mesajını kaydet ve göster
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Asistan cevabı
            with st.chat_message("assistant"):
                with st.spinner("⚽ FIFA Kartı hazırlanıyor..."):
                    try:
                        response = qa_chain.invoke({"input": prompt})
                        full_response = response['answer']
                        st.markdown(full_response)
                    except Exception as e:
                        st.error(f"❌ Hata: {e}")
                        full_response = "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
            
            # Asistan mesajını kaydet
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("❌ RAG zinciri kurulamadı.")
else:
    st.error("❌ Veritabanı yüklenemedi. Lütfen sayfayı yenileyin.")
