__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------- YAPILANDIRMA -------------------

GEMINI_KEY = os.environ.get("GEMINI_API_KEY") 
CSV_FILE = 'male_players.csv' 
PERSIST_DIRECTORY = "./chroma_db"

# ------------------- YARDIMCI FONKSİYON -------------------

def load_and_prepare_data():
    """CSV'yi yükler ve temizler."""
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"'{CSV_FILE}' dosyası bulunamadı.")
    
    df = pd.read_csv(CSV_FILE)
    
    # Gerekli kolonları seç
    df_clean = df[[
        'Name', 'Club', 'Overall', 'Pace', 'Shooting', 
        'Passing', 'Dribbling', 'Defending', 'Physicality'
    ]].copy()
    
    # Boş değerleri doldur
    df_clean.fillna({
        'Overall': 0, 'Pace': 0, 'Shooting': 0, 
        'Passing': 0, 'Dribbling': 0, 'Defending': 0, 'Physicality': 0
    }, inplace=True)
    df_clean.fillna('Bilinmiyor', inplace=True)
    
    return df_clean

def create_player_chunk(row):
    """Futbolcu verisini RAG için metin formatına çevirir."""
    return (
        f"Futbolcu Adı: {row['Name']}. Kulüp: {row['Club']}. "
        f"Genel Reyting (OVR): {int(row['Overall'])}. "
        f"Temel FIFA Kart İstatistikleri: "
        f"Hız (PAC): {int(row['Pace'])}, Şut (SHO): {int(row['Shooting'])}, "
        f"Pas (PAS): {int(row['Passing'])}, Dribbling (DRI): {int(row['Dribbling'])}, "
        f"Defans (DEF): {int(row['Defending'])}, Fizik (PHY): {int(row['Physicality'])}."
    )

# ------------------- RAG KURULUM -------------------

@st.cache_resource(show_spinner="Vektör Veritabanı Hazırlanıyor...")
def setup_rag_chain():
    """Vektör DB ve RAG zincirini kurar."""
    
    if not GEMINI_KEY:
        st.error("❌ API Anahtarı bulunamadı. Streamlit Cloud'da 'GEMINI_API_KEY' ayarlayın.")
        return None
    
    try:
        # Embedding fonksiyonu
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GEMINI_KEY
        )
        
        # Vektör DB'yi yükle veya oluştur
        if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
            st.info("✅ Mevcut vektör veritabanı yükleniyor...")
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_function
            )
        else:
            st.info("🔄 Yeni vektör veritabanı oluşturuluyor (bu işlem birkaç dakika sürebilir)...")
            
            # Veriyi yükle ve hazırla
            df_clean = load_and_prepare_data()
            df_clean['rag_chunk'] = df_clean.apply(create_player_chunk, axis=1)
            
            # Document listesi oluştur
            data_documents = [
                Document(page_content=chunk) 
                for chunk in df_clean['rag_chunk'].tolist()
            ]
            
            # Vektör DB'yi oluştur
            vectorstore = Chroma.from_documents(
                documents=data_documents, 
                embedding=embedding_function, 
                persist_directory=PERSIST_DIRECTORY
            )
            st.success(f"✅ {len(data_documents)} futbolcu başarıyla indekslendi!")
        
        # LLM'i yapılandır
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=GEMINI_KEY
        )
        
        # Prompt template
        prompt_template = """Sen, futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.

Aşağıdaki 'context' kısmında verilen futbolcu istatistiklerini kullanarak,
SADECE o verilere dayanarak, net ve görsel olarak tasarlanmış bir FIFA kartı formatında cevap oluştur.

Context:
{context}

Soru: {input}

FIFA Kartı:
"""
        
        prompt = ChatPromptTemplate.from_template(prompt_template)
        document_chain = create_stuff_documents_chain(llm, prompt)
        
        # RAG zincirini oluştur (sadece en yakın 1 sonuç)
        retrieval_chain = create_retrieval_chain(
            vectorstore.as_retriever(search_kwargs={"k": 1}),
            document_chain
        )
        
        return retrieval_chain
        
    except Exception as e:
        st.error(f"❌ RAG kurulum hatası: {e}")
        return None

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(page_title="⚽ FIFA Kartı Chatbot", layout="wide")
st.title("⚽ FIFA Kartı Oluşturucu")
st.markdown("🔍 Futbolcu adı girin ve FIFA kartını görün!")

# RAG zincirini kur
qa_chain = setup_rag_chain()

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
    st.warning("⚠️ Sistem başlatılamadı. Lütfen API anahtarını kontrol edin.")
