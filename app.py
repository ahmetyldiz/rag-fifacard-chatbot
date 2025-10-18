
import streamlit as st
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ------------------- API ANAHTARI VE YAPILANDIRMA -------------------

# API Anahtarı, Streamlit Cloud'da "Secrets" bölümünden çekilir. 
# Bu, anahtarınızı güvende tutar.
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") 
CSV_FILE = 'male_players.csv' 
PERSIST_DIRECTORY = "./chroma_db" 

# ------------------- RAG KURULUM FONKSİYONLARI -------------------

@st.cache_resource(show_spinner="Veri Seti Yükleniyor ve Vektör Veritabanı Kuruluyor...")
def setup_rag_chain():
    """Veri yükleme, DB oluşturma ve RAG zincirini kurar (Tekrar tekrar çalışmaz)."""

    if not GEMINI_KEY:
        st.error("API Anahtarı bulunamadı. Lütfen Streamlit Cloud'da 'GEMINI_API_KEY'i ayarlayın.")
        return None

    # 1. Veri Yükleme ve Hazırlama
    try:
        # Streamlit Cloud'da dosyanın ana dizinde olduğunu varsayıyoruz.
        if not os.path.exists(CSV_FILE):
             st.error(f"Kritik hata: '{CSV_FILE}' dosyası bulunamadı. Lütfen GitHub deponuza yüklediğinizden emin olun.")
             return None
             
        df = pd.read_csv(CSV_FILE)
        
        # Sütunları seçme ve temizleme
        df_clean = df[[
            'Name', 'Club', 'Overall', 'Pace', 'Shooting', 
            'Passing', 'Dribbling', 'Defending', 'Physicality'
        ]].copy()

        df_clean.fillna({'Overall': 0, 'Pace': 0, 'Shooting': 0, 'Passing': 0, 'Dribbling': 0, 'Defending': 0, 'Physicality': 0}, inplace=True)
        df_clean.fillna('Bilinmiyor', inplace=True)

        def create_player_chunk(row):
            chunk = (
                f"Futbolcu Adı: {row['Name']}. Kulüp: {row['Club']}. Genel Reyting (OVR): {int(row['Overall'])}. "
                f"Temel FIFA Kart İstatistikleri: "
                f"Hız (PAC): {int(row['Pace'])}, Şut (SHO): {int(row['Shooting'])}, "
                f"Pas (PAS): {int(row['Passing'])}, Dribbling (DRI): {int(row['Dribbling'])}, "
                f"Defans (DEF): {int(row['Defending'])}, Fizik (PHY): {int(row['Physicality'])}."
            )
            return chunk

        df_clean['rag_chunk'] = df_clean.apply(create_player_chunk, axis=1)
        data_documents = [Document(page_content=chunk) for chunk in df_clean['rag_chunk'].tolist()]

    except Exception as e:
        st.error(f"Veri hazırlama sırasında hata: {e}")
        return None

    # 2. Vektör İndeksleme
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GEMINI_KEY
    )

    vectorstore = Chroma.from_documents(
        documents=data_documents, 
        embedding=embedding_function, 
        persist_directory=PERSIST_DIRECTORY
    )
    
    # 3. RAG Zinciri Kurulumu (YENİ YÖNTEM)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",
        temperature=0.2,
        google_api_key=GEMINI_KEY
    )
    
    prompt_template = """
    Sen, futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.
    Aşağıdaki 'context' kısmında verilen futbolcu istatistiklerini kullanarak,
    SADECE o verilere dayanarak, net ve görsel olarak tasarlanmış bir FIFA kartı görünümünde cevap oluştur.
    Kesinlikle kartta yer almayan, başka bir bilgi ekleme.
    
    Context:
    {context}
    
    Soru: {input}
    
    Cevabın:
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 1}),
        document_chain
    )
    
    return retrieval_chain

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(page_title="⚽ FIFA Kartı RAG Chatbot", layout="wide")

st.title("⚽ FIFA Kartı Oluşturucu Chatbot")
st.markdown("Aradığınız futbolcunun tam ismini girerek güncel istatistiklerini **FIFA kartı** formatında isteyin.")

qa_chain = setup_rag_chain()

if qa_chain:
    # Sohbet geçmişini başlatma
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Geçmiş mesajları görüntüleme
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Kullanıcıdan giriş alma
    if prompt := st.chat_input("Hangi futbolcunun kartını görmek istersiniz?"):
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

       # Chatbot cevabını al ve göster (Streamlit arayüz kısmı)
with st.chat_message("assistant"):
    with st.spinner("FIFA Kartı Oluşturuluyor..."):
        try:
            # RAG zincirini çalıştırma (YENİ YÖNTEM)
            response = qa_chain.invoke({"input": prompt})
            full_response = response['answer']
            st.markdown(full_response)
        except Exception as e:
            st.error(f"Sorgu hatası oluştu: {e}")
            full_response = "Sorgu başarısız oldu."

                    
        st.session_state.messages.append({"role": "assistant", "content": full_response})
