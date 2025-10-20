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
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import re

def preprocess_query(query):
    """Süper basit ama etkili preprocessing"""
    
    # Karşılaştırma
    if any(word in query.lower() for word in ['en yüksek', 'en iyi', 'kimdir']):
        return "**COMPARE:highest_overall**"
    if 'hızlı' in query.lower():
        return "**COMPARE:highest_pace**"
    
    # Büyük harfli isim varsa al
    import re
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

# app.py - load_database fonksiyonunu böyle güncelle
@st.cache_resource(show_spinner=False)
# ------------------- GLOBAL CSV YÜKLEME -------------------

@st.cache_data(show_spinner=False)
def load_csv_data():
    """CSV dosyasını cache'le - tüm uygulamada kullanılacak"""
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

# Global CSV data - TÜM FONKSIYONLARDA KULLANILACAK
csv_df = load_csv_data()

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
        return vectordb
    except Exception as e:
        st.warning(f"⚠️ DB yükleme hatası: {e}")
        # ✅ Otomatik düzeltme
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
        try:
            with st.spinner("🔄 Veritabanı yeniden oluşturuluyor..."):
                create_database()
            vectordb = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embedding_function,
                collection_name=COLLECTION_NAME
            )
            st.success("✅ Veritabanı başarıyla yenilendi!")
            return vectordb
        except Exception as e2:
            st.error(f"❌ Yeniden oluşturma başarısız: {e2}")
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
        
        # ✅ GELİŞTİRİLMİŞ PROMPT - Sadece EN alakalı futbolcuyu göster
        prompt_template = """Sen futbolcu istatistiklerini FIFA kartı formatında sunan bir asistansın.

Aşağıda 'context' kısmında futbolcu bilgileri var. Bu bilgileri kullanarak kullanıcının sorduğu futbolcunun FIFA kartını oluştur. Eğer kullanıcı birden fazla futbolcu sorarsa, hepsinin kartını sırayla oluştur.

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
        
        # ✅ MMR değil, SIMILARITY kullan + k=5 (sonra filtrele)
        retriever = _vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,  # Daha fazla al, sonra filtrele
            }
        )
        
        # RAG zincirini oluştur
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )
        
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

#Sidebar
# Sidebar'a ekle
with st.sidebar:
    st.markdown("---")
    show_debug = st.checkbox("🐛 Debug Modu", value=False)

# Chat bloğunda
if show_debug:  # Sadece debug açıksa göster
    with st.expander("🔍 Debug: ..."):
        #Debug Bilgileri


# Veritabanını yükle
vectordb = load_database()

if vectordb:
    # RAG zincirini kur
    qa_chain = setup_rag_chain(vectordb)
    
    if qa_chain:
        # Chat geçmişi
        
        # Session state başlangıç (global alana ekle)
        if "last_request_time" not in st.session_state:
            st.session_state.last_request_time = 0
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Önceki mesajları göster
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
                    # Kullanıcı inputu
            # Kullanıcı inputu
        if prompt := st.chat_input(...):
            current_time = time.time()
            if current_time - st.session_state.last_request_time < 1.5:
                st.warning("⏳ Lütfen 1.5 saniye bekleyin...")
                st.stop()
            st.session_state.last_request_time = current_time
        if prompt := st.chat_input("Örnek: Lionel Messi, Benzema, en yüksek dereceli futbolcu..."):
            processed_query = preprocess_query(prompt)
            
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("⚽ FIFA Kartı hazırlanıyor..."):
                    try:
                        # ✅ KARŞILAŞTIRMA SORGULARİ - CSV SORTING
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
                            
                            # ✅ CSV'DEN DIREKT SIRALA
                            if csv_df is not None:
                                # Temizle ve sırala
                                df_clean = csv_df.dropna(subset=[stat_name])
                                top_df = df_clean.sort_values(by=stat_name, ascending=False).head(10)
                                best = top_df.iloc[0]
                                
                                # DEBUG
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
                        
                        # ✅ NORMAL İSİM SORGULARİ - HİBRİT ARAMA
                        else:
                            # 1. Embedding search
                            docs_with_scores = vectordb.similarity_search_with_score(
                                processed_query, 
                                k=10
                            )
                            
                            best_score = docs_with_scores[0][1] if docs_with_scores else 999
                            
                            # 2. CSV Fallback (skor kötüyse)
                            if best_score > 0.7 or not docs_with_scores:
                                if csv_df is not None:
                                    # Partial matching - regex=False önemli!
                                    matching = csv_df[
                                        csv_df['Name'].str.contains(
                                            processed_query, 
                                            case=False, 
                                            na=False, 
                                            regex=False  # ✅ Regex hatalarını önler
                                        )
                                    ]
                                    
                                    if len(matching) > 0:
                                        best = matching.iloc[0]
                                        
                                        # DEBUG
                                        with st.expander("🔍 Debug: CSV Fallback"):
                                            st.write(f"**Aranan:** '{processed_query}'")
                                            st.write(f"**Bulunan:** {best['Name']}")
                                            st.write(f"**Embedding skoru kötü:** {best_score:.3f}")
                                            if len(matching) > 1:
                                                st.write(f"**Diğer eşleşmeler:** {len(matching)} futbolcu")
                                                st.dataframe(matching[['Name', 'Club', 'Overall']].head(5))
                                        
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
                            
                            # 3. Embedding başarılı
                            else:
                                with st.expander("🔍 Debug: Embedding Search"):
                                    st.write(f"**Aranan:** '{processed_query}'")
                                    st.write(f"**En İyi Skor:** {best_score:.3f}")
                                    for i, (doc, score) in enumerate(docs_with_scores[:3], 1):
                                        st.text(f"{i}. {doc.page_content[:100]}... ({score:.3f})")
                                
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
