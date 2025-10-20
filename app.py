# pysqlite3 sadece Streamlit Cloud için gerekli
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from unidecode import unidecode
import time
import pandas as pd
import streamlit as st
import os
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
MAX_QUERIES_PER_SESSION = 20
RATE_LIMIT_SECONDS = 2

# ------------------- LLM PREPROCESSING -------------------

@st.cache_data(ttl=3600, show_spinner=False)
def extract_player_name_with_llm(query):
    """Cache'lenmiş LLM ile futbolcu adı çıkarma"""
    if not GEMINI_KEY:
        return None
    
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0,
            max_output_tokens=50,
            google_api_key=GEMINI_KEY
        )
        
        prompt = f"""V2: Aşağıdaki cümleden SADECE futbolcu adını çıkar. Hiçbir açıklama yapma.

Cümle: {query}

Futbolcu adı:"""
        
        response = llm.invoke(prompt)
        player_name = response.content.strip()
        
        if len(player_name) > 30:
            return None
        
        return player_name
        
    except Exception:
        return None

def preprocess_query(query):
    """Hybrid preprocessing: LLM + Fallback"""
    query_lower = query.lower()
    
    # DEBUG PRINT
    print(f"🔍 DEBUG: query_lower = '{query_lower}'")
    print(f"🔍 'fizik' in query_lower: {'fizik' in query_lower}")
    print(f"🔍 'oyuncu' in query_lower: {'oyuncu' in query_lower}")
    
    # Genel mesajlar
    if query_lower in ['merhaba', 'selam', 'hello', 'hi', 'hey']:
        return "**GREETING**"
    elif query_lower in ['teşekkürler', 'teşekkür ederim', 'sağol', 'thanks', 'thank you']:
        return "**THANKS**"
    elif 'nasılsın' in query_lower or 'naber' in query_lower:
        return "**HOW_ARE_YOU**"
    
    # EN KÖTÜ
    if 'en kötü' in query_lower or 'en düşük' in query_lower:
        if 'hız' in query_lower:
            return "**COMPARE:lowest_pace**"
        elif 'fizik' in query_lower:
            return "**COMPARE:lowest_physicality**"
        elif 'defans' in query_lower:
            return "**COMPARE:lowest_defending**"
        elif 'şut' in query_lower:
            return "**COMPARE:lowest_shooting**"
        elif 'pas' in query_lower:
            return "**COMPARE:lowest_passing**"
        elif 'dribl' in query_lower:
            return "**COMPARE:lowest_dribbling**"
        else:
            return "**COMPARE:lowest_overall**"
    
    # EN YÜKSEK
    if any(word in query_lower for word in ['en yüksek', 'en iyi', 'en hızlı', 'hızlı', 'kim', 'oyuncu']):
        if 'hız' in query_lower:
            return "**COMPARE:highest_pace**"
        elif 'fizik' in query_lower:
            return "**COMPARE:highest_physicality**"
        elif 'defans' in query_lower or 'savunma' in query_lower:
            return "**COMPARE:highest_defending**"
        elif 'şut' in query_lower:
            return "**COMPARE:highest_shooting**"
        elif 'pas' in query_lower:
            return "**COMPARE:highest_passing**"
        elif 'dribl' in query_lower:
            return "**COMPARE:highest_dribbling**"
        else:
            return "**COMPARE:highest_overall**"
    
    # LLM ile dene
    llm_result = extract_player_name_with_llm(query)
    if llm_result and llm_result not in ['Yok', 'Yok.', 'Bilinmiyor', '-', 'None']:
        return llm_result
    
    # Fallback: Manuel preprocessing
    names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    if names:
        return names[0]
    
    result = query_lower
    suffixes = ["'nın", "'nin", "'ın", "'in", "nın", "nin", "ın", "in", 
                "'un", "'ün", "un", "ün", "'nda", "'de", "da", "de"]
    for suffix in suffixes:
        result = result.replace(suffix, "")
    
    stop_words = ['kartı', 'kart', 'kartını', 'göster', 'oluştur', 'getir', 'bana', 'fifa']
    for word in stop_words:
        result = result.replace(word, "")
    
    result = result.strip().split()[0] if result.strip().split() else result
    return result.capitalize()

# ------------------- CSV YÜKLEME -------------------

@st.cache_data(show_spinner=False)
def load_csv_data():
    csv_path = 'male_players.csv'
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
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
    except Exception:
        return None

# ------------------- CUSTOM CSS -------------------

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 300px !important;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
    }
    
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .fifa-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin: 20px 0;
    }
    
    .fifa-card h2 {
        margin: 0;
        font-size: 2em;
    }
    
    .fifa-card p {
        margin: 10px 0;
        font-size: 1.1em;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        margin: 5px 0;
    }
    
    .stChatInput {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- STREAMLIT ARAYÜZÜ -------------------

st.set_page_config(
    page_title="⚽ FIFA Kartı Chatbot",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown('<h1 class="main-title">⚽ FIFA Kartı Oluşturucu</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#666;'>🔍 17,000+ futbolcudan istediğini ara ve kartını gör!</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# ------------------- SIDEBAR -------------------

with st.sidebar:
    st.markdown("### ⚽ FIFA Kartı Chatbot")
    st.markdown("---")
    
    st.markdown("### 📊 Sistem Durumu")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toplam Futbolcu", "17,000+")
    with col2:
        st.metric("Kalan Sorgu", max(0, MAX_QUERIES_PER_SESSION - st.session_state.query_count))
    
    st.markdown("---")
    
    st.markdown("### 📖 Örnek Sorgular")
    st.markdown("""
    **🔍 Futbolcu Ara:**
    - Lionel Messi
    - Messinin kartı
    - Kylian Mbappe
    
    **📊 İstatistik Sorgula:**
    - En yüksek dereceli futbolcu
    - En hızlı oyuncu
    - Fiziği en yüksek oyuncu
    - En iyi defans
    """)
    
    st.markdown("---")
    st.caption("🤖 LLM-powered search with CSV fallback")

# ------------------- CHAT -------------------

vectordb = load_database()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

if st.session_state.query_count >= MAX_QUERIES_PER_SESSION:
    st.error(f"❌ Maksimum sorgu limitine ulaştınız ({MAX_QUERIES_PER_SESSION}). Sayfayı yenileyerek devam edebilirsiniz.")
    st.stop()

if prompt := st.chat_input("Futbolcu adı girin (örn: Messi, en hızlı oyuncu)..."):
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
        with st.spinner("⚽ FIFA Kartı hazırlanıyor..."):
            time.sleep(0.3)
            
            try:
                # GREETING kontrolü
                if processed_query == "**GREETING**":
                    full_response_text = "Merhaba! ⚽ Ben FIFA Kartı Chatbot'uyum. Hangi futbolcunun kartını görmek istersin?"
                    st.info(full_response_text)
                
                elif processed_query == "**THANKS**":
                    full_response_text = "Rica ederim! 😊 Başka bir futbolcu aramak ister misin?"
                    st.success(full_response_text)
                
                elif processed_query == "**HOW_ARE_YOU**":
                    full_response_text = "Ben bir botum, ama iyi sayılırım! ⚽ Futbolcu kartları göstermekten keyif alıyorum. Sen ne aramak istersin?"
                    st.info(full_response_text)
                
                elif processed_query.startswith("**COMPARE:"):
                    compare_type = processed_query.replace("**COMPARE:", "").replace("**", "")
                    
                    is_lowest = compare_type.startswith("lowest_")
                    if is_lowest:
                        compare_type = compare_type.replace("lowest_", "highest_")
                        label_prefix = "En düşük"
                    else:
                        label_prefix = "En yüksek"
                    
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
                        best = df_clean.sort_values(by=stat_name, ascending=is_lowest).iloc[0]
                        
                        full_response = f"""
<div class="fifa-card">
    <h2>⚽ {best['Name']}</h2>
    <p>🏆 Overall: <b>{int(best['Overall'])}</b> | 🏟️ {best['Club']}</p>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
    <h3>📊 İSTATİSTİKLER</h3>
    <div class="stat-row">
        <span>⚡ Hız: <b>{int(best['Pace'])}</b></span>
        <span>🎯 Şut: <b>{int(best['Shooting'])}</b></span>
        <span>🎨 Pas: <b>{int(best['Passing'])}</b></span>
    </div>
    <div class="stat-row">
        <span>⚽ Dribling: <b>{int(best['Dribbling'])}</b></span>
        <span>🛡️ Defans: <b>{int(best['Defending'])}</b></span>
        <span>💪 Fizik: <b>{int(best['Physicality'])}</b></span>
    </div>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
    <p style="text-align:center; font-style:italic;">{label_prefix} {stat_label}: {int(best[stat_name])}</p>
</div>
"""
                        
                        st.markdown(full_response, unsafe_allow_html=True)
                        
                        st.markdown("### 📊 Detaylı İstatistikler")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("⚡ **Hız**")
                            st.progress(int(best['Pace']) / 100)
                            st.markdown("🎯 **Şut**")
                            st.progress(int(best['Shooting']) / 100)
                            st.markdown("🎨 **Pas**")
                            st.progress(int(best['Passing']) / 100)
                        
                        with col2:
                            st.markdown("⚽ **Dribling**")
                            st.progress(int(best['Dribbling']) / 100)
                            st.markdown("🛡️ **Defans**")
                            st.progress(int(best['Defending']) / 100)
                            st.markdown("💪 **Fizik**")
                            st.progress(int(best['Physicality']) / 100)
                        
                        full_response_text = f"{best['Name']} - Overall: {int(best['Overall'])}"
                    else:
                        full_response_text = "❌ CSV verisi yüklenemedi."
                        st.error(full_response_text)
                
                else:
                    if csv_df is not None:
                        matching = csv_df[csv_df['Name'].str.contains(processed_query, case=False, na=False, regex=False)]
                        
                        if len(matching) == 0:
                            csv_df['Name_normalized'] = csv_df['Name'].apply(lambda x: unidecode(str(x)).lower())
                            processed_normalized = unidecode(processed_query).lower()
                            matching = csv_df[csv_df['Name_normalized'].str.contains(processed_normalized, na=False, regex=False)]
                        
                        if len(matching) > 0:
                            best = matching.iloc[0]
                            
                            full_response = f"""
<div class="fifa-card">
    <h2>⚽ {best['Name']}</h2>
    <p>🏆 Overall: <b>{int(best['Overall'])}</b> | 🏟️ {best['Club']}</p>
    <hr style="border-color: rgba(255,255,255,0.3); margin: 15px 0;">
    <h3>📊 İSTATİSTİKLER</h3>
    <div class="stat-row">
        <span>⚡ Hız: <b>{int(best['Pace'])}</b></span>
        <span>🎯 Şut: <b>{int(best['Shooting'])}</b></span>
        <span>🎨 Pas: <b>{int(best['Passing'])}</b></span>
    </div>
    <div class="stat-row">
        <span>⚽ Dribling: <b>{int(best['Dribbling'])}</b></span>
        <span>🛡️ Defans: <b>{int(best['Defending'])}</b></span>
        <span>💪 Fizik: <b>{int(best['Physicality'])}</b></span>
    </div>
</div>
"""
                            
                            st.markdown(full_response, unsafe_allow_html=True)
                            
                            st.markdown("### 📊 Detaylı İstatistikler")
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("⚡ **Hız**")
                                st.progress(int(best['Pace']) / 100)
                                st.markdown("🎯 **Şut**")
                                st.progress(int(best['Shooting']) / 100)
                                st.markdown("🎨 **Pas**")
                                st.progress(int(best['Passing']) / 100)
                            
                            with col2:
                                st.markdown("⚽ **Dribling**")
                                st.progress(int(best['Dribbling']) / 100)
                                st.markdown("🛡️ **Defans**")
                                st.progress(int(best['Defending']) / 100)
                                st.markdown("💪 **Fizik**")
                                st.progress(int(best['Physicality']) / 100)
                            
                            full_response_text = f"{best['Name']} - Overall: {int(best['Overall'])}"
                        else:
                            full_response_text = f"Üzgünüm, '{processed_query}' bulunamadı. Tam futbolcu adı yazın."
                            st.warning(full_response_text)
                    else:
                        full_response_text = "❌ CSV verisi yüklenemedi."
                        st.error(full_response_text)
                
            except Exception as e:
                st.error(f"❌ Hata: {e}")
                full_response_text = "Üzgünüm, bir hata oluştu."
    
    st.session_state.messages.append({"role": "assistant", "content": full_response_text})
