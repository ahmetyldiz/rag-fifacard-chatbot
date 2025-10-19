"""df
FIFA Futbolcu Veritabanı Oluşturma Scripti
Bu script sadece bir kez çalıştırılmalıdır (veya DB bozulduğunda).
"""

import os
import sys
import pandas as pd
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Konfigürasyon
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
CSV_FILE = 'male_players.csv'
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fifa-players"
BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = 2

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

def create_database():
    """Vektör veritabanını oluşturur."""
    
    if not GEMINI_KEY:
        print("❌ HATA: GEMINI_API_KEY bulunamadı!")
        sys.exit(1)
    
    if not os.path.exists(CSV_FILE):
        print(f"❌ HATA: '{CSV_FILE}' dosyası bulunamadı!")
        sys.exit(1)
    
    print("=" * 60)
    print("⚽ FIFA FUTBOLCU VERİTABANI OLUŞTURULUYOR")
    print("=" * 60)
    
    # 1. Veriyi Yükle
    print("\n📂 1/3: CSV dosyası yükleniyor...")
    df = pd.read_csv(CSV_FILE)
    print(f"✅ {len(df)} futbolcu bulundu!")
    
    # TEST İÇİN: Sadece ilk 500 futbolcu (bu satırı kaldırarak tamamını kullanabilirsiniz)
    print(f"✅ TEST MODU: {len(df)} futbolcu kullanılacak")
    
    # Gerekli kolonları seç ve temizle
    df_clean = df[[
        'Name', 'Club', 'Overall', 'Pace', 'Shooting', 
        'Passing', 'Dribbling', 'Defending', 'Physicality'
    ]].copy()
    
    df_clean.fillna({
        'Overall': 0, 'Pace': 0, 'Shooting': 0, 
        'Passing': 0, 'Dribbling': 0, 'Defending': 0, 'Physicality': 0
    }, inplace=True)
    df_clean.fillna('Bilinmiyor', inplace=True)
    
    # 2. Document'leri Oluştur
    print("\n📝 2/3: Futbolcu verileri hazırlanıyor...")
    df_clean['rag_chunk'] = df_clean.apply(create_player_chunk, axis=1)
    data_documents = [
        Document(page_content=chunk) 
        for chunk in df_clean['rag_chunk'].tolist()
    ]
    print(f"✅ {len(data_documents)} futbolcu dokümana dönüştürüldü!")
    
    # 3. Embedding ve Vektör DB Oluştur
    print("\n🔄 3/3: Vektör veritabanı oluşturuluyor...")
    print(f"⚠️  Bu işlem ~{len(data_documents) * 0.5 / 60:.1f} dakika sürebilir...")
    
    embedding_function = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=GEMINI_KEY
    )
    
    # Batch işleme
    total_docs = len(data_documents)
    vectorstore = None
    
    for i in range(0, total_docs, BATCH_SIZE):
        batch = data_documents[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total_docs + BATCH_SIZE - 1) // BATCH_SIZE
        
        try:
            if vectorstore is None:
                # İlk batch
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embedding_function,
                    persist_directory=PERSIST_DIRECTORY,
                    collection_name=COLLECTION_NAME
                )
            else:
                # Sonraki batch'ler
                vectorstore.add_documents(batch)
            
            progress = (i + len(batch)) / total_docs * 100
            print(f"📊 Batch {batch_num}/{total_batches} ✅ ({progress:.1f}% tamamlandı)")
            
            # Rate limit'i aşmamak için bekle
            if i + BATCH_SIZE < total_docs:
                time.sleep(DELAY_BETWEEN_BATCHES)
                
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"⏳ Rate limit! 60 saniye bekleniyor...")
                time.sleep(60)
                # Tekrar dene
                if vectorstore is None:
                    vectorstore = Chroma.from_documents(
                        documents=batch,
                        embedding=embedding_function,
                        persist_directory=PERSIST_DIRECTORY,
                        collection_name=COLLECTION_NAME
                    )
                else:
                    vectorstore.add_documents(batch)
            else:
                print(f"❌ HATA: {e}")
                raise e
    
    print("\n" + "=" * 60)
    print("🎉 VERİTABANI BAŞARIYLA OLUŞTURULDU!")
    print(f"📁 Konum: {PERSIST_DIRECTORY}")
    print(f"📊 Toplam Futbolcu: {len(data_documents)}")
    print("=" * 60)

if __name__ == "__main__":
    create_database()
