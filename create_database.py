"""
FIFA Futbolcu Veritabanı Oluşturma Scripti
"""

import os
import sys
import shutil
import pandas as pd
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# ==================== YAPILANDIRMA ====================
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
CSV_FILE = 'male_players.csv'
PERSIST_DIRECTORY = "./chroma_db"
COLLECTION_NAME = "fifa-players"
BATCH_SIZE = 100
DELAY_BETWEEN_BATCHES = 2

# ==================== YARDIMCI FONKSİYONLAR ====================

def create_player_chunk(row):
    """Futbolcu verisini RAG için optimize edilmiş metin formatına çevirir."""
    return (
        f"Futbolcu Adı: {row['Name']}. "
        f"Kulüp: {row['Club']}. "
        f"Genel Reyting (Overall): {int(row['Overall'])}. "
        f"FIFA Kart İstatistikleri: "
        f"Hız (PAC): {int(row['Pace'])}, "
        f"Şut (SHO): {int(row['Shooting'])}, "
        f"Pas (PAS): {int(row['Passing'])}, "
        f"Dribbling (DRI): {int(row['Dribbling'])}, "
        f"Defans (DEF): {int(row['Defending'])}, "
        f"Fizik (PHY): {int(row['Physicality'])}."
    )

def clean_database_directory(db_path):
    """Veritabanı klasörünü temizler."""
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
            print(f"✅ Eski '{db_path}' klasörü temizlendi.")
        except Exception as e:
            print(f"⚠️  Uyarı: '{db_path}' silinemedi: {e}")
            print("Devam ediliyor...")

# ==================== ANA FONKSİYON ====================

def create_database():
    """Vektör veritabanını oluşturur."""
    
    # API Key Kontrolü
    if not GEMINI_KEY:
        print("❌ HATA: GEMINI_API_KEY bulunamadı!")
        print("Lütfen .env dosyanızı kontrol edin.")
        sys.exit(1)
    
    # CSV Kontrolü
    if not os.path.exists(CSV_FILE):
        print(f"❌ HATA: '{CSV_FILE}' dosyası bulunamadı!")
        sys.exit(1)
    
    print("=" * 60)
    print("⚽ FIFA FUTBOLCU VERİTABANI OLUŞTURULUYOR")
    print("=" * 60)
    
    # 1. Temiz Başlangıç
    print("\n🧹 1/4: Eski veritabanı temizleniyor...")
    clean_database_directory(PERSIST_DIRECTORY)
    
    # 2. CSV Yükleme
    print("\n📂 2/4: CSV dosyası yükleniyor...")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"✅ {len(df)} futbolcu bulundu!")
    except Exception as e:
        print(f"❌ CSV yükleme hatası: {e}")
        sys.exit(1)
    
    # 3. Veri Hazırlama
    print("\n📝 3/4: Futbolcu verileri hazırlanıyor...")
    
    required_cols = [
        'Name', 'Club', 'Overall', 'Pace', 'Shooting', 
        'Passing', 'Dribbling', 'Defending', 'Physicality'
    ]
    
    df_clean = df[required_cols].copy()
    
    # Eksik değerleri doldur
    df_clean.fillna({
        'Overall': 0, 'Pace': 0, 'Shooting': 0, 
        'Passing': 0, 'Dribbling': 0, 'Defending': 0, 'Physicality': 0
    }, inplace=True)
    df_clean.fillna('Bilinmiyor', inplace=True)
    
    # RAG formatına çevir
    df_clean['rag_chunk'] = df_clean.apply(create_player_chunk, axis=1)
    
    # Document objelerine dönüştür (METADATA ile)
    data_documents = [
        Document(
            page_content=chunk,
            metadata={
                "name": row['Name'],
                "club": row['Club'],
                "overall": int(row['Overall']),
                "pace": int(row['Pace']),
                "shooting": int(row['Shooting']),
                "passing": int(row['Passing']),
                "dribbling": int(row['Dribbling']),
                "defending": int(row['Defending']),
                "physicality": int(row['Physicality'])
            }
        ) 
        for chunk, (_, row) in zip(df_clean['rag_chunk'].tolist(), df_clean.iterrows())
    ]
    
    print(f"✅ {len(data_documents)} futbolcu dokümana dönüştürüldü!")
    
    # 4. Embedding ve Vektör DB Oluşturma
    print("\n🔄 4/4: Vektör veritabanı oluşturuluyor...")
    print(f"⏳ Tahmini süre: ~{len(data_documents) * 0.5 / 60:.1f} dakika")
    
    try:
        # Embedding fonksiyonu
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
                
                # Rate limit korumasi
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
        print(f"📋 Collection: {COLLECTION_NAME}")
        print("=" * 60)
        
        return vectorstore
        
    except Exception as e:
        print(f"\n❌ Veritabanı oluşturma başarısız: {e}")
        sys.exit(1)

# ==================== ÇALIŞTIRMA ====================

if __name__ == "__main__":
    load_dotenv()
    create_database()
