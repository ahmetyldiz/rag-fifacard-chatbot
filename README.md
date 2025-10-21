# ⚽ FIFA Kartı Oluşturucu Chatbot

17,000+ futbolcu verisi ile çalışan, yapay zeka destekli FIFA kartı oluşturma uygulaması.

## 📋 Proje Hakkında

Bu proje, kullanıcıların futbolcu adı veya istatistik sorguları yaparak FIFA tarzı oyuncu kartları oluşturmasını sağlayan bir chatbot uygulamasıdır. Google Gemini LLM ve CSV tabanlı veri işleme ile hibrit bir arama sistemi kullanır.

## ✨ Özellikler

- 🔍 **Akıllı Futbolcu Arama**: LLM destekli doğal dil işleme ile futbolcu bulma
- 📊 **İstatistik Karşılaştırma**: En hızlı, en iyi defans, en yüksek fizik gibi sorgular
- 🎨 **Görsel FIFA Kartları**: Gradyan arka planlı, modern tasarımlı oyuncu kartları
- 📈 **Detaylı İstatistikler**: Progress bar'larla görselleştirilmiş 6 farklı stat
- 🇹🇷 **Türkçe Destek**: Türkçe karakter ve dil desteği
- ⚡ **Hızlı Yanıt**: Cache mekanizması ile optimize edilmiş performans
- 🛡️ **Güvenlik**: Rate limiting ve sorgu limiti ile korumalı

## 🛠️ Kullanılan Teknolojiler

- **Streamlit**: Web arayüzü ve chatbot UI
- **LangChain**: RAG pipeline ve LLM entegrasyonu
- **Google Gemini**: Doğal dil işleme (gemini-2.0-flash-exp)
- **ChromaDB**: Vektör veritabanı
- **Pandas**: CSV veri işleme
- **Unidecode**: Türkçe karakter normalizasyonu
- **Google Generative AI Embeddings**: Text embedding (text-embedding-004)

## 🚀 Kurulum

### 1. Gerekli Dosyaları Hazırlayın

Proje dizinine `male_players.csv` dosyasını ekleyin (17,000+ futbolcu verisi içeren CSV).

### 2. Gerekli Paketleri Yükleyin
Virtual environment oluşturun (önerilir)
python -m venv venv
source venv/bin/activate # macOS/Linux

venv\Scripts\activate # Windows
Paketleri yükleyin
pip install -r requirements.txt

**requirements.txt içeriği:**
streamlit
pandas
python-dotenv
langchain
langchain-community
langchain-google-genai
chromadb
unidecode
pysqlite3-binary

### 3. API Anahtarını Ayarlayın

Proje kök dizininde `.env` dosyası oluşturun:

GEMINI_API_KEY=your_google_api_key_here
- **Google API Key**: [Google AI Studio](https://aistudio.google.com/app/apikey) üzerinden ücretsiz alabilirsiniz

### 4. ChromaDB Veritabanını Oluşturun (Opsiyonel)
İleri seviye semantik arama için ChromaDB kullanılabilir. Veritabanı yoksa uygulama otomatik olarak CSV'ye geri döner.

ChromaDB klasörü varsa:
./chroma_db/
text

### 5. Uygulamayı Çalıştırın
streamlit run app.py

Tarayıcınızda otomatik olarak açılacaktır (genellikle http://localhost:8501).


## 📁 Proje Yapısı

.
├── app.py # Ana uygulama dosyası
├── male_players.csv # Futbolcu veri seti (17,000+ oyuncu)
├── requirements.txt # Python bağımlılıkları
├── .env # API anahtarları (git'e eklenmez)
├── chroma_db/ # ChromaDB vektör veritabanı (opsiyonel)
└── README.md # Bu dosya


## 💡 Nasıl Çalışır?

### Veri İşleme Akışı

1. **CSV Yükleme**: `male_players.csv` dosyası Pandas ile yüklenir ve cache'lenir
2. **Sorgu İşleme**: Kullanıcı sorgusu LLM ve kural tabanlı sistemle işlenir
3. **Futbolcu Eşleştirme**: 
   - İlk aşama: Tam metin eşleşmesi
   - İkinci aşama: Türkçe karakter normalizasyonu ile eşleşme
4. **Kart Oluşturma**: HTML/CSS ile dinamik FIFA kartı render edilir
5. **İstatistik Görselleştirme**: Streamlit progress bar'ları ile 6 stat gösterilir

### LLM Preprocessing

- **Akıllı İsim Çıkarma**: Gemini-2.0-flash-exp modeli ile doğal dilden futbolcu adı tespiti
- **Fallback Mekanizması**: LLM başarısız olursa regex ve string manipülasyonu devreye girer
- **Cache Sistemi**: Aynı sorgular 1 saat boyunca cache'den sunulur

## 🎯 Örnek Sorgular

### 🔍 Futbolcu Arama
- "Lionel Messi"
- "Messinin kartı"
- "Kylian Mbappe"
- "Cristiano Ronaldo kartını göster"

### 📊 İstatistik Sorguları
- "En yüksek dereceli futbolcu"
- "En hızlı oyuncu"
- "Fiziği en yüksek oyuncu"
- "En iyi defans"
- "En düşük şutu olan futbolcu"
- "En iyi dribling"

### 💬 Genel Mesajlar
- "Merhaba"
- "Teşekkürler"
- "Nasılsın"

## 📊 Veri Seti Bilgisi

CSV dosyası şu sütunları içermelidir:

| Sütun | Açıklama |
|-------|----------|
| `Name` | Futbolcu adı |
| `Overall` | Genel derece (0-100) |
| `Club` | Kulüp adı |
| `Pace` | Hız (0-100) |
| `Shooting` | Şut (0-100) |
| `Passing` | Pas (0-100) |
| `Dribbling` | Dribling (0-100) |
| `Defending` | Defans (0-100) |
| `Physicality` | Fizik (0-100) |

## ⚙️ Yapılandırma

`app.py` içindeki sabitler:

MAX_QUERIES_PER_SESSION = 20 # Oturum başına maksimum sorgu
RATE_LIMIT_SECONDS = 2 # Sorgular arası minimum süre
PERSIST_DIRECTORY = "./chroma_db" # ChromaDB klasörü
COLLECTION_NAME = "fifa-players" # Koleksiyon adı

## ⚠️ Önemli Notlar

- **Sorgu Limiti**: Her oturumda 20 sorgu yapılabilir (sayfa yenilemesiyle sıfırlanır)
- **Rate Limiting**: Sorgular arası 2 saniye bekleme süresi
- **Cache Mekanizması**: LLM ve CSV yüklemeleri cache'lenir, performans artışı sağlar
- **Türkçe Karakter Desteği**: `unidecode` ile İ→i, Ş→s gibi dönüşümler yapılır
- **Streamlit Cloud**: `pysqlite3` uyumluluk kodu ile cloud deployment desteklenir

## 🚀 Deployment (Streamlit Cloud)

### 1. GitHub'a Yükleyin

git init
git add .
git commit -m "Initial commit"
git remote add origin your-repo-url
git push -u origin main

### 2. Streamlit Cloud'a Deploy Edin

1. [share.streamlit.io](https://share.streamlit.io) adresine gidin
2. GitHub repo'nuzu bağlayın
3. `app.py` dosyasını seçin
4. **Advanced settings** → **Secrets** bölümüne ekleyin:
GEMINI_API_KEY = "your_api_key_here"
5. Deploy butonuna tıklayın

### 3. CSV Dosyasını Yükleyin

CSV dosyası GitHub'da olmalıdır veya alternatif olarak:
- Google Drive/Dropbox link'i kullanın
- Kaggle dataset'i entegre edin
- Hugging Face Datasets kullanın

## 🐛 Sorun Giderme

### ModuleNotFoundError: No module named 'streamlit'
pip install -r requirements.txt


### ChromaDB bağlantı hatası
- ChromaDB yoksa uygulama otomatik olarak CSV moduna geçer
- Sorun olmaz, normal çalışmaya devam eder

### API key hatası
- `.env` dosyasını kontrol edin
- Streamlit Cloud'daysa Secrets bölümünü kontrol edin
- API key'in geçerli olduğundan emin olun

### CSV dosyası bulunamadı
- `male_players.csv` dosyasının proje dizininde olduğundan emin olun
- Dosya adının tam olarak eşleştiğini kontrol edin

### Türkçe karakterler görünmüyor
pip install unidecode


### Rate limit uyarısı
- 2 saniye bekleyin ve tekrar deneyin
- `RATE_LIMIT_SECONDS` değerini azaltabilirsiniz (önerilmez)

## 🎨 Özelleştirme

### Kart Tasarımını Değiştirme

`app.py` içindeki CSS bölümünü düzenleyin:


.fifa-card {
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# Renkleri değiştirin
}


### Sorgu Limitini Artırma

MAX_QUERIES_PER_SESSION = 50 # İstediğiniz değere çıkarın

### Farklı Stat Eklemek

1. CSV'ye yeni sütun ekleyin
2. `preprocess_query()` fonksiyonuna yeni keyword ekleyin
3. `stat_mapping` dictionary'sine ekleyin
4. HTML kartına yeni stat satırı ekleyin

## 📚 İleri Seviye Geliştirmeler

### Önerilen İyileştirmeler

- [ ] Futbolcu fotoğrafları ekleme (API veya scraping)
- [ ] Pozisyon bazlı filtreleme (forvet, defans, kaleci)
- [ ] Karşılaştırma modu (2 futbolcuyu yan yana göster)
- [ ] Kullanıcı favorileri (session state ile)
- [ ] Export PDF/PNG özelliği
- [ ] Dark/Light mode toggle
- [ ] Çoklu dil desteği (İngilizce, İspanyolca)
- [ ] Takım bazlı arama
- [ ] Fiyat/değer bilgisi
- [ ] Transfer geçmişi

### Teknik İyileştirmeler

- [ ] PostgreSQL/MongoDB entegrasyonu
- [ ] Redis cache katmanı
- [ ] FastAPI backend
- [ ] React/Next.js frontend
- [ ] Docker containerization
- [ ] CI/CD pipeline
- [ ] Unit test coverage
- [ ] Logging sistemi

## 📝 Lisans

Bu proje eğitim amaçlıdır.

## 🤝 Katkıda Bulunma

Sorularınız, önerileriniz veya hata bildirimleriniz için GitHub Issues kullanabilirsiniz.

## 👨‍💻 Geliştirici

Bu proje bir bilgisayar mühendisliği öğrencisi tarafından AI/ML öğrenme sürecinde geliştirilmiştir.


**⚽ İyi aramalar!**
https://rag-fifacard-chatbot-6urllsgerlylkurvl62zbh.streamlit.app/
