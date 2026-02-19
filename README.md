Yerel şirket dokümanları üzerinde çalışan, RAG (Retrieval-Augmented Generation) tabanlı bir yapay zeka asistanı. Kullanıcıların yüklediği belgeleri analiz ederek sorulara yalnızca bağlam dahilinde, kaynak göstererek yanıt verir.

## Özellikler

* **Çoklu Format Desteği:** PDF, DOCX, TXT ve MD dosyalarını işleme yeteneği.
* **Modern Altyapı:** Arayüz için Streamlit, LLM ve embedding işlemleri için Google GenAI, vektör veritabanı için ChromaDB ve doküman yönetimi için LangChain entegrasyonu.
* **Doğru ve Güvenilir Yanıtlar:** Yanıtları oluştururken referans alınan belge isimlerini kaynak çipleri olarak kullanıcıya sunma.

## Kurulum

1.  Depoyu klonlayın:
    ```bash
    git clone <repo-url>
    cd <repo-adi>
    ```

2.  Gerekli bağımlılıkları yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3.  Çevre değişkenlerini yapılandırın:
    `.env.example` dosyasının adını `.env` olarak değiştirin ve Gemini API anahtarınızı tanımlayın:
    ```env
    GEMINI_API_KEY="sizin_api_anahtariniz_buraya"
    ```
    *(Not: Proje, `.env` dosyasındaki `GEMINI_API_KEY` değişkenini okuyarak çalışır.)*

## Kullanım

1.  Uygulamayı başlatın:
    ```bash
    streamlit run app.py
    ```
2.  Sol menüdeki **Dosya Yükle** alanından belgelerinizi seçin.
3.  **Dosyaları İşle** butonuna tıklayarak vektör veritabanını oluşturun.
4.  Alt kısımdaki sohbet ekranından belgelerinize dair sorular sorun. Asistan ilgili doküman parçalarını tarayarak yanıtlayacaktır.
