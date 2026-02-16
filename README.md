## CORN (Mısır Yaprak Hastalığı) Sınıflandırması için DenseNet169 Tabanlı Bir Yaklaşım

### Özet

Bu çalışmada, mısır yaprak hastalıklarını otomatik olarak sınıflandırmak amacıyla **DenseNet169** tabanlı derin öğrenme modelleri kullanılmaktadır. Çalışma kapsamında, dört sınıftan (Healthy, Blight, Common_Rust, Gray_Leaf_Spot) oluşan bir görüntü veri seti üzerinde; (i) ImageNet üzerinde ön-eğitimli DenseNet169 ile ince ayar (fine-tuning), (ii) daha verimli veri okuma ve eğitim altyapısı ile optimize edilmiş bir sürüm ve (iii) ağın tamamen sıfırdan eğitildiği bir senaryo olmak üzere üç farklı deneysel kurgu incelenmiştir. Modeller, eğitim/validasyon/test ayrımı ve çeşitli değerlendirme metrikleri (doğruluk, precision, recall, f1-score, karışıklık matrisi) ile kapsamlı biçimde analiz edilmiştir.

---

### 1. Giriş

Mısır, dünya genelinde en önemli tarımsal ürünlerden biridir ve yaprak hastalıkları, verim kayıplarının başlıca nedenleri arasındadır. Bu nedenle, hastalıkların erken ve doğru teşhisi hem ekonomik hem de gıda güvenliği açısından kritik öneme sahiptir. Geleneksel olarak uzman gözüyle yapılan görsel inceleme hem zaman alıcıdır hem de sübjektif hatalara açıktır. Bu bağlamda, **derin öğrenme tabanlı görüntü sınıflandırma yöntemleri**, tarımsal hastalık tespiti için güçlü bir alternatif sunmaktadır.

Bu proje, mısır yaprak hastalıklarını sınıflandırmak için **DenseNet169** mimarisini temel alan bir yaklaşım önermekte ve üç farklı eğitim stratejisinin (ön-eğitimli, optimize edilmiş veri işleme, sıfırdan eğitim) karşılaştırılmasını amaçlamaktadır.

---

### 2. Veri Seti

Çalışmada, mısır yapraklarının farklı sağlık durumlarını temsil eden dört sınıf kullanılmıştır:

- **Healthy**
- **Blight**
- **Common_Rust**
- **Gray_Leaf_Spot**

Veri, `torchvision.datasets.ImageFolder` formatına uygun olacak şekilde sınıf bazlı klasör yapısında organize edilmiştir. Örnek yapı:

- `data/Healthy`
- `data/Blight`
- `data/Common_Rust`
- `data/Gray_Leaf_Spot`

Google Colab üzerinde, veri varsayılan olarak `"/content/drive/MyDrive/data"` dizininde konumlandırılmaktadır.

#### 2.1. Eğitim / Doğrulama / Test Ayrımı

Veri seti, **%70 eğitim**, **%15 doğrulama** ve **%15 test** olmak üzere üçe bölünmüştür. Bölme işlemi, `torch.utils.data.random_split` fonksiyonu ve sabit bir rastgelelik tohumu (`manual_seed(42)`) kullanılarak tekrarlanabilirlik sağlanacak şekilde yapılmıştır.

---

### 3. Yöntem

Bu çalışmada, temel mimari olarak **DenseNet169** seçilmiştir. Üç ana senaryo uygulanmıştır:

1. **Ön-eğitimli Model ile İnce Ayar (Fine-Tuning)**  
   `DenseNet169.py` dosyasında, ImageNet üzerinde ön-eğitimli (`weights=IMAGENET1K_V1`) DenseNet169 modeli kullanılmış ve son sınıflandırma katmanı, mısır yaprak hastalığı sınıflarının sayısına uyumlu olacak şekilde yeniden düzenlenmiştir.

2. **Veri Okuma ve Eğitim Altyapısının Optimizasyonu (Final Sürüm)**  
   `DenseNet169part2.py` dosyasında, veri önce Google Drive'dan Colab'in yerel diskine (`/content/data`) kopyalanarak I/O darboğazı azaltılmış, `BATCH_SIZE = 16` ve `num_workers=2`, `pin_memory=True` gibi ayarlarla veri yükleme hızı artırılmıştır. Bu betik, çalışmanın **önerilen ve pratikte kullanılabilir nihai sürümü** olarak tasarlanmıştır.

3. **Sıfırdan Eğitim (Random Initialization)**  
   `DenseNet169part3.py` dosyasında, DenseNet169 modeli `weights=None` ile tamamen rastgele ağırlıklarla başlatılarak sıfırdan eğitilmiş ve ön-eğitimli senaryolarla karşılaştırma yapılması hedeflenmiştir.

#### 3.1. Ön İşleme ve Veri Artırma

Tüm senaryolarda aşağıdaki temel dönüşümler kullanılmıştır:

- Yeniden boyutlandırma: `224x224`
- Yatay çevirme (random horizontal flip)
- Rastgele döndürme (random rotation, ±15 derece)
- Normalizasyon: ImageNet ortalama ve standart sapmaları ile

Bu sayede, modelin genelleme yeteneğini artırmak için veri artırma (data augmentation) tekniklerinden yararlanılmıştır.

#### 3.2. Eğitim Ayrıntıları

- Optimizasyon algoritması: **Adam**
- Kayıp fonksiyonu: **Çapraz Entropi (CrossEntropyLoss)**
- Öğrenme oranı: `LR = 0.0001`
- Epoch sayısı: senaryoya göre değişmekle birlikte tipik olarak `EPOCHS = 10–50`
- Donanım: `cuda` mevcutsa GPU, aksi halde CPU

---

### 4. Dosya Yapısı ve Uygulama Ayrıntıları

#### 4.1. `DenseNet169.py` – Ön-eğitimli Model ile Eğitim

- ImageNet ön-eğitimli DenseNet169 (`weights=IMAGENET1K_V1`) kullanır.
- Veri, doğrudan Google Drive üzerindeki `"/content/drive/MyDrive/data"` dizininden okunur.
- Eğitim, doğrulama ve test ayrımı (70/15/15) gerçekleştirilir.
- Eğitim sonunda:
  - Eğitim/doğrulama loss ve doğruluk (accuracy) grafikleri üretilir.
  - Test kümesi için sınıflandırma raporu ve karışıklık matrisi hesaplanır.

#### 4.2. `DenseNet169part2.py` – Optimizasyon ve Final Sürüm

- Veri önce Google Drive'dan Colab yerel diskine kopyalanır:
  - Kaynak: `SRC = "/content/drive/MyDrive/data"`
  - Hedef: `DST = "/content/data"`
- Daha büyük batch size (`BATCH_SIZE = 16`) ve çok iş parçacıklı veri yükleme (`num_workers=2`, `pin_memory=True`) ile eğitim süreci hızlandırılır.
- Eğitim, doğrulama ve test adımları optimize edilmiştir; pratik kullanım için **önerilen ana betik** budur.

#### 4.3. `DenseNet169part3.py` – Sıfırdan Eğitim

- Model, `model = models.densenet169(weights=None)` ile rastgele başlatılır.
- Ön-eğitimli senaryoya kıyasla eğitim süresi ve başarı oranlarının karşılaştırılması için kullanılabilir.

---

### 5. Deneysel Kurulum

#### 5.1. Ortam

- Platform: **Google Colab**
- Donanım: GPU destekli runtime (önerilir)
- İşletim sistemi ve CUDA sürümü, Colab'in varsayılan konfigürasyonuna bağlıdır.

#### 5.2. Kullanılan Kütüphaneler

- **PyTorch**: `torch`, `torchvision`
- **Veri Yükleme**: `DataLoader`, `Dataset`, `ImageFolder`
- **Değerlendirme**: `sklearn.metrics.classification_report`, `confusion_matrix`
- **Görselleştirme**: `matplotlib`, `seaborn`
- **Yardımcı Araçlar**: `numpy`, `time`, `os`, `shutil` (özellikle `DenseNet169part2.py`), `google.colab.drive`

---

### 6. Sonuçlar

Her üç senaryoda da eğitim süreci boyunca epoch bazlı **eğitim/doğrulama kaybı** ve **doğruluk** değerleri izlenmiştir. Eğitim tamamlandıktan sonra:

- Test kümesi üzerinde **sınıflandırma raporu** (precision, recall, f1-score, support) oluşturulmakta,
- Sınıflar arası karışmayı görselleştirmek için **karışıklık matrisi (confusion matrix)** heatmap'i üretilmektedir.

Genel olarak, **ImageNet ön-eğitimli** modelin, sıfırdan eğitime göre daha hızlı yakınsadığı ve sınırlı veri durumlarında daha yüksek performans gösterdiği; optimize edilmiş veri işleme altyapısının ise eğitim süresini anlamlı derecede kısalttığı gözlenmektedir.

---

### 7. Google Colab Üzerinde Çalıştırma

#### 7.1. Verinin Hazırlanması

1. Yerel veri klasörünüzü (içinde `Healthy`, `Blight`, `Common_Rust`, `Gray_Leaf_Spot` alt klasörleri olacak şekilde) ZIP arşivine dönüştürün.
2. Arşivi Google Drive üzerinde `MyDrive/data` altına açın:
   - Örnek: `MyDrive/data/Healthy`, `MyDrive/data/Blight`, vb.

#### 7.2. Notebook Ortamının Hazırlanması

1. Google Colab'de yeni bir notebook açın.
2. İlgili `.py` dosyasının içeriğini bir kod hücresine kopyalayın **veya** dosyayı Colab'e yükleyip komut satırından çalıştırın (örneğin: `!python DenseNet169part2.py`).
3. GPU kullanımı için:
   - `Runtime` → `Change runtime type` → `Hardware accelerator: GPU` seçimini yapın.

---

### 8. Tartışma ve Gelecek Çalışmalar

Bu proje, mısır yaprak hastalıklarının otomatik sınıflandırılmasında **DenseNet169** mimarisinin etkinliğini göstermektedir. Özellikle ön-eğitimli modeller, daha kısa sürede daha yüksek doğruluk oranlarına ulaşabilmektedir. Bununla birlikte:

- Daha büyük ve dengeli veri setleri ile performansın artırılması,
- Farklı mimariler (örneğin EfficientNet, Vision Transformer vb.) ile karşılaştırmalar yapılması,
- Gerçek zamanlı saha uygulamaları için hafif modellerin (mobil uyumlu) tasarlanması

gelecek çalışmalar için önemli araştırma başlıkları olarak değerlendirilebilir.

---

### 9. Sonuç

Sonuç olarak, bu çalışma mısır yaprak hastalıklarının teşhisinde derin öğrenme tabanlı bir çerçeve sunmakta ve farklı eğitim stratejilerinin (ön-eğitimli, optimize edilmiş veri altyapısı, sıfırdan eğitim) pratikte nasıl uygulanabileceğine dair bir yol haritası oluşturmaktadır. Proje betikleri (`DenseNet169.py`, `DenseNet169part2.py`, `DenseNet169part3.py`), Google Colab ortamında kolayca çalıştırılabilecek şekilde yapılandırılmış ve araştırmacıların/öğrencilerin kendi deneylerini hızlıca tekrarlayabilmesine olanak tanımaktadır.
