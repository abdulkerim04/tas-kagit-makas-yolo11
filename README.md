\# Taş-Kâğıt-Makas Nesne Tespiti



Bu projede YOLO11n kullanılarak taş, kâğıt ve makas el hareketlerini tespit eden bir nesne tespiti modeli geliştirilmiştir.



\## Kullanılan Teknolojiler

\- Python

\- Ultralytics YOLO11n

\- PyTorch

\- Gradio



\## Projeyi Çalıştırma



Önce gerekli kütüphaneleri yükleyin:



```bash

pip install -r requirements.txt

```



Ardından uygulamayı başlatın:



```bash

python uygulama.py

```



Uygulama tarayıcıda açılır. Bir fotoğraf yükleyerek modelin tahminlerini görüntüleyebilirsiniz.



Eğitilmiş model `modeller/best.pt` konumundadır. Uygulamayı çalıştırmak için modeli yeniden eğitmek gerekmez.



\## Model ve Eğitim Bilgileri

\- Model: YOLO11n

\- Sınıflar: Taş, Kâğıt, Makas

\- Görüntü boyutu: 416

\- Epoch: 15

\- Batch size: 8

\- Optimizer: AdamW

\- Öğrenme oranı: 0.001



\## Test Sonuçları

\- Test görüntüsü: 149

\- Precision: %93,81

\- Recall: %94,09

\- mAP@50: %96,19

\- mAP@50-95: %61,26



\## Proje Dosyaları

\- `uygulama.py`: Gradio uygulaması

\- `model\_egit.py`: Model eğitimi

\- `model\_degerlendir.py`: Model değerlendirmesi

\- `veri\_indir.py`: Veri setini indirme

\- `veri\_hazirla.py`: Veri setini hazırlama

\- `modeller/best.pt`: Eğitilmiş model

\- `rapor\_gorselleri/`: Beş farklı test görüntüsü ve sonuçları

