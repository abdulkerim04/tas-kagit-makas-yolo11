from pathlib import Path
from time import perf_counter
from datetime import datetime
import json

# YOLO11n: Faster R-CNN'e göre daha hafif yapısı ve düşük hesaplama
# maliyeti nedeniyle tercih edildi.
from ultralytics import YOLO

# PyTorch: Ultralytics YOLO11'in doğrudan kullandığı
# derin öğrenme altyapısı olduğu için tercih edildi.
import torch


# ==========================================================
# 1. PROJE YOLLARI
# ==========================================================

# pathlib: Sabit bilgisayar yolları yerine taşınabilir
# proje yolları oluşturmak için kullanılır.

PROJE_KLASORU = Path(__file__).resolve().parent

VERI_KLASORU = PROJE_KLASORU / "veri_seti" / "hazir"
VERI_DOSYASI = VERI_KLASORU / "data.yaml"

SONUC_KLASORU = PROJE_KLASORU / "sonuclar" / "egitim"


# ==========================================================
# 2. İLK EĞİTİM DENEYİ - 1 EPOCH
# ==========================================================

# Model         : YOLO11n
# Epoch         : 1
# Batch Size    : 8
# Image Size    : 416
# Optimizer     : AdamW
# Learning Rate : 0.001
# Device        : CPU
#
# Eğitim Süresi : 3.90 dakika
# Precision     : 0.175
# Recall        : 0.487
# mAP50         : 0.277
# mAP50-95      : 0.134
#
# Sınıf bazlı mAP50:
# Rock          : 0.324
# Paper         : 0.339
# Scissors      : 0.167
#
# Sonuç: Eğitim başarıyla tamamlandı.
# Performansı geliştirmek için 15 epoch ana eğitim planlandı.
# Bu deneyin sonuçları hiz_testi klasöründe saklanmaktadır.


# ==========================================================
# 3. ANA EĞİTİM AYARLARI
# ==========================================================

# 15 epoch : Eğitim süresi ve öğrenme dengesi.
# 416 px   : 640 px'e göre daha düşük hesaplama maliyeti.
# Batch 8  : Kontrollü bellek kullanımı.
# AdamW    : Adaptif optimizasyon ve weight decay desteği.
# LR 0.001 : Kontrollü başlangıç öğrenme oranı.
# CPU      : İlk denemede yeterli eğitim hızı elde edildi.
# Seed 42  : Deneylerin tekrar edilebilirliğini artırır.

EGITIM_AYARLARI = {
    "epochs": 15,
    "imgsz": 416,
    "batch": 8,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "device": "cpu",
    "workers": 0,
    "seed": 42
}


# ==========================================================
# 4. MODEL EĞİTİMİ
# ==========================================================

def modeli_egit():

    if not VERI_DOSYASI.exists():
        raise FileNotFoundError(
            "Hazırlanmış veri seti bulunamadı!"
        )

    print("\nYOLO11n modeli yükleniyor...")

    # Önceden eğitilmiş ağırlıklarla başlangıç (Transfer Learning).
    # İlk 1 epoch denemesindeki ağırlıklar kullanılmaz.
    model = YOLO("yolo11n.pt")

    print("\nEğitim ayarları:")

    for ayar, deger in EGITIM_AYARLARI.items():
        print(f"{ayar}: {deger}")

    print("\n15 epoch ana eğitim başlıyor...")

    baslangic = perf_counter()

    # YOLO11 nesne tespiti için Box, Classification ve DFL
    # kayıp bileşenlerini kullanır.
    # Eğitim grafikleri raporlama amacıyla kaydedilir.

    model.train(
        data=str(VERI_DOSYASI),
        project=str(SONUC_KLASORU),
        name="ana_egitim",
        plots=True,
        **EGITIM_AYARLARI
    )

    egitim_suresi = perf_counter() - baslangic

    kayit_klasoru = Path(model.trainer.save_dir)

    en_iyi_model = kayit_klasoru / "weights" / "best.pt"

    if not en_iyi_model.exists():
        raise FileNotFoundError(
            "Eğitim tamamlandı ancak best.pt bulunamadı!"
        )

    print("\nAna eğitim tamamlandı!")
    print(f"Eğitim süresi: {egitim_suresi / 60:.2f} dakika")


    # ======================================================
    # 5. MODEL DOĞRULAMA
    # ======================================================

    # best.pt: Doğrulama performansına göre seçilen en iyi model.
    # Test kümesi bu aşamada kullanılmaz.

    print("\nEn iyi model doğrulanıyor...")

    egitilmis_model = YOLO(str(en_iyi_model))

    metrikler = egitilmis_model.val(
        data=str(VERI_DOSYASI),
        split="val",
        imgsz=EGITIM_AYARLARI["imgsz"],
        batch=EGITIM_AYARLARI["batch"],
        device=EGITIM_AYARLARI["device"],
        workers=0,
        plots=False
    )


    # ======================================================
    # 6. EĞİTİM RAPORU
    # ======================================================

    # JSON: Eğitim ayarlarının ve ölçülen sonuçların
    # yapılandırılmış biçimde saklanması için tercih edildi.

    goruntu_uzantilari = {".jpg", ".jpeg", ".png"}

    def goruntu_sayisi(bolum):
        klasor = VERI_KLASORU / "images" / bolum

        return sum(
            1 for dosya in klasor.iterdir()
            if dosya.is_file()
            and dosya.suffix.lower() in goruntu_uzantilari
        )

    rapor = {
        "proje": "Taş-Kağıt-Makas Nesne Tespiti",
        "gorev": "Object Detection",
        "model": "YOLO11n",

        "framework": "Ultralytics / PyTorch",
        "pytorch_surumu": torch.__version__,

        "egitim_ortami": "CPU",

        "tarih": datetime.now().isoformat(
            timespec="seconds"
        ),

        "veri_seti": {
            "kaynak": "Hugging Face - Xuban",
            "egitim": goruntu_sayisi("train"),
            "dogrulama": goruntu_sayisi("val"),
            "test": goruntu_sayisi("test"),
            "siniflar": [
                "rock",
                "paper",
                "scissors"
            ]
        },

        "hiperparametreler": EGITIM_AYARLARI,

        "egitim_suresi_saniye": round(
            egitim_suresi, 2
        ),

        "egitim_suresi_dakika": round(
            egitim_suresi / 60, 2
        ),

        "performans_dogrulama": {
            "precision": round(
                float(metrikler.box.mp), 4
            ),

            "recall": round(
                float(metrikler.box.mr), 4
            ),

            "mAP50": round(
                float(metrikler.box.map50), 4
            ),

            "mAP50_95": round(
                float(metrikler.box.map), 4
            )
        },

        "tahmin_suresi_ms": round(
            float(metrikler.speed.get("inference", 0)), 2
        ),

        "model_dosyasi": str(en_iyi_model)
    }

    rapor_dosyasi = kayit_klasoru / "egitim_ozeti.json"

    with open(
        rapor_dosyasi,
        "w",
        encoding="utf-8"
    ) as dosya:

        json.dump(
            rapor,
            dosya,
            indent=4,
            ensure_ascii=False
        )


    # ======================================================
    # 7. SONUÇLAR
    # ======================================================

    print("\n========== ANA EĞİTİM SONUÇLARI ==========")

    print(f"Eğitim süresi : {egitim_suresi / 60:.2f} dakika")
    print(f"Precision     : {metrikler.box.mp:.4f}")
    print(f"Recall        : {metrikler.box.mr:.4f}")
    print(f"mAP50         : {metrikler.box.map50:.4f}")
    print(f"mAP50-95      : {metrikler.box.map:.4f}")

    print(
        f"Tahmin süresi : "
        f"{metrikler.speed.get('inference', 0):.2f} ms"
    )

    print(f"\nEn iyi model  : {en_iyi_model}")
    print(f"Rapor dosyası : {rapor_dosyasi}")

    print("\nTüm işlemler tamamlandı!")


# ==========================================================
# 8. PROGRAMI BAŞLAT
# ==========================================================

if __name__ == "__main__":
    modeli_egit()