from pathlib import Path
from datetime import datetime
import json

from ultralytics import YOLO


# ==========================================================
# 1. PROJE YOLLARI
# ==========================================================

PROJE_KLASORU = Path(__file__).resolve().parent

VERI_KLASORU = PROJE_KLASORU / "veri_seti" / "hazir"
VERI_DOSYASI = VERI_KLASORU / "data.yaml"

MODEL_DOSYASI = (
    PROJE_KLASORU
    / "sonuclar"
    / "egitim"
    / "ana_egitim"
    / "weights"
    / "best.pt"
)

SONUC_KLASORU = PROJE_KLASORU / "sonuclar" / "test"


# ==========================================================
# 2. TEST AYARLARI
# ==========================================================

# Eğitimde kullanılan görüntü boyutu korunuyor.
# CPU tercihimiz değişmedi.
# IoU 0.50: Nesne kutularını eşleştirmek için kullanılan eşik.
# Confidence 0.25: Ayrı IoU analizinde düşük güvenli
# tahminleri elemek için seçildi.

TEST_AYARLARI = {
    "imgsz": 416,
    "batch": 8,
    "device": "cpu",
    "workers": 0
}

IOU_ESIGI = 0.50
GUVEN_ESIGI = 0.25


# ==========================================================
# 3. IoU HESAPLAMA
# ==========================================================

def iou_hesapla(kutu1, kutu2):

    x1 = max(kutu1[0], kutu2[0])
    y1 = max(kutu1[1], kutu2[1])

    x2 = min(kutu1[2], kutu2[2])
    y2 = min(kutu1[3], kutu2[3])

    kesisim = max(0, x2 - x1) * max(0, y2 - y1)

    alan1 = (
        (kutu1[2] - kutu1[0])
        * (kutu1[3] - kutu1[1])
    )

    alan2 = (
        (kutu2[2] - kutu2[0])
        * (kutu2[3] - kutu2[1])
    )

    birlesim = alan1 + alan2 - kesisim

    if birlesim <= 0:
        return 0.0

    return kesisim / birlesim


# ==========================================================
# 4. GERÇEK ETİKETLERİ OKUMA
# ==========================================================

def gercek_etiketleri_oku(dosya, genislik, yukseklik):

    gercekler = []

    for satir in dosya.read_text(
        encoding="utf-8"
    ).splitlines():

        if not satir.strip():
            continue

        sinif, x, y, w, h = map(float, satir.split())

        # YOLO'nun normalize edilmiş xywh koordinatlarını
        # piksel tabanlı xyxy koordinatlarına dönüştürüyoruz.

        x1 = (x - w / 2) * genislik
        y1 = (y - h / 2) * yukseklik

        x2 = (x + w / 2) * genislik
        y2 = (y + h / 2) * yukseklik

        gercekler.append({
            "sinif": int(sinif),
            "kutu": [x1, y1, x2, y2]
        })

    return gercekler


# ==========================================================
# 5. IoU ANALİZİ
# ==========================================================

def iou_analizi(model):

    goruntu_klasoru = VERI_KLASORU / "images" / "test"
    etiket_klasoru = VERI_KLASORU / "labels" / "test"

    eslesen_iou = []

    dogru_tespit = 0
    yanlis_tespit = 0
    kacirilan_nesne = 0
    goruntu_sayisi = 0

    # Her görüntüde aynı sınıftaki tahmin ve gerçek
    # kutuları bire bir eşleştiriliyor.
    # Eşleşme için IoU >= 0.50 koşulu kullanılıyor.

    for sonuc in model.predict(
        source=str(goruntu_klasoru),
        imgsz=416,
        device="cpu",
        conf=GUVEN_ESIGI,
        stream=True,
        verbose=False
    ):

        goruntu_sayisi += 1

        goruntu_yolu = Path(sonuc.path)

        etiket_dosyasi = (
            etiket_klasoru / f"{goruntu_yolu.stem}.txt"
        )

        if not etiket_dosyasi.exists():
            raise FileNotFoundError(
                f"Etiket bulunamadı: {etiket_dosyasi}"
            )

        yukseklik, genislik = sonuc.orig_shape

        gercekler = gercek_etiketleri_oku(
            etiket_dosyasi,
            genislik,
            yukseklik
        )

        eslesenler = set()

        kutular = sonuc.boxes.xyxy.cpu().tolist()
        siniflar = sonuc.boxes.cls.cpu().tolist()
        guvenler = sonuc.boxes.conf.cpu().tolist()

        # Yüksek güvenli tahminlere öncelik veriyoruz.
        siralama = sorted(
            range(len(kutular)),
            key=lambda i: guvenler[i],
            reverse=True
        )

        for i in siralama:

            tahmin_kutusu = kutular[i]
            tahmin_sinifi = int(siniflar[i])

            en_iyi_iou = 0.0
            en_iyi_indeks = None

            for j, gercek in enumerate(gercekler):

                if j in eslesenler:
                    continue

                if tahmin_sinifi != gercek["sinif"]:
                    continue

                iou = iou_hesapla(
                    tahmin_kutusu,
                    gercek["kutu"]
                )

                if iou > en_iyi_iou:
                    en_iyi_iou = iou
                    en_iyi_indeks = j

            if (
                en_iyi_indeks is not None
                and en_iyi_iou >= IOU_ESIGI
            ):

                eslesenler.add(en_iyi_indeks)

                eslesen_iou.append(en_iyi_iou)
                dogru_tespit += 1

            else:
                yanlis_tespit += 1

        kacirilan_nesne += (
            len(gercekler) - len(eslesenler)
        )

    ortalama_iou = (
        sum(eslesen_iou) / len(eslesen_iou)
        if eslesen_iou else 0.0
    )

    return {
        "goruntu_sayisi": goruntu_sayisi,
        "ortalama_eslesen_iou": round(ortalama_iou, 4),
        "dogru_tespit": dogru_tespit,
        "yanlis_tespit": yanlis_tespit,
        "kacirilan_nesne": kacirilan_nesne,
        "iou_esigi": IOU_ESIGI,
        "guven_esigi": GUVEN_ESIGI
    }


# ==========================================================
# 6. MODEL DEĞERLENDİRME
# ==========================================================

def modeli_degerlendir():

    if not MODEL_DOSYASI.exists():
        raise FileNotFoundError(
            "Eğitilmiş best.pt modeli bulunamadı!"
        )

    if not VERI_DOSYASI.exists():
        raise FileNotFoundError(
            "Veri seti bulunamadı!"
        )

    print("\nEğitilmiş model yükleniyor...")

    model = YOLO(str(MODEL_DOSYASI))

    print("\nBağımsız test başlıyor...")

    # Test kümesi modelin eğitiminde kullanılmadı.
    # mAP hesaplamasını Ultralytics'in doğrulama
    # altyapısıyla gerçekleştiriyoruz.

    kayit_adi = datetime.now().strftime(
        "test_%Y%m%d_%H%M%S"
    )

    metrikler = model.val(
        data=str(VERI_DOSYASI),
        split="test",
        project=str(SONUC_KLASORU),
        name=kayit_adi,
        plots=True,
        **TEST_AYARLARI
    )

    print("\nIoU analizi yapılıyor...")

    iou_sonuclari = iou_analizi(model)

    # ======================================================
    # 7. RAPORLAMA
    # ======================================================

    rapor = {
        "proje": "Taş-Kağıt-Makas Nesne Tespiti",
        "model": "YOLO11n",
        "degerlendirme_kumesi": "test",

        "test_ayarlari": TEST_AYARLARI,

        "performans": {
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

        "iou_analizi": iou_sonuclari,

        "tahmin_suresi_ms": round(
            float(metrikler.speed.get("inference", 0)), 2
        ),

        "model_dosyasi": str(MODEL_DOSYASI)
    }

    kayit_klasoru = SONUC_KLASORU / kayit_adi
    kayit_klasoru.mkdir(parents=True, exist_ok=True)

    rapor_dosyasi = kayit_klasoru / "test_ozeti.json"

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
    # 8. SONUÇLAR
    # ======================================================

    print("\n========== TEST SONUÇLARI ==========")

    print(f"Precision     : {metrikler.box.mp:.4f}")
    print(f"Recall        : {metrikler.box.mr:.4f}")
    print(f"mAP50         : {metrikler.box.map50:.4f}")
    print(f"mAP50-95      : {metrikler.box.map:.4f}")

    print(
        f"Ortalama IoU  : "
        f"{iou_sonuclari['ortalama_eslesen_iou']:.4f}"
    )

    print(
        f"Doğru tespit  : "
        f"{iou_sonuclari['dogru_tespit']}"
    )

    print(
        f"Yanlış tespit : "
        f"{iou_sonuclari['yanlis_tespit']}"
    )

    print(
        f"Kaçırılan     : "
        f"{iou_sonuclari['kacirilan_nesne']}"
    )

    print(
        f"Tahmin süresi : "
        f"{metrikler.speed.get('inference', 0):.2f} ms"
    )

    print(f"\nRapor: {rapor_dosyasi}")
    print("\nTest tamamlandı!")


if __name__ == "__main__":
    modeli_degerlendir()