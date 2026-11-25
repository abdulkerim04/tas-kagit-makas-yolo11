from pathlib import Path

# Gradio: Flask'a göre daha az kodla model arayüzü
# oluşturabildiğimiz için tercih edildi.
import gradio as gr

from PIL import Image
from ultralytics import YOLO


# ==================================================
# 1. MODEL YÜKLEME
# ==================================================

PROJE_KLASORU = Path(__file__).resolve().parent

MODEL_DOSYASI = PROJE_KLASORU / "modeller" / "best.pt"

if not MODEL_DOSYASI.exists():
    raise FileNotFoundError("Eğitilmiş model bulunamadı!")

model = YOLO(str(MODEL_DOSYASI))

SINIFLAR = {
    0: "Taş",
    1: "Kağıt",
    2: "Makas"
}


# ==================================================
# 2. NESNE TESPİTİ
# ==================================================

def tahmin_yap(gorsel):

    if gorsel is None:
        return None, "Lütfen bir fotoğraf yükleyin."

    # Eğitimde kullanılan görüntü boyutu korunur.
    # Confidence 0.25: Düşük güvenli tahminleri filtreler.
    sonuc = model.predict(
        source=gorsel,
        imgsz=416,
        conf=0.25,
        device="cpu",
        verbose=False
    )[0]

    # YOLO kutuları görüntünün üzerine çizer.
    # BGR görüntüsü, Gradio için RGB'ye dönüştürülür.
    isaretli_gorsel = Image.fromarray(
        sonuc.plot()[:, :, ::-1]
    )

    tahminler = []

    for kutu in sonuc.boxes:

        sinif = int(kutu.cls.item())
        guven = float(kutu.conf.item())

        tahminler.append(
            f"{SINIFLAR[sinif]} - Güven: %{guven * 100:.1f}"
        )

    if not tahminler:
        return isaretli_gorsel, "Nesne tespit edilemedi."

    return isaretli_gorsel, "\n".join(tahminler)


# ==================================================
# 3. GRADIO ARAYÜZÜ
# ==================================================

# Gradio Interface tercih edildi:
# Ayrı HTML/CSS geliştirmeden fotoğraf yükleme
# ve model sonuçlarını görüntüleme imkânı sağlar.

uygulama = gr.Interface(
    fn=tahmin_yap,

    inputs=gr.Image(
        type="pil",
        label="Fotoğraf Yükle"
    ),

    outputs=[
        gr.Image(label="Tespit Sonucu"),
        gr.Textbox(label="Model Tahminleri")
    ],

    title="Taş - Kağıt - Makas Nesne Tespiti",

    description=(
        "YOLO11n ile geliştirilmiş nesne tespiti uygulaması. "
        "Bir fotoğraf yükleyerek el hareketlerini tespit edebilirsiniz."
    )
)


# ==================================================
# 4. UYGULAMAYI BAŞLAT
# ==================================================

if __name__ == "__main__":
    uygulama.launch(
        share=False,
        inbrowser=True
    )