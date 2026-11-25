from pathlib import Path
from huggingface_hub import snapshot_download

# Hugging Face tercih edildi çünkü veri setine API anahtarı
# veya kullanıcı hesabı gerektirmeden erişebiliyoruz.


VERI_SETI_ADRESI = "Xuban/rock-paper-scissors-roboflow"

# Veri setini proje klasörümüzün içinde tutuyoruz.
# Böylece bilgisayardan bilgisayara değişen dosya yollarına bağımlı kalmıyoruz.
PROJE_KLASORU = Path(__file__).resolve().parent
VERI_KLASORU = PROJE_KLASORU / "veri_seti" / "kaynak"


def veri_setini_indir():

    print("Veri seti indiriliyor...")

    snapshot_download(
        repo_id=VERI_SETI_ADRESI,
        repo_type="dataset",
        local_dir=str(VERI_KLASORU),
        allow_patterns=[
            "data.yaml",
            "images/**",
            "labels/**"
        ]
    )

    print("Veri seti indirildi!")

    # PDF'de eğitim, doğrulama ve test kümeleri ayrı isteniyor.
    # Şimdilik kaynak veri setinin train ve val bölümlerini
    # kontrol ediyoruz. Bağımsız test kümesini sonraki adımda oluşturacağız.

    for bolum in ["train", "val"]:

        goruntu_klasoru = VERI_KLASORU / "images" / bolum
        etiket_klasoru = VERI_KLASORU / "labels" / bolum

        goruntu_sayisi = sum(
            1 for dosya in goruntu_klasoru.glob("*")
            if dosya.suffix.lower() in [".jpg", ".jpeg", ".png"]
        )

        etiket_sayisi = len(list(etiket_klasoru.glob("*.txt")))

        print(
            f"{bolum}: {goruntu_sayisi} görüntü, "
            f"{etiket_sayisi} etiket dosyası"
        )

        if goruntu_sayisi == 0 or etiket_sayisi == 0:
            raise RuntimeError(
                f"{bolum} klasöründe görüntü veya etiket bulunamadı!"
            )


if __name__ == "__main__":
    veri_setini_indir()