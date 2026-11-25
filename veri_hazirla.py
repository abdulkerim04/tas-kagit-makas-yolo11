from pathlib import Path
import random
import math
import shutil
import yaml

# Dosya yollarını bilgisayara özel tanımlamak yerine pathlib kullanıyoruz.
# Böylece proje başka bir bilgisayarda da aynı klasör mantığıyla çalışabilir.

PROJE_KLASORU = Path(__file__).resolve().parent
KAYNAK_KLASOR = PROJE_KLASORU / "veri_seti" / "kaynak"
HAZIR_KLASOR = PROJE_KLASORU / "veri_seti" / "hazir"

TEST_ORANI = 0.10
RASTGELE_TOHUM = 42

GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png"}


def veri_kontrol(bolum):

    goruntu_klasoru = KAYNAK_KLASOR / "images" / bolum
    etiket_klasoru = KAYNAK_KLASOR / "labels" / bolum

    if not goruntu_klasoru.is_dir() or not etiket_klasoru.is_dir():
        raise FileNotFoundError(f"{bolum} klasörleri bulunamadı.")

    goruntuler = sorted(
        dosya for dosya in goruntu_klasoru.iterdir()
        if dosya.suffix.lower() in GORUNTU_UZANTILARI
    )

    if not goruntuler:
        raise ValueError(f"{bolum} klasöründe görüntü yok.")

    goruntu_adlari = [dosya.stem for dosya in goruntuler]
    etiket_adlari = {
        dosya.stem for dosya in etiket_klasoru.glob("*.txt")
    }

    if (
        len(goruntu_adlari) != len(set(goruntu_adlari))
        or set(goruntu_adlari) != etiket_adlari
    ):
        raise ValueError(
            f"{bolum}: Görüntü ve etiket dosyaları eşleşmiyor."
        )

    for goruntu in goruntuler:

        etiket = etiket_klasoru / f"{goruntu.stem}.txt"

        for satir in etiket.read_text(
            encoding="utf-8"
        ).splitlines():

            if not satir.strip():
                continue

            parcalar = satir.split()

            if len(parcalar) != 5:
                raise ValueError(f"Geçersiz etiket: {etiket}")

            try:
                sinif = int(parcalar[0])
                koordinatlar = [
                    float(deger) for deger in parcalar[1:]
                ]
            except ValueError as hata:
                raise ValueError(
                    f"Sayısal etiket hatası: {etiket}"
                ) from hata

            if sinif not in (0, 1, 2):
                raise ValueError(f"Geçersiz sınıf: {etiket}")

            if not all(
                0 <= deger <= 1 for deger in koordinatlar
            ):
                raise ValueError(
                    f"Geçersiz koordinat: {etiket}"
                )

            if koordinatlar[2] <= 0 or koordinatlar[3] <= 0:
                raise ValueError(
                    f"Geçersiz kutu boyutu: {etiket}"
                )

    print(
        f"{bolum}: {len(goruntuler)} görüntü kontrol edildi."
    )

    return goruntuler


def dosyalari_kopyala(goruntuler, kaynak_bolum, hedef_bolum):

    goruntu_hedefi = HAZIR_KLASOR / "images" / hedef_bolum
    etiket_hedefi = HAZIR_KLASOR / "labels" / hedef_bolum

    goruntu_hedefi.mkdir(parents=True, exist_ok=True)
    etiket_hedefi.mkdir(parents=True, exist_ok=True)

    for goruntu in goruntuler:

        etiket = (
            KAYNAK_KLASOR
            / "labels"
            / kaynak_bolum
            / f"{goruntu.stem}.txt"
        )

        shutil.copy2(goruntu, goruntu_hedefi / goruntu.name)
        shutil.copy2(etiket, etiket_hedefi / etiket.name)


def veri_setini_hazirla():

    print("Veri seti kontrol ediliyor...")

    # Kaynak sınıfları doğrulayarak yanlış veri setiyle
    # eğitim yapma ihtimalini azaltıyoruz.
    kaynak_bilgisi = yaml.safe_load(
        (KAYNAK_KLASOR / "data.yaml").read_text(
            encoding="utf-8"
        )
    )

    siniflar = {
        0: "rock",
        1: "paper",
        2: "scissors"
    }

    if kaynak_bilgisi.get("names") != siniflar:
        raise ValueError("Veri setindeki sınıflar beklenenden farklı.")

    egitim_verileri = veri_kontrol("train")
    dogrulama_verileri = veri_kontrol("val")

    # Sabit tohum kullandığımız için program her çalıştırıldığında
    # aynı eğitim ve test ayrımını üretir.
    # Bu tercih deneyin tekrar edilebilirliğini artırır.

    random.Random(RASTGELE_TOHUM).shuffle(egitim_verileri)

    test_sayisi = math.ceil(
        len(egitim_verileri) * TEST_ORANI
    )

    test_verileri = egitim_verileri[:test_sayisi]
    egitim_verileri = egitim_verileri[test_sayisi:]

    # Daha önce hazırlanmış verileri yanlışlıkla değiştirmiyoruz.
    if HAZIR_KLASOR.exists():
        raise FileExistsError(
            "Hazır veri klasörü zaten mevcut. "
            "Mevcut dosyaları kontrol et."
        )

    print("\nVeriler kopyalanıyor...")

    dosyalari_kopyala(egitim_verileri, "train", "train")
    dosyalari_kopyala(dogrulama_verileri, "val", "val")
    dosyalari_kopyala(test_verileri, "train", "test")

    # YAML dosyası, YOLO'ya veri kümelerinin konumlarını
    # ve modelin öğreneceği sınıfları bildirir.

    veri_bilgisi = {
        "path": str(HAZIR_KLASOR),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": siniflar
    }

    with open(
        HAZIR_KLASOR / "data.yaml",
        "w",
        encoding="utf-8"
    ) as dosya:

        yaml.safe_dump(
            veri_bilgisi,
            dosya,
            sort_keys=False,
            allow_unicode=True
        )

    print("\nVeri seti hazırlandı!")
    print(f"Eğitim: {len(egitim_verileri)}")
    print(f"Doğrulama: {len(dogrulama_verileri)}")
    print(f"Test: {len(test_verileri)}")


if __name__ == "__main__":
    veri_setini_hazirla()