# Kestirimci Bakım — Makine Arızası Tahmini

Sensör verisinden makine arızasını, arıza gerçekleşmeden önce tahmin etme çalışması.
AI4I 2020 Predictive Maintenance veri seti üzerinde uçtan uca bir analiz: veri
incelemesi, fiziksel bilgiye dayalı özellik türetme, sınıf dengesizliği yönetimi ve
çalışma noktası seçimi.

## Soru

Bir freze makinesinin hava/proses sıcaklığı, devir, tork ve takım aşınması ölçümleri
elimizde. **Bu ölçümlerden arızayı önceden görebilir miyiz, ve hangi fiziksel büyüklük
bunu en çok belirliyor?**

## Veri

| | |
|---|---|
| Kaynak | AI4I 2020 Predictive Maintenance Dataset (UCI) |
| Kayıt | 10.000 |
| Özellik | 5 ham ölçüm + 3 türetilmiş |
| Hedef | `Machine failure` (ikili) |
| Arıza oranı | **%3.39** — ciddi sınıf dengesizliği |

## Yaklaşım

**Fiziksel bilgiye dayalı özellik türetme.** Ham ölçümlere üç değişken eklendi:

| Değişken | Formül | Neyi yakalar |
|---|---|---|
| Güç | `tork × 2π × devir / 60` | Aşırı yüklenme |
| Sıcaklık farkı | `proses − hava` | Soğutma yetersizliği |
| Aşınma × Tork | `takım aşınması × tork` | Yıpranmış takımla zorlama |

**Sızıntı kontrolü.** Veri setindeki alt-arıza bayrakları (TWF, HDF, PWF, OSF, RNF)
modelden çıkarıldı. Bunlar arızanın kendisini kodluyor; modele verilseydi sonuç
gerçek dışı iyi çıkardı.

**Dengesizlik yönetimi.** Arıza oranı %3.4 olduğu için doğruluk (accuracy) yanıltıcı:
"hiç arıza yok" diyen bir model %96.6 doğruluk alır ama hiçbir işe yaramaz. Bu yüzden
ölçüt olarak **PR-AUC** kullanıldı ve modeller `class_weight='balanced'` ile eğitildi.

## Sonuçlar

| Model | PR-AUC |
|---|---|
| Taban çizgi (rastgele) | 0.034 |
| Lojistik Regresyon | 0.417 |
| **Random Forest** | **0.835** |

Çalışma noktası, yanlış alarmı sınırlamak için **%90 kesinlik** hedefiyle seçildi:

- Arızaların **%63.5**'i yakalanıyor (54 / 85)
- 2500 kayıtlık test kümesinde yalnızca **6 yanlış alarm**
- Kaçan arıza: 31

Bakım planlaması açısından anlamı: her on uyarının dokuzu gerçek bir arıza. Bu oran,
uyarıların ciddiye alınabilmesi için gereken eşik.

### En belirleyici büyüklükler

Devir ve tork başı çekiyor; türetilen **Güç** değişkeni üçüncü sırada ve üç ham
sıcaklık ölçümünün hepsinden daha bilgilendirici. Arızalar tork–devir düzleminde iki
uçta toplanıyor: yüksek tork + düşük devir (aşırı yük) ve düşük tork + yüksek devir.

## Grafikler

| | |
|---|---|
| `figur_1_tork_devir.png` | Arızaların tork–devir düzlemindeki dağılımı |
| `figur_2_pr_egrisi.png` | Precision-Recall eğrisi ve seçilen çalışma noktası |
| `figur_3_ozellik_onem.png` | Özellik önemleri |
| `figur_4_karisiklik.png` | Karışıklık matrisi |

## Çalıştırma

```bash
pip install -r requirements.txt
python analiz.py
```

Veri seti `data/ai4i2020.csv` yolunda olmalıdır.

## Neden bu konu

Endüstriyel otomasyon tarafında step motor ve fırçasız DC sürücüleri Modbus üzerinden
sürdüğüm çalışmalarda, sürücülerden gerçek hız, tork ve alarm bilgisi zaten okunuyor.
Bu analiz, o veriyi yalnızca izlemek yerine arıza öngörüsü için kullanmanın ne kadar
mümkün olduğunu görme denemesi.

---

Beşir Oğuz Yılmaz · Mekatronik Mühendisliği, Yıldız Teknik Üniversitesi
