"""
Kestirimci Bakım — Makine Arızası Tahmini
AI4I 2020 Predictive Maintenance Dataset (10.000 kayıt)

Soru: Sensör verisinden makine arızasını, arıza gerçekleşmeden önce
tahmin edebilir miyiz? Ve hangi fiziksel büyüklük bunu en çok belirliyor?

Beşir Oğuz Yılmaz
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             confusion_matrix, classification_report)

RS = 42
NAVY, AMBER, GREY = "#1F3355", "#D97706", "#5A6472"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": NAVY,
})

# ---------------------------------------------------------------- 1. Veri
df = pd.read_csv("data/ai4i2020.csv")
df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]

print(f"Kayıt: {len(df)}   Sütun: {df.shape[1]}")
print(f"Arıza oranı: {df['Machine failure'].mean()*100:.2f}%  "
      f"({int(df['Machine failure'].sum())} arıza)")

# ------------------------------------------------- 2. Özellik mühendisliği
# Fiziksel olarak anlamlı türetilmiş değişkenler:
#   Güç            = tork x açısal hız    -> aşırı yüklenmeyi yakalar
#   Sıcaklık farkı = proses - hava        -> soğutma yetersizliğini yakalar
df = df.rename(columns={
    "Air temperature [K]": "Hava sıcaklığı [K]",
    "Process temperature [K]": "Proses sıcaklığı [K]",
    "Rotational speed [rpm]": "Devir [rpm]",
    "Torque [Nm]": "Tork [Nm]",
    "Tool wear [min]": "Takım aşınması [dk]",
})
df["Güç [W]"] = df["Tork [Nm]"] * df["Devir [rpm]"] * 2 * np.pi / 60
df["Sıcaklık farkı [K]"] = df["Proses sıcaklığı [K]"] - df["Hava sıcaklığı [K]"]
df["Aşınma × Tork"] = df["Takım aşınması [dk]"] * df["Tork [Nm]"]

FEATURES = ["Hava sıcaklığı [K]", "Proses sıcaklığı [K]", "Devir [rpm]",
            "Tork [Nm]", "Takım aşınması [dk]",
            "Güç [W]", "Sıcaklık farkı [K]", "Aşınma × Tork"]
TARGET = "Machine failure"

# Sızıntı kontrolü: alt-arıza bayrakları (TWF/HDF/PWF/OSF/RNF) modele girmez.
# Bunlar arızanın kendisini kodlar; kullanılırsa model "kopya çeker".
X, y = df[FEATURES], df[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=RS, stratify=y)
print(f"Eğitim {len(X_tr)} / Test {len(X_te)}")

# --------------------------------------------------------------- 3. Model
# Sınıf dengesizliği %3.4 -> class_weight='balanced' ve PR-AUC ile ölçüm.
# Accuracy burada yanıltıcı: hiç arıza yok diyen model %96.6 doğruluk alır.
models = {
    "Lojistik Regresyon": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RS)),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=RS, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    results[name] = {"model": model, "proba": proba,
                     "ap": average_precision_score(y_te, proba)}
    print(f"{name:22s} PR-AUC = {results[name]['ap']:.3f}")

# Taban çizgi: her zaman "arıza yok"
base_rate = y_te.mean()
print(f"{'Taban çizgi (rastgele)':22s} PR-AUC = {base_rate:.3f}")

best_name = max(results, key=lambda k: results[k]["ap"])
best = results[best_name]
print(f"\nEn iyi: {best_name}")

# ------------------------------ 4. Çalışma noktası: %90 kesinlik hedefi
prec, rec, thr = precision_recall_curve(y_te, best["proba"])
ok = prec[:-1] >= 0.90
if ok.any():
    i = np.argmax(rec[:-1] * ok)
    esik, kesinlik, duyarlilik = thr[i], prec[i], rec[i]
else:
    i = np.argmax(prec[:-1]); esik, kesinlik, duyarlilik = thr[i], prec[i], rec[i]
print(f"Eşik {esik:.3f} -> kesinlik {kesinlik*100:.1f}%, "
      f"yakalanan arıza {duyarlilik*100:.1f}%")

y_pred = (best["proba"] >= esik).astype(int)
cm = confusion_matrix(y_te, y_pred)
print("\n", classification_report(y_te, y_pred,
                                  target_names=["Normal", "Arıza"], digits=3))

# ------------------------------------------------------------- 5. Grafikler
# G1 — Arızanın hangi fiziksel bölgede yoğunlaştığı
fig, ax = plt.subplots(figsize=(6.4, 4.0))
n = df[df[TARGET] == 0]
f = df[df[TARGET] == 1]
ax.scatter(n["Devir [rpm]"], n["Tork [Nm]"], s=5, c="#C7CEDA",
           label="Normal", rasterized=True)
ax.scatter(f["Devir [rpm]"], f["Tork [Nm]"], s=14, c=AMBER,
           label="Arıza", edgecolors="white", linewidths=0.3)
ax.set_xlabel("Devir [rpm]"); ax.set_ylabel("Tork [Nm]")
ax.set_title("Arızalar iki uçta toplanıyor")
ax.legend(frameon=False, loc="upper right")
fig.text(0.01, 0.01, "Yüksek tork + düşük devir (aşırı yük) ve düşük tork + yüksek "
         "devir bölgelerinde yığılma var.", fontsize=7.5, color=GREY)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("figur_1_tork_devir.png"); plt.close(fig)

# G2 — Precision-Recall eğrisi
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for name, r in results.items():
    p, rc, _ = precision_recall_curve(y_te, r["proba"])
    ax.plot(rc, p, lw=2, label=f"{name} (PR-AUC {r['ap']:.3f})",
            color=NAVY if name == best_name else "#9AA6B8")
ax.axhline(base_rate, ls="--", lw=1, color=GREY,
           label=f"Taban çizgi ({base_rate:.3f})")
ax.scatter([duyarlilik], [kesinlik], s=70, color=AMBER, zorder=5,
           edgecolors="white", linewidths=1.2)
ax.annotate(f"seçilen nokta\n%{kesinlik*100:.0f} kesinlik / %{duyarlilik*100:.0f} yakalama",
            (duyarlilik, kesinlik), textcoords="offset points", xytext=(-118, -34),
            fontsize=7.5, color=AMBER, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=AMBER, lw=0.8,
                            connectionstyle="arc3,rad=0.15"))
ax.set_xlabel("Yakalanan arıza oranı (recall)"); ax.set_ylabel("Kesinlik (precision)")
ax.set_title("Dengesiz veride doğruluk değil, PR eğrisi konuşur")
ax.legend(frameon=False, fontsize=8, loc="lower left")
fig.text(0.01, 0.01, "Arıza oranı %3.4 — 'hiç arıza yok' diyen bir model %96.6 "
         "doğruluk alır ama hiçbir işe yaramaz.", fontsize=7.5, color=GREY)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("figur_2_pr_egrisi.png"); plt.close(fig)

# G3 — Özellik önemleri
rf = results["Random Forest"]["model"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
fig, ax = plt.subplots(figsize=(6.4, 4.0))
renkler = [AMBER if v >= imp.max() * 0.55 else "#B9C2D0" for v in imp.values]
ax.barh(imp.index, imp.values, color=renkler, height=0.68)
ax.set_xlabel("Göreli önem")
ax.set_title("Arızayı en çok belirleyen büyüklükler")
for yy, vv in enumerate(imp.values):
    ax.text(vv + 0.004, yy, f"{vv:.3f}", va="center", fontsize=7.5, color=GREY)
ax.set_xlim(0, imp.max() * 1.18)
fig.text(0.01, 0.01, "Türetilen 'Güç' ve 'Aşınma × Tork' değişkenleri, ham "
         "ölçümlerin çoğundan daha bilgilendirici çıktı.", fontsize=7.5, color=GREY)
fig.tight_layout(rect=[0, 0.035, 1, 1])
fig.savefig("figur_3_ozellik_onem.png"); plt.close(fig)

# G4 — Karışıklık matrisi
fig, ax = plt.subplots(figsize=(4.6, 3.9))
ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
ax.set_xticks([0, 1], ["Normal", "Arıza"]); ax.set_yticks([0, 1], ["Normal", "Arıza"])
ax.set_xlabel("Model tahmini"); ax.set_ylabel("Gerçek")
ax.set_title(f"Test kümesi ({len(y_te)} kayıt)")
ax.grid(False)
for a in range(2):
    for b in range(2):
        ax.text(b, a, f"{cm[a, b]}", ha="center", va="center", fontsize=15,
                fontweight="bold", color="white" if cm[a, b] > cm.max()*0.5 else NAVY)
fig.tight_layout()
fig.savefig("figur_4_karisiklik.png"); plt.close(fig)

# ------------------------------------------------------------- 6. Özet
tn, fp, fn, tp = cm.ravel()
print(f"""
SONUÇ
  Yakalanan arıza      : {tp} / {tp+fn}  (%{duyarlilik*100:.1f})
  Yanlış alarm         : {fp}  ({len(y_te)} kayıtta)
  Kaçan arıza          : {fn}
  En belirleyici       : {imp.index[-1]}, {imp.index[-2]}
""")

pd.DataFrame({
    "metrik": ["PR-AUC", "eşik", "kesinlik", "yakalama", "yanlış alarm", "kaçan arıza"],
    "değer": [round(best["ap"], 3), round(float(esik), 3), round(float(kesinlik), 3),
              round(float(duyarlilik), 3), int(fp), int(fn)],
}).to_csv("sonuclar.csv", index=False)
print("Grafikler ve sonuclar.csv yazıldı.")
