"""
Kredi Başvurularının Risk Durumunun Sınıflandırılması
Decision Tree ve Random Forest ile Kredi Risk Sınıflandırması

Amaç:
  - Kredi başvurularına ait bilgileri kullanarak başvurunun onaylanıp
    onaylanmayacağını (Loan_Status) sınıflandırmak
  - Decision Tree Classifier ve Random Forest Classifier modellerini
    karşılaştırmak

Veri seti:
  - Kaynak: Kaggle - Loan Prediction Dataset
    (https://www.kaggle.com/code/yonatanrabinovich/loan-prediction-dataset-ml-project)
  - loan_prediction.csv
  - 614 satır, 13 sütun
  - 11 feature ve 1 hedef değişken
  - Hedef değişken: Loan_Status (Y = onaylandı, N = reddedildi)

Plan/program:
  1. Veri setinin yüklenmesi ve incelenmesi
  2. Eksik değerlerin doldurulması (kategorik: mod, sayısal: ortalama)
  3. Aykırı değerlerin IQR yöntemi ile analiz edilmesi
  4. Kategorik değişkenlerin encode edilmesi
     (label encoding + one-hot encoding)
  5. Feature ve target değişkenlerin tanımlanması
  6. Eğitim ve test veri setlerinin oluşturulması
  7. Karar ağacı ve random forest modellerinin oluşturulması
  8. Test verisi ile tahmin yapılması
  9. Model başarımının accuracy ile ölçülmesi
  10. Karar ağacı sonuçlarının confusion matrix ile görselleştirilmesi
  11. Karar ağacının görselleştirilmesi
  12. Random forest feature importance incelenmesi

Kurulumlar:
pip install scikit-learn pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# =========================================================
# 1. VERİ SETİNİN YÜKLENMESİ VE İNCELENMESİ
# =========================================================
df = pd.read_csv("loan_prediction.csv")

print(df.head(10))
print(df.isnull().sum())


# =========================================================
# 2. EKSİK DEĞERLERİN DOLDURULMASI
# =========================================================
df = df.drop(columns=["Loan_ID"])


# =========================================================
# 2a. EKSİK DEĞERLERİN DOLDURULMASI
# =========================================================
# kategorik sütunlar: mod (en sık görülen değer) ile dolduruluyor
kategorik_sutunlar = [
    "Gender",
    "Married",
    "Dependents",
    "Self_Employed",
    "Credit_History"
]

for sutun in kategorik_sutunlar:
    df[sutun] = df[sutun].fillna(df[sutun].mode()[0])


# sayısal sütunlar: ortalama (mean) ile dolduruluyor
sayisal_sutunlar = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term"
]

for sutun in sayisal_sutunlar:
    df[sutun] = df[sutun].fillna(df[sutun].mean())

print(df.isnull().sum())  # kontrol: artık hiç eksik değer kalmamalı


# =========================================================
# 3. AYKIRI DEĞER ANALİZİ
# =========================================================
# IQR yöntemi ile aykırı değerler analiz ediliyor
print("\n--- Aykırı Değer Analizi ---")

for sutun in sayisal_sutunlar:

    Q1 = df[sutun].quantile(0.25)
    Q3 = df[sutun].quantile(0.75)

    IQR = Q3 - Q1

    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR

    aykiri_degerler = df[
        (df[sutun] < alt_sinir) |
        (df[sutun] > ust_sinir)
    ]

    print(f"\n{sutun}")
    print(f"Alt sınır: {alt_sinir:.2f}")
    print(f"Üst sınır: {ust_sinir:.2f}")
    print(f"Aykırı değer sayısı: {len(aykiri_degerler)}")


# aykırı değerleri boxplot ile görselleştirme
plt.figure(figsize=(12, 6))

sns.boxplot(data=df[sayisal_sutunlar])

plt.title("Sayısal Değişkenlerde Aykırı Değer Analizi")
plt.xticks(rotation=45)

plt.show()


# =========================================================
# 4. LABEL ENCODING VE ONE-HOT ENCODING
# =========================================================
df["Dependents"] = df["Dependents"].replace("3+", "3")

# Hedef değişken için ayrı encoder
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df["Loan_Status"])

# Sınıf isimlerini hemen kaydedin ve string'e dönüştürün
class_names = [str(c) for c in target_encoder.classes_]

print(f"\nHedef değişken sınıfları: \n{class_names}")

# Hedef sütunu veri setinden çıkart
X = df.drop(columns=["Loan_Status"])

# Özellikler için ayrı encoder kullanın veya dizeye çevirin
feature_encoder = LabelEncoder()
X["Gender"] = feature_encoder.fit_transform(X["Gender"])
X["Married"] = feature_encoder.fit_transform(X["Married"])
X["Dependents"] = feature_encoder.fit_transform(X["Dependents"])
X["Education"] = feature_encoder.fit_transform(X["Education"])
X["Self_Employed"] = feature_encoder.fit_transform(X["Self_Employed"])
X["Credit_History"] = feature_encoder.fit_transform(X["Credit_History"])
X["Property_Area"] = feature_encoder.fit_transform(X["Property_Area"])

# =========================================================
# 5. FEATURE VE TARGET DEĞİŞKENLERİN TANIMLANMASI
# =========================================================
feature_names = list(X.columns)

# =========================================================
# 6. EĞİTİM VE TEST VERİ SETLERİNİN OLUŞTURULMASI
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================================
# 7. KARAR AĞACI VE RANDOM FOREST MODELLERİNİN OLUŞTURULMASI
# =========================================================
tree_clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

random_forest_clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

tree_clf.fit(X_train, y_train)
random_forest_clf.fit(X_train, y_train)


# =========================================================
# 8. TEST VERİSİ İLE TAHMİN YAPILMASI
# =========================================================
tree_y_pred = tree_clf.predict(X_test)
random_forest_y_pred = random_forest_clf.predict(X_test)


# =========================================================
# 9. MODEL BAŞARIMININ ACCURACY İLE ÖLÇÜLMESİ
# =========================================================
tree_accuracy = accuracy_score(
    y_test,
    tree_y_pred
)

random_forest_accuracy = accuracy_score(
    y_test,
    random_forest_y_pred
)

print(f"tree_accuracy: {tree_accuracy}")
print(f"random_forest_accuracy: {random_forest_accuracy}")


# =========================================================
# 10. KARAR AĞACI SONUÇLARININ CONFUSION MATRIX İLE GÖRSELLEŞTİRİLMESİ
# =========================================================
conf_matrix = confusion_matrix(
    y_test,
    tree_y_pred
)

plt.figure()

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Tahmin edilen sınıf")
plt.ylabel("Gerçek sınıf")
plt.title("Karar ağacı confusion matrix")
plt.show()


# =========================================================
# 11. KARAR AĞACININ GÖRSELLEŞTİRİLMESİ
# =========================================================
plt.figure(figsize=(20, 10))
plot_tree(
    tree_clf,
    filled=True,
    feature_names=feature_names,
    class_names=class_names
)
plt.show()


# =========================================================
# 12. RANDOM FOREST FEATURE IMPORTANCE İNCELENMESİ
# =========================================================
feature_importances = random_forest_clf.feature_importances_

# önem derecelerini büyükten küçüğe sırala
feature_importances_sorted = sorted(
    zip(feature_importances, feature_names),
    reverse=True
)

for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")