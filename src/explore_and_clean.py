from sklearn.datasets import fetch_california_housing
import pandas as pd

# Veri setini yükle
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("📌 İlk 5 satır:")
print(df.head())

print("\n📌 Sütun isimleri:")
print(df.columns)

print("\n📌 Eksik değer var mı?")
print(df.isnull().sum())

print("\n📌 Temel istatistikler:")
print(df.describe())


# Feature (X) ve hedef (y) ayır
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

print("\n📌 Feature matrix (X) shape:", X.shape)
print("📌 Target vector (y) shape:", y.shape)