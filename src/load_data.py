from sklearn.datasets import fetch_california_housing
import pandas as pd

# Veri setini yükle
housing = fetch_california_housing(as_frame=True)

# DataFrame'e çevir
df = housing.frame

print(df.head())
