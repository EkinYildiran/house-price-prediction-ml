from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veri seti
housing = fetch_california_housing(as_frame=True)
df = housing.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Train / Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Grafik
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Gerçek Ev Fiyatı")
plt.ylabel("Tahmin Edilen Ev Fiyatı")
plt.title("Linear Regression: Gerçek vs Tahmin")

# Referans çizgisi (ideal durum)
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()
