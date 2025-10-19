import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("kc_house_data.csv")
print("Preview data:")
print(data.head())

X = data[['sqft_living', 'bedrooms', 'bathrooms']]
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Akurasi model (R²): {score:.3f}")

joblib.dump(model, "house_price_model.pkl")
print("Model disimpan sebagai house_price_model.pkl")
