# 1) Create a realistic synthetic HousePrices.csv
import numpy as np, pandas as pd

rng = np.random.default_rng(42)
n = 800

size_sqft = rng.normal(1800, 450, n).clip(400, 5000)
bedrooms = np.clip((size_sqft / 500 + rng.normal(0, 0.8, n)).round().astype(int), 1, 8)
age_years = np.clip(rng.normal(20, 12, n), 0, 100)
distance_city_km = np.abs(rng.normal(12, 7, n))
lot_size_sqft = (size_sqft * rng.normal(2.5, 0.6, n)).clip(500, 25000)
bathrooms = np.clip((bedrooms - 1 + rng.normal(0.2, 0.5, n)).round().astype(int), 1, 6)
is_renovated = rng.integers(0, 2, n)
has_garage = rng.integers(0, 2, n)
school_score = np.clip(rng.normal(7, 1.5, n), 1, 10)

base_price = 50000
price = (
    base_price
    + 180 * size_sqft
    + 12000 * bedrooms
    + 8000 * bathrooms
    - 900 * age_years
    - 2500 * distance_city_km
    + 2.5 * lot_size_sqft
    + 18000 * is_renovated
    + 10000 * has_garage
    + 15000 * (school_score - 5)
    + 0.03 * size_sqft * school_score
    - 0.6 * size_sqft * (age_years > 40)
    + rng.normal(0, 25000, n)
)

df = pd.DataFrame({
    "size_sqft": size_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "age_years": age_years,
    "distance_city_km": distance_city_km,
    "lot_size_sqft": lot_size_sqft,
    "is_renovated": is_renovated,
    "has_garage": has_garage,
    "school_score": school_score,
    "price": price
})

df.to_csv("HousePrices.csv", index=False)
print("Saved HousePrices.csv with", len(df), "rows")
print(df.head())

######################################
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("HousePrices.csv")

X = df.drop(columns=["price"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


##########################################
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

lin = LinearRegression()
rf = RandomForestRegressor(
    n_estimators=200,  # solid default; raise if your machine is fast
    random_state=42
)

lin.fit(X_train, y_train)
rf.fit(X_train, y_train)

############################
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def eval_reg(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2

pred_lin = lin.predict(X_test)
pred_rf  = rf.predict(X_test)

mse_lin, rmse_lin, r2_lin = eval_reg(y_test, pred_lin)
mse_rf,  rmse_rf,  r2_rf  = eval_reg(y_test, pred_rf)

print("Linear Regression  →  MSE:", round(mse_lin,2), "RMSE:", round(rmse_lin,2), "R²:", round(r2_lin,4))
print("Random Forest      →  MSE:", round(mse_rf,2),  "RMSE:", round(rmse_rf,2),  "R²:", round(r2_rf,4))

############################3
import matplotlib.pyplot as plt

# Linear Regression
plt.figure()
plt.scatter(y_test, pred_lin, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Linear Regression)")
plt.show()

# Random Forest
plt.figure()
plt.scatter(y_test, pred_rf, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()
