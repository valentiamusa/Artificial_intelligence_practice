"""import numpy as np
from sklearn.linear_model import LinearRegression

# Example data
X = np.array([[1000], [1500], [2000], [2500]])  # house size
y = np.array([120000, 180000, 250000, 310000])  # house price

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict a new price
new_house = np.array([[1800]])
prediction = model.predict(new_house)
print("Predicted price for 1800 sq ft:", prediction[0])
print("Slope:", model.coef_)        # m
print("Intercept:", model.intercept_)  # b


 
# Example: Predict study score from hours studied
X = np.array([[1], [2], [3], [4], [5]])  # hours
y = np.array([20, 40, 60, 80, 100])       # scores

model = LinearRegression()
model.fit(X, y)

# Predict score for a student who studied 6 hours
prediction = model.predict([[6]])
print("Predicted score for 6 hours:", prediction[0])
"""
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
data={
   'size':[1000, 1500, 2000, 2500, 3000],
   'price' :[200000, 250000, 300000, 350000, 400000]
}
df=pd.DataFrame(data)
print(df)

x=df[['size']]
y=df['price']

model=lm.LinearRegression()
model.fit(x,y)
print("slope(m):",model.coef_)
print("intercept(b):",model.intercept_)
predicted_price_1800=model.predict([[1800]])
print("predicted_price_1800:",predicted_price_1800)

# Step 6: Visualize
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', label='Regression Line')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.title('Linear Regression: House Size vs Price')
plt.show()
