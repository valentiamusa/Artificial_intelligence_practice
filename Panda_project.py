import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("cvs_dataset.csv")

"""print(data.isnull().sum())
data = data.dropna()  # or fill with 0 or mean
print(data.isnull().sum())
data['Sales_Date'] = pd.to_datetime(data['Sales_Date'])

print(data.head())
print(data.tail())"""

data['Sales_Date'] = pd.to_datetime(data['Sales_Date'])
"""data['Month'] = data['Sales_Date'].dt.to_period('M')
monthly_sales = data.groupby('Month')['Total_Sales'].sum()
print(monthly_sales)"""

"""monthly_sales.plot(kind='line', figsize=(5,5))
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()"""
product_sales = data.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False)
"""category_sales=data.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)
print(category_sales)

category_sales.plot(kind='bar', figsize=(8,5), color='skyblue')
plt.title('Sales by Product Category')
plt.ylabel('Total Sales')
plt.show()"""

top_products = product_sales.head(10)
top_products.plot(kind='bar', figsize=(10,5), color='orange')
plt.title('Top 10 Best-Selling Products')
plt.ylabel('Total Sales')
plt.show()

 
