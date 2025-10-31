import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset
df = sns.load_dataset('tips')

# Inspect the first 5 rows
print(df.head())

"""sns.countplot(x='day', data=df)
plt.title("Number of Customers per Day")
plt.show()
"""
"""sns.countplot(x='sex', data=df)
plt.title("Number of Customers by Gender")
plt.show()"""

"""sns.countplot(x='smoker', data=df)
plt.title("Smoker vs Non-Smoker Customers")
plt.show()"""

"""sns.scatterplot(x='total_bill', y='tip', data=df, hue='sex')
plt.title("Total Bill vs Tip by Gender")
plt.show()"""

"""sns.boxplot(x='day', y='tip', data=df)
plt.title("Tip Distribution by Day")
plt.show()"""

"""sns.boxplot(x='time', y='total_bill', data=df)
plt.title("Total Bill Distribution by Time")
plt.show()
"""
# Correlation between numerical columns
numeric_df = df.select_dtypes(include='number')
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numerical Features")
plt.show()

