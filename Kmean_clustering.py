from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Example: Customer data (Annual Income vs Spending Score)
df = pd.read_csv("Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create and fit model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Add cluster labels to data
df['Cluster'] = kmeans.labels_
print(df.head())


# Visualize
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=df['Cluster'])
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Customer Segmentation')
plt.show()

"""inertias = []
for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    inertias.append(model.inertia_)

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()"""

"""from sklearn.metrics import silhouette_score
score = silhouette_score(X, kmeans.labels_)
print("Silhouette Score:", score)"""

 
