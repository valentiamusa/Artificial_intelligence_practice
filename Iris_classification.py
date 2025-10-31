# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
# Step 2: Load the dataset
iris = load_iris()

# Convert to DataFrame for easier viewing
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
#print(df.head())
# Step 3: Split data into features (X) and target (y)
X = df[iris.feature_names]   # inputs
y = df['species']            # labels

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 4: Create and train the model
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
# Step 5: Predict on test data
y_pred = model.predict(X_test)
# Step 6: Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Accuracy:", accuracy)

# Step 8: Compare predictions vs actual results
results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

# Replace numeric labels (0, 1, 2) with actual species names
results['Actual'] = results['Actual'].apply(lambda x: iris.target_names[x])
results['Predicted'] = results['Predicted'].apply(lambda x: iris.target_names[x])

print(results.head(10))

"""plt.figure(figsize=(12,8))
tree.plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()"""
