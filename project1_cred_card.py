import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

df=pd.read_csv('creditcard.csv')
#print(df.describe())
print(df.head())
print("tatal null columns are:",df.isnull().sum().sum())
sns.countplot(x='Class',data=df)
plt.title("Fraud (1) vs Normal (0) Transactions")
plt.show()
print(df['Class'].value_counts())

##################################separate features from target##############
x=df.drop(columns=["Class"])
y=df['Class']

##split train and test data by keeping 20% data fro testing
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
#Train a Model (Logistic Regression)

# Scale numeric features and train a classifier
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(class_weight="balanced", random_state=42))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

y_proba = model.predict_proba(X_test)[:,1]
PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision–Recall Curve (Logistic Regression)")
plt.show()
