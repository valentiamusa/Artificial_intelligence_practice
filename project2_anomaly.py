import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

rng = np.random.default_rng(42)   # random generator for reproducibility
n = 300                           # total number of records
attack_ratio = 0.1                # 10% of traffic will be attack
n_attack = int(n * attack_ratio)  # number of attack samples
# Normal connections
normal = {
    "duration": rng.normal(2, 1, n - n_attack).clip(0, None),
    "src_bytes": rng.normal(500, 150, n - n_attack).clip(0, None),
    "dst_bytes": rng.normal(600, 200, n - n_attack).clip(0, None),
    "count": rng.normal(10, 3, n - n_attack).clip(0, None),
    "srv_diff_host_rate": rng.uniform(0, 0.3, n - n_attack)
}

# Attack connections (abnormal)
attack = {
    "duration": rng.normal(15, 5, n_attack).clip(0, None),
    "src_bytes": rng.normal(2000, 800, n_attack).clip(0, None),
    "dst_bytes": rng.normal(50, 30, n_attack).clip(0, None),
    "count": rng.normal(50, 15, n_attack).clip(0, None),
    "srv_diff_host_rate": rng.uniform(0.5, 1.0, n_attack)
}

# Combine and label
df_normal = pd.DataFrame(normal)
df_attack = pd.DataFrame(attack)
df_normal["label"] = 0  # normal
df_attack["label"] = 1  # attack

df = pd.concat([df_normal, df_attack], ignore_index=True)
df.to_csv("network_intrusion.csv", index=False)
print(df.head())
print(df['label'].value_counts())
df.describe()


df = pd.read_csv("network_intrusion.csv")
X = df.drop(columns=["label"])
y = df["label"]

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
