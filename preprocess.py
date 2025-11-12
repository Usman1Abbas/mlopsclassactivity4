import pandas as pd
from sklearn.datasets import load_iris

import os

os.makedirs("data", exist_ok=True)
# Proceed with saving the file (e.g., df.to_csv("data/yourfile.csv"))
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target

data.to_csv("data/preprocessed.csv", index=False)
print("✅ Data preprocessed and saved to data/preprocessed.csv")
