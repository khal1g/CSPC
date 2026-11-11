import os
import pandas as pd
from ucimlrepo import fetch_ucirepo

concrete = fetch_ucirepo(id=165)
X = concrete.data.features.copy()
y = concrete.data.targets

print("Metadata:", concrete.metadata)
print("\nVariables:", concrete.variables)

X.columns = [
    "cement", "slag", "ash",
    "coarse_agg", "fine_agg",
    "water", "superplastic",
    "age"
]

y = y.iloc[:, 0].rename("strength")

os.makedirs("data", exist_ok=True)
df_to_save = X.assign(strength=y)
df_to_save.to_csv("data/concrete.csv", index=False)
print("Data is saved to data/concrete.csv")