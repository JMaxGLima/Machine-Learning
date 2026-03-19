import pandas as pd
from sklearn import datasets
import numpy as np  


data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
boston_df = pd.DataFrame(data, columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"])
print(boston_df.head())

boston_df = boston_df.drop(columns='DIS') # Eliminação Manual de Coluna "DIS"
print(boston_df.head())

df1 = pd.DataFrame(
    {
        "key":["K0", "K1", "K2", "K3"],
        "A":["A0", "A1", "A2", "A3"],
        "B":["B0", "B1", "B2", "B3"]
    }
)
print(df1)

df2 = pd.DataFrame(
    {
        "key":["K0", "K1", "K2", "K3"],
        "C":["C0", "C1", "C2", "C3"],   
        "D":["D0", "D1", "D2", "D3"]
    }
)
print(df2)

np.testing.assert_array_equal(df1.key, df2.key)
result = pd.merge(df1, df2, on="key")
print(result)

print(boston_df.sample(n=100))

print(boston_df.sample(n=100, replace=True))

print(boston_df.sample(frac=0.6))