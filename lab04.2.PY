import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("austinreese/usa-housing-listings")
print(path)

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "austinreese/usa-housing-listings",
    "realtor-data.csv"
)

print(df.head())
print(df.shape)
print(df.columns)

df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

plt.figure()
plt.hist(df["price"], bins=50)
plt.xlabel("Price")
plt.ylabel("Count")
plt.title("House Price Distribution")
plt.show()
