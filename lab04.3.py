import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("ionosphere.csv")

print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:", df.shape)

# Data Cleaning
df.drop_duplicates(inplace=True)

# Select Numerical Columns
num_df = df.select_dtypes(include=[np.number])

# Descriptive Statistics
mean = num_df.mean()
median = num_df.median()
mode = num_df.mode().iloc[0]
std_dev = num_df.std()
minimum = num_df.min()
maximum = num_df.max()
total_sum = num_df.sum()

# Quartiles & Percentiles
q1 = num_df.quantile(0.25)
q2 = num_df.quantile(0.50)
q3 = num_df.quantile(0.75)
p90 = num_df.quantile(0.90)

# Correlation & Covariance
correlation = num_df.corr()
covariance = num_df.cov()

# Combine Stats
stats_df = pd.DataFrame({
    "Mean": mean,
    "Median": median,
    "Mode": mode,
    "Std Dev": std_dev,
    "Min": minimum,
    "Max": maximum,
    "Sum": total_sum,
    "Q1": q1,
    "Q2": q2,
    "Q3": q3,
    "90th Percentile": p90
})

print("\nStatistical Summary:\n", stats_df)
print("\nCorrelation Matrix:\n", correlation)
print("\nCovariance Matrix:\n", covariance)

# Better Features for Visualization
feature_a = df.iloc[:,2]
feature_b = df.iloc[:,3]

# Histogram
plt.figure()
plt.hist(feature_a, bins=30)
plt.title("Histogram of Feature 3")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# Boxplot
plt.figure()
plt.boxplot(feature_a)
plt.title("Boxplot of Feature 3")
plt.show()

# Scatter Plot
plt.figure()
plt.scatter(feature_a, feature_b)
plt.xlabel("Feature 3")
plt.ylabel("Feature 4")
plt.title("Feature 3 vs Feature 4")
plt.show()

# Line Plot
plt.figure()
plt.plot(feature_a)
plt.title("Line Plot of Feature 3")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()
