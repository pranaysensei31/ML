import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def ensure_dataset():
    if os.path.exists("USA_Housing.csv") and os.path.getsize("USA_Housing.csv") > 50:
        return

    import kagglehub

    print("✅ Downloading dataset...")
    path = kagglehub.dataset_download("kanths028/usa-housing")

    csv_file = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".csv"):
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break

    if not csv_file:
        raise FileNotFoundError("❌ CSV file not found inside downloaded dataset")

    shutil.copy(csv_file, "USA_Housing.csv")
    print("✅ Dataset saved as USA_Housing.csv")


ensure_dataset()

df = pd.read_csv("USA_Housing.csv", engine="python")

print("\n✅ Dataset Loaded Successfully\n")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", list(df.columns))


numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=30)
plt.title("Histogram of House Prices")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=30)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()


plt.figure(figsize=(10, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, linewidths=1)
plt.title("Correlation Heatmap")
plt.show()


l_column = list(df.columns)
len_feature = len(l_column)

X = df[l_column[0:len_feature - 2]]
y = df[l_column[len_feature - 2]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

lm = LinearRegression()
lm.fit(X_train, y_train)

pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)

print("\n✅ Linear Regression Model Trained\n")
print("Intercept:", lm.intercept_)
print("Coefficients:", lm.coef_)

coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": lm.coef_})
print("\nCoefficients Table:\n")
print(coef_df)

print("\n✅ Training R²:", round(metrics.r2_score(y_train, pred_train), 4))
print("✅ Testing R² :", round(metrics.r2_score(y_test, pred_test), 4))

print("\n✅ Evaluation Metrics (Test Set)")
print("MAE :", metrics.mean_absolute_error(y_test, pred_test))
print("MSE :", metrics.mean_squared_error(y_test, pred_test))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pred_test)))


plt.figure(figsize=(10, 7))
plt.title("Actual vs Predicted (Test Set)", fontsize=18)
plt.scatter(y_test, pred_test, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

residuals = y_test - pred_test

plt.figure(figsize=(10, 7))
plt.title("Histogram of Residuals", fontsize=18)
plt.hist(residuals, bins=30)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(10, 7))
plt.title("Residuals vs Predicted", fontsize=18)
plt.scatter(pred_test, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.show()


min_val = np.min(pred_test / 6000)
max_val = np.max(pred_test / 6000)

print("\nmin_val:", min_val)
print("max_val:", max_val)

L = (100 - min_val) / (max_val - min_val)
print("Min-Max scaled value for Price=100:", L)
