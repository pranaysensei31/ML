import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("ionosphere.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.columns)

df.drop_duplicates(inplace=True)

sns.pairplot(df.iloc[:,0:5])
plt.show()

df.iloc[:,2].plot.hist(bins=30, figsize=(8,4))
plt.show()

df.iloc[:,2].plot.density()
plt.show()

corr = df.corr()
print(corr)

plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=False)
plt.show()

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

cols = X.columns[:5]

plt.figure(figsize=(15,8))
for i, col in enumerate(cols):
    plt.subplot(2,3,i+1)
    plt.scatter(df[col], y)
    plt.xlabel(col)
    plt.ylabel("Class")
    plt.title(col)
plt.tight_layout()
plt.show()

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print(accuracy_score(y_train, train_pred))
print(accuracy_score(y_test, test_pred))

print(confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred))

cdf = pd.DataFrame(model.coef_[0], index=X.columns, columns=["Coefficients"])
print(cdf)

n = X_train.shape[0]
k = X_train.shape[1]
dfN = n - k

train_error = (train_pred - y_train)**2
sum_error = np.sum(train_error)

se = []
for i in range(k):
    r = (sum_error/dfN)
    r = r / np.sum((X_train.iloc[:,i] - X_train.iloc[:,i].mean())**2)
    se.append(np.sqrt(r))

cdf["Standard Error"] = se
cdf["t-statistic"] = cdf["Coefficients"] / cdf["Standard Error"]

print(cdf)
