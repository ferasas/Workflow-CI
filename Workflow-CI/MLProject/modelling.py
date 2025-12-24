import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset hasil preprocessing
df = pd.read_csv("UCI Bank Marketing dataset_preprocessing.csv")

# Asumsi kolom target bernama 'y'
X = df.drop(columns=["y"])
y = df["y"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autolog
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))
