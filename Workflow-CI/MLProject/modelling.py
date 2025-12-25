import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Konfigurasi URI tracking berdasarkan environment
# Gunakan server lokal (port 5000) hanya jika TIDAK berjalan di CI (GitHub Actions)
if not os.environ.get("CI"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Mengatur nama eksperimen
mlflow.set_experiment("Bank_Marketing_Project")

# Memuat dataset hasil preprocessing
df = pd.read_csv("UCI Bank Marketing dataset_preprocessing.csv")

# Memisahkan fitur (features) dan target
X = df.drop(columns=["y"])
y = df["y"]

# Membagi data menjadi set latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Mengaktifkan autologging untuk menyimpan model, parameter, dan metrik secara otomatis
mlflow.sklearn.autolog()

# Menjalankan proses training
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluasi model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))
