import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os
import argparse

def train_model(n_estimators, random_state):
    # 1. Mengaktifkan Autologging
    # Ini akan secara otomatis mencatat parameter dan metrik
    mlflow.sklearn.autolog(log_models=True)

    # 2. Penentuan Path Dataset
    # Menyesuaikan dengan nama folder hasil preprocessing kriteria 1 di komputer Anda
    data_path = "Gaming_Hours_vs_Performance_1000_Rows_preprocessing"
    
    # Deteksi folder otomatis jika folder di atas tidak ditemukan (fallback)
    if not os.path.exists(data_path):
        potential_folders = [d for d in os.listdir('.') if os.path.isdir(d) and d.endswith('_preprocessing')]
        if potential_folders:
            data_path = potential_folders[0]
            print(f"ℹ️ Menggunakan folder data: {data_path}")

    try:
        # Memuat data train dan test hasil preprocessing
        train_df = pd.read_csv(os.path.join(data_path, "train_cleaned.csv"))
        test_df = pd.read_csv(os.path.join(data_path, "test_cleaned.csv"))
        
        # Pisahkan fitur dan target (kolom target bernama 'target')
        X_train = train_df.drop('target', axis=1)
        y_train = train_df['target']
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']

        # 3. Logika Eksekusi (Mencegah Konflik Run ID)
        # Jika MLFLOW_RUN_ID ada di environment, berarti dijalankan via 'mlflow run' (CI Mode)
        if "MLFLOW_RUN_ID" in os.environ:
            print(f"🚀 Menjalankan dalam konteks Workflow CI (ID: {os.environ['MLFLOW_RUN_ID']})")
            
            # Inisialisasi dan latih model
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Paksa simpan artefak model agar muncul di tab Artifacts dashboard
            mlflow.sklearn.log_model(model, "model")
            
            # Evaluasi sederhana
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"✅ Retraining Berhasil! Akurasi: {acc:.4f}")

        else:
            # Jika dijalankan manual: python modelling.py (Manual Mode)
            print("ℹ️ Menjalankan secara manual (Manual Mode)")
            # Pastikan MLflow UI menyala di terminal lain jika menggunakan URI ini
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment("Gaming_Performance_Workflow")
            
            with mlflow.start_run(run_name="Manual_Execution"):
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
                model.fit(X_train, y_train)
                
                # Simpan artefak model
                mlflow.sklearn.log_model(model, "model")
                
                acc = accuracy_score(y_test, model.predict(X_test))
                print(f"✅ Training Manual Berhasil! Akurasi: {acc:.4f}")

    except Exception as e:
        print(f"❌ Terjadi kesalahan selama proses pemodelan: {e}")

if __name__ == "__main__":
    # Menangkap parameter dari file MLProject atau Command Line
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    train_model(args.n_estimators, args.random_state)