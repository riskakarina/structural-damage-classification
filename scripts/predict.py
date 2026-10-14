import os
import joblib
import numpy as np
import pandas as pd

from preprocess import clean_and_build_features
from feature_engineering import add_target_encoding_and_stats

DATA_DIR = "data"
MODELS_DIR = "models"
SUB_DIR = "submissions"

def main():
    os.makedirs(SUB_DIR, exist_ok=True)
    
    # Load test IDs
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    test_ids = test_df["building_id"].values
    
    # Process Data
    df, y, n_train = clean_and_build_features(
        os.path.join(DATA_DIR, "train.csv"), 
        os.path.join(DATA_DIR, "test.csv")
    )
    df = add_target_encoding_and_stats(df, y, n_train)
    
    drop = ["building_id", "district_id", "vdcmun_id", "ward_id", "id6", "id8", "id9", "id10"]
    X = df.drop(columns=drop)
    X_test = X.iloc[n_train:]
    
    # Load Model
    saved_data = joblib.load(os.path.join(MODELS_DIR, "lgb_model_damage.pkl"))
    model = saved_data["model"]
    weights = saved_data["weights"]
    
    # Prediksi
    proba_te = model.predict(X_test)
    pred = (proba_te * weights).argmax(1) + 1 # Kembali ke rentang 1..5
    
    # Buat File Submission
    predictions = [f"Grade {k}" for k in pred]
    pred_by_id = pd.Series(predictions, index=test_ids)
    
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    submission = sample.copy()
    submission["damage_grade"] = submission["building_id"].map(pred_by_id)
    
    assert submission["damage_grade"].notna().all(), "Ada building_id tanpa prediksi!"
    
    sub_path = os.path.join(SUB_DIR, "submission(1).csv")
    submission.to_csv(sub_path, index=False)
    print(f"File submission berhasil dibuat di {sub_path}")

if __name__ == "__main__":
    main()