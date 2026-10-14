import gc
import os
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score

from preprocess import clean_and_build_features
from feature_engineering import add_target_encoding_and_stats

DATA_DIR = "data"
MODELS_DIR = "models"
SEED = 42
N_CLASS = 5

LGB_PARAMS = dict(
    objective="multiclass", num_class=N_CLASS, learning_rate=0.12,
    num_leaves=63, min_data_in_leaf=40, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    max_bin=63, n_jobs=1, verbose=-1,
)

def search_class_weights(proba, y_true, rounds=3):
    w = np.ones(N_CLASS)
    best = f1_score(y_true, (proba * w).argmax(1), average="macro")
    for _ in range(rounds):
        for c in range(N_CLASS):
            for v in np.arange(0.5, 2.6, 0.05):
                w2 = w.copy()
                w2[c] = v
                s = f1_score(y_true, (proba * w2).argmax(1), average="macro")
                if s > best:
                    best, w = s, w2
    return w, best

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load & Process Data
    df, y, n_train = clean_and_build_features(
        os.path.join(DATA_DIR, "train.csv"), 
        os.path.join(DATA_DIR, "test.csv")
    )
    df = add_target_encoding_and_stats(df, y, n_train, seed=SEED)
    
    drop = ["building_id", "district_id", "vdcmun_id", "ward_id", "id6", "id8", "id9", "id10"]
    X = df.drop(columns=drop)
    del df
    
    # Optimasi Tipe Data Memori
    for c in X.columns:
        if X[c].dtype == np.float64:
            X[c] = X[c].astype(np.float32)
        elif X[c].dtype == np.int64:
            X[c] = X[c].astype(np.int32)
    gc.collect()
    
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    
    # 2. Holdout Validation
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_train)
    va, tr = perm[:100_000], perm[100_000:]
    
    model = lgb.train(
        LGB_PARAMS,
        lgb.Dataset(X_train.iloc[tr], y[tr]),
        num_boost_round=400,
        valid_sets=[lgb.Dataset(X_train.iloc[va], y[va])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    
    proba_va = model.predict(X_train.iloc[va])
    print(f"Holdout Macro F1 argmax: {f1_score(y[va], proba_va.argmax(1), average='macro'):.4f}")
    
    # 3. Search Class Weights
    weights, f1_w = search_class_weights(proba_va, y[va])
    print(f"Bobot kelas terpilih: {weights.round(2)}")
    
    # 4. Retrain pada seluruh data train
    n_rounds = int(model.best_iteration * 1.2)
    final_model = lgb.train(
        LGB_PARAMS | {"seed": 7},
        lgb.Dataset(X_train, y), 
        num_boost_round=n_rounds
    )
    
    # Simpan model dan bobot kelas
    joblib.dump({"model": final_model, "weights": weights}, os.path.join(MODELS_DIR, "lgb_model_damage.pkl"))
    print("Model berhasil disimpan ke models/lgb_model_damage.pkl")

if __name__ == "__main__":
    main()