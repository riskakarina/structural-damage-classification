import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TE_SMOOTH = 5.0
TE_KEYS = ["id6", "id8", "id9", "id10", "ward_id", "vdcmun_id"]
N_CLASS = 5

def _encode(keys_fit, Y_fit, keys_apply, prior):
    d = pd.DataFrame(Y_fit)
    d["k"] = keys_fit
    sums = d.groupby("k").sum()
    cnt = d.groupby("k").size()
    out = np.nan_to_num(sums.reindex(keys_apply).values)
    c = np.nan_to_num(cnt.reindex(keys_apply).values)
    return (out + TE_SMOOTH * prior) / (c[:, None] + TE_SMOOTH), c

def add_target_encoding_and_stats(df, y, n_train, seed=42):
    # Statistik kelompok dasar
    for g in ("id6", "id8", "id9", "id10", "ward_id", "vdcmun_id"):
        df[f"cnt_{g}"] = df.groupby(g)[g].transform("size")
    for g in ("id9", "id10", "ward_id"):
        df[f"{g}_age_mean"] = df.groupby(g)["age_clean"].transform("mean")
        df[f"{g}_floors_mean"] = df.groupby(g)["count_floors_pre_eq"].transform("mean")
        df[f"{g}_area_mean"] = df.groupby(g)["plinth_area_sq_ft"].transform("mean")
        
    # Target Encoding out-of-fold
    Y = np.eye(N_CLASS)[y]
    prior = Y.mean(0)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(n_train), y))
    
    for g in TE_KEYS:
        keys = df[g].values
        oof = np.zeros((n_train, N_CLASS))
        oof_n = np.zeros(n_train)
        for fit_idx, app_idx in folds:
            r, c = _encode(keys[fit_idx], Y[fit_idx], keys[app_idx], prior)
            oof[app_idx], oof_n[app_idx] = r, c
        r_te, c_te = _encode(keys[:n_train], Y, keys[n_train:], prior)
        
        full = np.vstack([oof, r_te])
        for j in range(N_CLASS):
            df[f"te_{g}_{j}"] = full[:, j]
        df[f"te_{g}_mean"] = full @ np.arange(N_CLASS)
        df[f"te_{g}_n"] = np.concatenate([oof_n, c_te])
        
    return df