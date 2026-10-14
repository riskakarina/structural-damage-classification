import numpy as np
import pandas as pd

def clean_and_build_features(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Target: "Grade 3" -> 2 (indeks 0..4)
    y = train["damage_grade"].str[-1].astype(int).values - 1
    n_train = len(train)
    
    df = pd.concat([train.drop(columns=["damage_grade"]), test], ignore_index=True)
    
    # Hierarki dari building_id
    s = df["building_id"].astype(str)
    for k in (6, 8, 9, 10):
        df[f"id{k}"] = s.str[:k].astype("int64")
        
    # Umur bangunan (999 dianggap missing/NaN)
    df["age_clean"] = df["age_building"].where(df["age_building"] < 900, np.nan)
    df["age_missing"] = (df["age_building"] >= 900).astype(int)
    
    # Fitur geometri bangunan
    df["area_per_floor"] = df["plinth_area_sq_ft"] / df["count_floors_pre_eq"]
    df["floor_height"] = df["height_ft_pre_eq"] / df["count_floors_pre_eq"]
    df["vol"] = df["plinth_area_sq_ft"] * df["height_ft_pre_eq"]
    df["hxa"] = df["height_ft_pre_eq"] / np.sqrt(df["plinth_area_sq_ft"])
    
    # Konversi kategorikal
    cat_cols = [
        "land_surface_condition", "foundation_type", "roof_type",
        "ground_floor_type", "other_floor_type", "position",
        "plan_configuration", "superstructure",
    ]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    for c in ("district_id", "vdcmun_id", "ward_id"):
        df[c + "_cat"] = df[c].astype("category")
        
    return df, y, n_train