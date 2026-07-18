from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]

FEATURE_DIR = BASE_DIR / "data" / "features"
FEATURE_V2_DIR = BASE_DIR / "data" / "features_v2"
FEATURE_V2_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FEATURES = FEATURE_DIR / "master_features.csv"
CODE_MIXED_FEATURES = FEATURE_DIR / "code_mixed_features.csv"

RANDOM_STATE = 42

english_df = pd.read_csv(MASTER_FEATURES, low_memory=False)
code_mixed_df = pd.read_csv(CODE_MIXED_FEATURES, low_memory=False)

# Align columns safely
all_columns = sorted(set(english_df.columns).union(set(code_mixed_df.columns)))
english_df = english_df.reindex(columns=all_columns)
code_mixed_df = code_mixed_df.reindex(columns=all_columns)

master_v2 = pd.concat([english_df, code_mixed_df], ignore_index=True)

master_v2 = master_v2[master_v2["category"].notna()].copy()

master_v2.to_csv(
    FEATURE_V2_DIR / "master_features_v2.csv",
    index=False,
    encoding="utf-8-sig"
)

train_df, temp_df = train_test_split(
    master_v2,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=master_v2["category"]
)

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["category"]
)

train_df.to_csv(FEATURE_V2_DIR / "train_features_v2.csv", index=False, encoding="utf-8-sig")
validation_df.to_csv(FEATURE_V2_DIR / "validation_features_v2.csv", index=False, encoding="utf-8-sig")
test_df.to_csv(FEATURE_V2_DIR / "test_features_v2.csv", index=False, encoding="utf-8-sig")

print("V2 datasets created successfully")
print("Master V2:", master_v2.shape)
print("Train:", train_df.shape)
print("Validation:", validation_df.shape)
print("Test:", test_df.shape)

print("\nCategory distribution:")
print(master_v2["category"].value_counts())