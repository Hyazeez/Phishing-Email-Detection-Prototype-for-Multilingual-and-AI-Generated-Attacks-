from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

FILES = {
    "Enron": RAW_DIR / "Enron.csv",
    "SpamAssasin": RAW_DIR / "SpamAssasin.csv",
    "Nazario": RAW_DIR / "Nazario.csv",
    "Nigerian_Fraud": RAW_DIR / "Nigerian_Fraud.csv",
}

CATEGORY_TO_LABEL = {
    "Legitimate": 0,
    "Spam": 1,
    "Traditional_Phishing": 2,
    "Business_Email_Compromise": 3,
}


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def add_missing_metadata_columns(df):
    for column in ["sender", "receiver", "date", "urls"]:
        if column not in df.columns:
            df[column] = ""
    return df


def standardize_common_columns(df, source_dataset, category_series):
    df = add_missing_metadata_columns(df.copy())

    standardized = pd.DataFrame(index=df.index)
    standardized["source_dataset"] = source_dataset
    standardized["sender"] = df["sender"].apply(clean_text)
    standardized["receiver"] = df["receiver"].apply(clean_text)
    standardized["date"] = df["date"].apply(clean_text)
    standardized["subject"] = df["subject"].apply(clean_text)
    standardized["body"] = df["body"].apply(clean_text)

    standardized["email_text"] = (
        standardized["subject"] + " " + standardized["body"]
    ).str.strip()

    standardized["urls"] = df["urls"].fillna(0)
    standardized["original_label"] = df["label"]
    standardized["category"] = category_series
    standardized["label"] = standardized["category"].map(CATEGORY_TO_LABEL)
    standardized["language"] = "English"

    return standardized


def load_enron():
    df = pd.read_csv(FILES["Enron"], low_memory=False)
    category = df["label"].map({0: "Legitimate", 1: "Spam"})
    return standardize_common_columns(df, "Enron", category)


def load_spamassassin():
    df = pd.read_csv(FILES["SpamAssasin"], low_memory=False)
    category = df["label"].map({0: "Legitimate", 1: "Spam"})
    return standardize_common_columns(df, "SpamAssasin", category)


def load_nazario():
    df = pd.read_csv(FILES["Nazario"], low_memory=False)
    category = pd.Series(["Traditional_Phishing"] * len(df), index=df.index)
    return standardize_common_columns(df, "Nazario", category)


def load_nigerian_fraud():
    df = pd.read_csv(FILES["Nigerian_Fraud"], low_memory=False)
    category = pd.Series(["Business_Email_Compromise"] * len(df), index=df.index)
    return standardize_common_columns(df, "Nigerian_Fraud", category)


def remove_invalid_rows(df):
    df = df.copy()

    df = df[df["body"].str.strip() != ""]
    df = df[df["email_text"].str.strip() != ""]
    df = df[df["category"].notna()]
    df = df[df["label"].notna()]

    internal_nazario = (
        (df["source_dataset"] == "Nazario")
        & df["subject"].str.contains("FOLDER INTERNAL DATA", case=False, na=False)
    )
    df = df[~internal_nazario]

    df = df.drop_duplicates(subset=["email_text"])
    df = df.reset_index(drop=True)

    df.insert(0, "email_id", range(1, len(df) + 1))
    df["label"] = df["label"].astype(int)

    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [
        load_enron(),
        load_spamassassin(),
        load_nazario(),
        load_nigerian_fraud(),
    ]

    master = pd.concat(datasets, ignore_index=True)
    master = remove_invalid_rows(master)

    output_path = OUT_DIR / "master_dataset.csv"
    master.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Master dataset created successfully")
    print("Saved:", output_path)
    print("Rows:", len(master))

    print("\nCategory counts:")
    print(master["category"].value_counts().to_string())

    print("\nLabel mapping:")
    for category, label in CATEGORY_TO_LABEL.items():
        print(f"{label}: {category}")


if __name__ == "__main__":
    main()