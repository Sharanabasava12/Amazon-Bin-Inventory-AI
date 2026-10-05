import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "dataset.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original samples: {len(df)}")

# Reference poster uses images with fewer than 6 items.
df = df[df["expected_quantity"] < 6].copy()

print(f"Samples after filtering (<6): {len(df)}")

print("\nClass distribution:")
print(df["expected_quantity"].value_counts().sort_index())

# Make sure the target is integer
df["label"] = df["expected_quantity"].astype(int)

# ---------------------------------------------------------
# 70% TRAIN / 20% VALIDATION / 10% TEST
# ---------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=1/3,
    random_state=42,
    stratify=temp_df["label"]
)

print("\nSplit sizes:")
print(f"Train      : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test       : {len(test_df)}")

# Save files
train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)

print("\nTrain distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())

print("\nCreated:")
print(OUTPUT_DIR / "train.csv")
print(OUTPUT_DIR / "val.csv")
print(OUTPUT_DIR / "test.csv")