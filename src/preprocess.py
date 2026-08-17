import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("../data/paysim.csv")

print("Original Dataset Shape:", df.shape)


# --------------------------------------------------
# 2. Keep only TRANSFER and CASH_OUT transactions
# --------------------------------------------------

df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

print("Filtered Dataset Shape:", df.shape)

print("\nTransaction Types After Filtering:")
print(df["type"].value_counts())


# --------------------------------------------------
# 3. Select useful columns
# --------------------------------------------------

df = df[
    [
        "step",
        "type",
        "amount",
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud"
    ]
].copy()


# --------------------------------------------------
# 4. Check missing values
# --------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 5. Check duplicate transactions
# --------------------------------------------------

print("\nDuplicate Rows:", df.duplicated().sum())


# --------------------------------------------------
# 6. Check fraud distribution
# --------------------------------------------------

print("\nFraud Distribution:")
print(df["isFraud"].value_counts())


# --------------------------------------------------
# 7. Save cleaned dataset
# --------------------------------------------------

df.to_csv("../data/cleaned_paysim.csv", index=False)

print("\nCleaned dataset saved successfully!")
print("File: data/cleaned_paysim.csv")