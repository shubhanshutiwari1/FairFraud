import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/dataset.csv")

print("=" * 70)
print("FAIRFRAUD - CLEANED DATASET VALIDATION")
print("=" * 70)


# ============================================================
# 1. DATASET SHAPE
# ============================================================

print("\n1. DATASET SHAPE")
print("-" * 50)

print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")


# ============================================================
# 2. COLUMN NAMES
# ============================================================

print("\n2. COLUMN NAMES")
print("-" * 50)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n3. DATA TYPES")
print("-" * 50)

print(df.dtypes)


# ============================================================
# 4. DATASET INFO
# ============================================================

print("\n4. DATASET INFO")
print("-" * 50)

df.info()


# ============================================================
# 5. FIRST 5 ROWS
# ============================================================

print("\n5. FIRST 5 ROWS")
print("-" * 50)

print(df.head())


# ============================================================
# 6. STATISTICAL SUMMARY
# ============================================================

print("\n6. NUMERICAL FEATURE SUMMARY")
print("-" * 50)

print(df.describe().T)


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n7. MISSING VALUES")
print("-" * 50)

missing_values = df.isnull().sum()

print(missing_values[missing_values > 0])

total_missing = df.isnull().sum().sum()

print(f"\nTotal Missing Values: {total_missing:,}")

if total_missing == 0:
    print("✓ No missing values found")
else:
    print("⚠ Missing values detected")


# ============================================================
# 8. DUPLICATE ROWS
# ============================================================

print("\n8. DUPLICATE CHECK")
print("-" * 50)

duplicate_count = df.duplicated().sum()

print(f"Duplicate Rows: {duplicate_count:,}")

if duplicate_count == 0:
    print("✓ No duplicate rows found")
else:
    print("⚠ Duplicate rows detected")


# ============================================================
# 9. CONSTANT FEATURES
# ============================================================

print("\n9. CONSTANT FEATURE CHECK")
print("-" * 50)

constant_cols = [
    col for col in df.columns
    if df[col].nunique() <= 1
]

if len(constant_cols) == 0:
    print("✓ No constant features found")
else:
    print("⚠ Constant features found:")
    print(constant_cols)


# ============================================================
# 10. UNIQUE VALUES PER COLUMN
# ============================================================

print("\n10. UNIQUE VALUES PER FEATURE")
print("-" * 50)

unique_summary = pd.DataFrame({
    "Feature": df.columns,
    "Unique Values": [
        df[col].nunique()
        for col in df.columns
    ]
})

print(
    unique_summary
    .sort_values("Unique Values")
    .to_string(index=False)
)


# ============================================================
# 11. FEATURE TYPES
# ============================================================

print("\n11. FEATURE TYPE SUMMARY")
print("-" * 50)

numerical_cols = df.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

target_col = "fraud_bool"

if target_col in numerical_cols:
    numerical_features = [
        col for col in numerical_cols
        if col != target_col
    ]
else:
    numerical_features = numerical_cols


print(f"Numerical Features: {len(numerical_features)}")
print(numerical_features)

print(f"\nCategorical Features: {len(categorical_cols)}")
print(categorical_cols)


# ============================================================
# 12. TARGET VARIABLE CHECK
# ============================================================

print("\n12. TARGET VARIABLE CHECK")
print("-" * 50)

if target_col in df.columns:

    print("Target column found: fraud_bool")

    print("\nClass Counts:")
    print(df[target_col].value_counts())

    print("\nClass Percentages:")
    print(
        (
            df[target_col]
            .value_counts(normalize=True)
            * 100
        ).round(4)
    )

    legitimate = (df[target_col] == 0).sum()
    fraud = (df[target_col] == 1).sum()

    print(f"\nLegitimate cases: {legitimate:,}")
    print(f"Fraud cases: {fraud:,}")

    imbalance_ratio = legitimate / fraud

    print(
        f"Imbalance Ratio: "
        f"{imbalance_ratio:.2f}:1"
    )

else:
    print("⚠ Target column fraud_bool NOT found")


# ============================================================
# 13. CHECK CATEGORICAL FEATURES
# ============================================================

print("\n13. CATEGORICAL FEATURE VALIDATION")
print("-" * 50)

for col in categorical_cols:

    print(f"\nFeature: {col}")

    print(f"Unique values: {df[col].nunique()}")

    print("Categories:")

    print(
        df[col]
        .value_counts()
        .to_string()
    )


# ============================================================
# 14. NEGATIVE VALUES CHECK
# ============================================================

print("\n14. NEGATIVE VALUE CHECK")
print("-" * 50)

negative_summary = []

for col in numerical_features:

    negative_count = (df[col] < 0).sum()

    if negative_count > 0:

        negative_summary.append({
            "Feature": col,
            "Negative Count": negative_count,
            "Percentage": round(
                negative_count / len(df) * 100,
                4
            )
        })

if negative_summary:

    negative_df = pd.DataFrame(
        negative_summary
    )

    print(
        negative_df
        .sort_values(
            "Negative Count",
            ascending=False
        )
        .to_string(index=False)
    )

else:

    print("✓ No negative values found")


# ============================================================
# 15. INFINITE VALUES CHECK
# ============================================================

print("\n15. INFINITE VALUE CHECK")
print("-" * 50)

numeric_df = df[numerical_features]

infinite_count = np.isinf(
    numeric_df
).sum().sum()

print(
    f"Infinite values: {infinite_count:,}"
)

if infinite_count == 0:
    print("✓ No infinite values found")
else:
    print("⚠ Infinite values detected")


# ============================================================
# 16. OUTLIER SUMMARY
# ============================================================

print("\n16. OUTLIER SUMMARY (IQR METHOD)")
print("-" * 50)

outlier_summary = []

for col in numerical_features:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ).sum()

    outlier_summary.append({
        "Feature": col,
        "Outlier Count": outliers,
        "Outlier Percentage": round(
            outliers / len(df) * 100,
            4
        )
    })


outlier_df = pd.DataFrame(
    outlier_summary
)

print(
    outlier_df
    .sort_values(
        "Outlier Percentage",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# 17. TEMPORAL FEATURE CHECK
# ============================================================

print("\n17. TEMPORAL FEATURE CHECK")
print("-" * 50)

if "month" in df.columns:

    print("Month distribution:")

    month_summary = df.groupby("month").agg(
        transactions=("fraud_bool", "count"),
        fraud_cases=("fraud_bool", "sum"),
        fraud_rate=("fraud_bool", "mean")
    )

    month_summary["fraud_rate"] *= 100

    print(month_summary)

    print(
        f"\nMonth range: "
        f"{df['month'].min()} to "
        f"{df['month'].max()}"
    )

else:
    print("⚠ Month column not found")


# ============================================================
# 18. AGE FEATURE CHECK
# ============================================================

print("\n18. AGE FEATURE CHECK")
print("-" * 50)

if "customer_age" in df.columns:

    print(df["customer_age"].describe())

    print(
        f"\nAge range: "
        f"{df['customer_age'].min()} - "
        f"{df['customer_age'].max()}"
    )

else:
    print("⚠ customer_age column not found")


# ============================================================
# 19. DATASET MEMORY USAGE
# ============================================================

print("\n19. MEMORY USAGE")
print("-" * 50)

memory_mb = (
    df.memory_usage(deep=True).sum()
    / 1024**2
)

print(
    f"Dataset Memory Usage: "
    f"{memory_mb:.2f} MB"
)


# ============================================================
# 20. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET VALIDATION SUMMARY")
print("=" * 70)

checks = {
    "Target column exists":
        target_col in df.columns,

    "No missing values":
        df.isnull().sum().sum() == 0,

    "No duplicate rows":
        df.duplicated().sum() == 0,

    "No constant features":
        len(constant_cols) == 0,

    "No infinite values":
        infinite_count == 0,

    "Temporal feature exists":
        "month" in df.columns,

    "Protected feature exists":
        "customer_age" in df.columns
}

for check, result in checks.items():

    status = "✓ PASS" if result else "✗ FAIL"

    print(
        f"{status} | {check}"
    )


print("\nDataset validation completed!")
print("=" * 70)
from pathlib import Path


# Create validation summary
validation_summary = pd.DataFrame({
    "Check": [
        "Total Rows",
        "Total Columns",
        "Missing Values",
        "Duplicate Rows",
        "Constant Features",
        "Infinite Values",
        "Fraud Cases",
        "Non-Fraud Cases",
        "Fraud Rate (%)",
        "Imbalance Ratio"
    ],
    
    "Result": [
        len(df),
        df.shape[1],
        df.isnull().sum().sum(),
        df.duplicated().sum(),
        len(constant_cols),
        infinite_count,
        (df["fraud_bool"] == 1).sum(),
        (df["fraud_bool"] == 0).sum(),
        round(df["fraud_bool"].mean() * 100, 4),
        round(
            (df["fraud_bool"] == 0).sum() /
            (df["fraud_bool"] == 1).sum(),
            2
        )
    ]
})

# Save CSV
output_path = "src/data/results/dataset_validation_summary.csv"

validation_summary.to_csv(
    output_path,
    index=False
)

print(f"\n Validation summary saved to:")
print(output_path)