import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

# CONFIGURATION
TARGET = "fraud_bool"
RANDOM_SEED = 42

# LOAD DATA
print("FAIRFRAUD - CATBOOST BASELINE")
print("\nLoading dataset...")
df = pd.read_csv("data/dataset.csv")
print(f"Dataset shape: {df.shape}")

# temporal data split
train_df = df[df["month"] <= 4].copy()
val_df = df[df["month"] == 5].copy()
test_df = df[df["month"] >= 6].copy()
print(f"\nTrain shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")


print("\nFraud distribution:")

for name, data in [
    ("Train", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:

    fraud_rate = data[TARGET].mean() * 100

    print(
        f"{name}: "
        f"{data[TARGET].sum():,} fraud cases "
        f"({fraud_rate:.4f}%)"
    )

# SEPARATE FEATURES AND TARGET
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET]
X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET]
X_test = test_df.drop(columns=[TARGET])
y_test = test_df[TARGET]

# IDENTIFY CATEGORICAL FEATURES
categorical_cols = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()
print("\n" + "=" * 70)
print("CATEGORICAL FEATURES")
print("=" * 70)
print(categorical_cols)

# CATBOOST REQUIRES CATEGORICAL VALUES AS STRINGS
for col in categorical_cols:
    X_train[col] = X_train[col].astype(str)
    X_val[col] = X_val[col].astype(str)
    X_test[col] = X_test[col].astype(str)


# ============================================================
# CLASS IMBALANCE CALCULATION
# ============================================================

fraud_count = y_train.sum()

non_fraud_count = len(y_train) - fraud_count

imbalance_ratio = non_fraud_count / fraud_count


print("\n" + "=" * 70)
print("CLASS IMBALANCE")
print("=" * 70)

print(f"Non-Fraud Cases: {non_fraud_count:,}")
print(f"Fraud Cases: {fraud_count:,}")
print(f"Imbalance Ratio: {imbalance_ratio:.2f}:1")


# ============================================================
# CATBOOST BASELINE MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CATBOOST BASELINE")
print("=" * 70)


model = CatBoostClassifier(

    iterations=500,

    depth=8,

    learning_rate=0.05,

    loss_function="Logloss",

    eval_metric="AUC",

    class_weights=[
        1,
        imbalance_ratio
    ],

    random_seed=RANDOM_SEED,

    verbose=100,

    early_stopping_rounds=50
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(

    X_train,
    y_train,

    cat_features=categorical_cols,

    eval_set=(X_val, y_val),

    use_best_model=True
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print("\nGenerating validation predictions...")

y_val_prob = model.predict_proba(X_val)[:, 1]

y_val_pred = (
    y_val_prob >= 0.5
).astype(int)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("Generating test predictions...")

y_test_prob = model.predict_proba(X_test)[:, 1]

y_test_pred = (
    y_test_prob >= 0.5
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    y_true,
    y_pred,
    y_prob,
    dataset_name
):

    print("\n" + "=" * 70)

    print(f"{dataset_name.upper()} RESULTS")

    print("=" * 70)


    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )


    auc_pr = average_precision_score(
        y_true,
        y_prob
    )


    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )


    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )


    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )


    print(f"\nROC-AUC: {roc_auc:.6f}")

    print(f"AUC-PR: {auc_pr:.6f}")

    print(f"Recall: {recall:.6f}")

    print(f"Precision: {precision:.6f}")

    print(f"F1 Score: {f1:.6f}")


    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print(cm)


    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )
    )


    return {

        "Dataset": dataset_name,

        "ROC_AUC": roc_auc,

        "AUC_PR": auc_pr,

        "Recall": recall,

        "Precision": precision,

        "F1_Score": f1
    }


# ============================================================
# EVALUATE VALIDATION SET
# ============================================================

val_results = evaluate_model(

    y_val,

    y_val_pred,

    y_val_prob,

    "Validation"
)


# ============================================================
# EVALUATE TEST SET
# ============================================================

test_results = evaluate_model(

    y_test,

    y_test_pred,

    y_test_prob,

    "Test"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame([

    val_results,

    test_results

])


results_path = (
    RESULTS_DIR /
    "catboost_baseline_metrics.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print("\nResults saved to:")

print(results_path)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODELS_DIR /
    "catboost_baseline.cbm"
)


model.save_model(
    model_path
)


print("\nModel saved to:")

print(model_path)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "actual": y_test.values,

    "fraud_probability": y_test_prob,

    "prediction": y_test_pred,

    "customer_age": X_test["customer_age"].values,

    "month": X_test["month"].values

})


predictions_path = (
    RESULTS_DIR /
    "catboost_test_predictions.csv"
)


predictions_df.to_csv(
    predictions_path,
    index=False
)


print("\nTest predictions saved to:")

print(predictions_path)


print("\n" + "=" * 70)

print("CATBOOST BASELINE TRAINING COMPLETED")

print("=" * 70)