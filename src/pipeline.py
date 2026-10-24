# pipeline.py
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ============================================================
# CONFIGURATION & CORE SETTINGS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "dataset.csv"

TARGET = "fraud_bool"
TIME_COL = "month"

# REMOVED: 'device_fraud_count', 'days_since_request', 'bank_months_count', and 'month'
NUMERICAL_FEATURES = [
    "income", "name_email_similarity", "prev_address_months_count",
    "current_address_months_count", "customer_age", "intended_balcon_amount",
    "zip_count_4w", "velocity_6h", "velocity_24h", "velocity_4w",
    "bank_branch_count_8w", "date_of_birth_distinct_emails_4w", "credit_risk_score",
    "email_is_free", "phone_home_valid", "phone_mobile_valid",
    "has_other_cards", "proposed_credit_limit", "foreign_request",
    "session_length_in_minutes", "keep_alive_session", "device_distinct_emails_8w"
]

CATEGORICAL_FEATURES = [
    "payment_type", "employment_status", "housing_status", "source", "device_os"
]

# UPDATED: Added all columns found in EDA containing negative value placeholders
SENTINEL_MISSING_COLUMNS = [
    "prev_address_months_count", "current_address_months_count", 
    "intended_balcon_amount", "credit_risk_score", "velocity_6h", 
    "session_length_in_minutes", "device_distinct_emails_8w"
]

# ============================================================
# CORE PROCESSING FUNCTION
# ============================================================
def get_data():
    """
    Loads raw data, cleans negative anomalies, applies clean encodings,
    and returns perfectly isolated splits tailored for your 7 models.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset missing at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # Clean the raw targets and split fields
    df[TARGET] = df[TARGET].astype("int8")
    df[TIME_COL] = pd.to_numeric(df[TIME_COL], errors="coerce")

    # Fix hidden negative value placeholders by converting them to clean NaNs
    for col in SENTINEL_MISSING_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] < 0, col] = np.nan

    # Split data chronologically by month (Train: 0-4, Val: 5, Test: 6-7)
    train_df = df[df[TIME_COL] <= 4].copy()
    val_df = df[df[TIME_COL] == 5].copy()
    test_df = df[df[TIME_COL] >= 6].copy()

    # Extract target label arrays
    y_train = train_df[TARGET].to_numpy(dtype=np.int64)
    y_val = val_df[TARGET].to_numpy(dtype=np.int64)
    y_test = test_df[TARGET].to_numpy(dtype=np.int64)

    # --------------------------------------------------------
    # REPRESENTATION A: RAW DATA (For CatBoost)
    # --------------------------------------------------------
    X_train_raw = train_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    X_val_raw = val_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    X_test_raw = test_df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].copy()
    
    # Fill text categories so CatBoost reads missing strings properly
    for col in CATEGORICAL_FEATURES:
        for mat in [X_train_raw, X_val_raw, X_test_raw]:
            mat[col] = mat[col].fillna("__MISSING__").astype(str)

    # --------------------------------------------------------
    # REPRESENTATION B: MATH CLEANING (Imputation & Scaling)
    # --------------------------------------------------------
    num_imputer = SimpleImputer(strategy="median", add_indicator=True)
    scaler = StandardScaler()

    X_train_num = scaler.fit_transform(num_imputer.fit_transform(train_df[NUMERICAL_FEATURES])).astype(np.float32)
    X_val_num = scaler.transform(num_imputer.transform(val_df[NUMERICAL_FEATURES])).astype(np.float32)
    X_test_num = scaler.transform(num_imputer.transform(test_df[NUMERICAL_FEATURES])).astype(np.float32)

    cat_imputer = SimpleImputer(strategy="most_frequent")
    cat_train = cat_imputer.fit_transform(train_df[CATEGORICAL_FEATURES].astype(str))
    cat_val = cat_imputer.transform(val_df[CATEGORICAL_FEATURES].astype(str))
    cat_test = cat_imputer.transform(test_df[CATEGORICAL_FEATURES].astype(str))

    # --------------------------------------------------------
    # REPRESENTATION C: ONE-HOT ENCODING (For LightGBM, XGBoost, TabNet, Deep Learning)
    # --------------------------------------------------------
    # handle_unknown='ignore' ensures test values don't crash your models
    oh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    
    cat_train_oh = oh_encoder.fit_transform(cat_train)
    cat_val_oh = oh_encoder.transform(cat_val)
    cat_test_oh = oh_encoder.transform(cat_test)

    # Stack numerical and one-hot matrices horizontally to make the complete feature matrix
    X_train_processed = np.hstack([X_train_num, cat_train_oh])
    X_val_processed = np.hstack([X_val_num, cat_val_oh])
    X_test_processed = np.hstack([X_test_num, cat_test_oh])

    # Convert to sparse representations for high-speed LightGBM / XGBoost calculation
    X_train_sparse = csr_matrix(X_train_processed)
    X_val_sparse = csr_matrix(X_val_processed)
    X_test_sparse = csr_matrix(X_test_processed)

    # Calculate precise class imbalance scale weight (Legitimate vs Fraud ratio)
    class_counts = np.bincount(y_train)
    scale_pos_weight = float(class_counts[0] / class_counts[1])

    return {
        # Core Labels
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        
        # Raw Data (Use this for CatBoost)
        "X_train_raw": X_train_raw, "X_val_raw": X_val_raw, "X_test_raw": X_test_raw,
        
        # Dense Encoded Data (Use this for TabNet, FT-Transformer, SAINT, GraphSAGE)
        "X_train_dense": X_train_processed, "X_val_dense": X_val_processed, "X_test_dense": X_test_processed,
        
        # Sparse Matrix Data (Use this for LightGBM, XGBoost)
        "X_train_sparse": X_train_sparse, "X_val_sparse": X_val_sparse, "X_test_sparse": X_test_sparse,
        
        # Metadata / Parameters
        "scale_pos_weight": scale_pos_weight,
        "categorical_features": CATEGORICAL_FEATURES,
        "raw_customer_age_test": test_df["customer_age"].to_numpy(), # Saved for your fairness evaluation later

        "num_imputer": num_imputer,
        "scaler": scaler,
        "cat_imputer": cat_imputer,
        "onehot_encoder": oh_encoder
    }

if __name__ == "__main__":
    print("Testing pipeline initialization...")
    try:
        data = get_data()
        print(f"Success! Dense Data Shape: {data['X_train_dense'].shape}")
        print(f"Calculated Class Balance Scale Weight Factor: {data['scale_pos_weight']:.2f}")
    except FileNotFoundError:
        print("Pipeline test complete (Dataset file path not loaded locally yet).")
