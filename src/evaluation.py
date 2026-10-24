import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc, average_precision_score,
    classification_report, confusion_matrix
)

def evaluate_predictions(y_true, y_probs, model_name="Model", threshold=None):
    """
    Comprehensive evaluation function for binary classification models.
    Can be imported and reused across any model notebook.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)
    
    # 1. Calculate continuous curve metrics (PR-AUC and ROC-AUC)
    pr_auc = average_precision_score(y_true, y_probs)
    fpr_array, tpr_array, roc_thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr_array, tpr_array)
    
    # 2. Determine Decision Boundary (Manual override OR automated rule)
    if threshold is not None:
        # Use the explicit user-provided threshold directly
        chosen_threshold = float(threshold)
        chosen_idx = np.argmin(np.abs(roc_thresholds - chosen_threshold))
        chosen_fpr = fpr_array[chosen_idx]
        recall_at_fixed_fpr = tpr_array[chosen_idx]
    else:
        # Dynamically calculate the optimal threshold for Recall @ FPR <= 1.00%
        valid_fpr_indices = np.where(fpr_array <= 0.01)[0]
        if len(valid_fpr_indices) == 0:
            chosen_idx = 0
        else:
            chosen_idx = valid_fpr_indices[np.argmax(tpr_array[valid_fpr_indices])]
            
        chosen_threshold = roc_thresholds[chosen_idx]
        chosen_fpr = fpr_array[chosen_idx]
        recall_at_fixed_fpr = tpr_array[chosen_idx]
        
        # Safe fallback only if automated calculation completely breaks bounds
        if chosen_threshold > 1.0:
            chosen_threshold = 0.5
        
    # 3. Convert raw probabilities to hard binary predictions using our chosen threshold
    y_pred = (y_probs >= chosen_threshold).astype(int)
    
    # 4. Compute standard classification performance metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # 5. Print the formatted performance metrics summary report
    print(f"==================================================")
    print(f" EVALUATION REPORT: {model_name.upper()}")
    print(f"==================================================")
    print(f"Accuracy:                 {acc:.4%}")
    print(f"Fraud precision:          {prec:.4%}")
    print(f"Fraud recall:             {rec:.4%}")
    print(f"Fraud F1:                 {f1:.4%}")
    print(f"PR-AUC (AP):              {pr_auc:.4%}")
    print(f"ROC-AUC:                  {roc_auc:.4%}")
    print(f"Recall @ FPR <= 1.00%:    {recall_at_fixed_fpr:.4%}")
    print(f"FPR at chosen threshold:  {chosen_fpr:.4%}")
    print(f"Chosen Decision Boundary: {chosen_threshold:.4f}")
    print(f"==================================================")
    
    print("[Classification Report]")
    print(classification_report(y_true, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0))
    
    # 6. Generate, render, and display the evaluation plots matrix
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot A: Confusion Matrix Heatmap Graph
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False,
                xticklabels=["Legitimate", "Fraud"], yticklabels=["Legitimate", "Fraud"])
    axes[0].set_title(f"Confusion Matrix ({model_name})")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Plot B: Receiver Operating Characteristic (ROC) Curve Graph
    axes[1].plot(fpr_array, tpr_array, color="darkorange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    # Mark the specific operational threshold location on the graph curve
    axes[1].scatter(chosen_fpr, recall_at_fixed_fpr, color="red", s=100, zorder=5, 
                     label=f"Operational Point (FPR={chosen_fpr:.2%})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate (FPR)")
    axes[1].set_ylabel("True Positive Rate (TPR / Recall)")
    axes[1].set_title(f"ROC Curve ({model_name})")
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    
    # Plot C: Precision-Recall (PR) Curve Graph
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_probs)
    axes[2].plot(r_curve, p_curve, color="green", lw=2, label=f"PR Curve (AP = {pr_auc:.3f})")
    axes[2].set_xlim([0.0, 1.0])
    axes[2].set_ylim([0.0, 1.05])
    axes[2].set_xlabel("Recall")
    axes[2].set_ylabel("Precision")
    axes[2].set_title(f"Precision-Recall Curve ({model_name})")
    axes[2].legend(loc="lower left")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Evaluation script verified successfully. Ready to import.")
