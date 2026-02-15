from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
import pandas as pd


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy score"""
    return accuracy_score(y_true, y_pred)


def calculate_auc(y_true, y_pred_proba):
    """Calculate AUC score using predicted probabilities"""
    return roc_auc_score(y_true, y_pred_proba)


def calculate_precision(y_true, y_pred, pos_label='M'):
    """Calculate precision score"""
    return precision_score(y_true, y_pred, pos_label=pos_label)


def calculate_recall(y_true, y_pred, pos_label='M'):
    """Calculate recall score"""
    return recall_score(y_true, y_pred, pos_label=pos_label)


def calculate_f1(y_true, y_pred, pos_label='M'):
    """Calculate F1 score"""
    return f1_score(y_true, y_pred, pos_label=pos_label)


def calculate_mcc(y_true, y_pred):
    """Calculate Matthews Correlation Coefficient"""
    return matthews_corrcoef(y_true, y_pred)


def calculate_all_metrics(y_true, y_pred, y_pred_proba, pos_label='M'):
    """
    Calculate all 6 evaluation metrics at once

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (for AUC calculation)
        pos_label: Positive class label (default 'M' for Malignant)

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'Accuracy': calculate_accuracy(y_true, y_pred),
        'AUC': calculate_auc(y_true, y_pred_proba),
        'Precision': calculate_precision(y_true, y_pred, pos_label),
        'Recall': calculate_recall(y_true, y_pred, pos_label),
        'F1': calculate_f1(y_true, y_pred, pos_label),
        'MCC': calculate_mcc(y_true, y_pred)
    }
    return metrics


def display_metrics(metrics, model_name="Model"):
    """
    Display metrics in a formatted way

    Args:
        metrics: Dictionary of metric values
        model_name: Name of the model
    """
    print("=" * 60)
    print(f"{model_name.upper()} - EVALUATION METRICS")
    print("=" * 60)
    print(f"1. Accuracy:   {metrics['Accuracy']:.4f}")
    print(f"2. AUC Score:  {metrics['AUC']:.4f}")
    print(f"3. Precision:  {metrics['Precision']:.4f}")
    print(f"4. Recall:     {metrics['Recall']:.4f}")
    print(f"5. F1 Score:   {metrics['F1']:.4f}")
    print(f"6. MCC Score:  {metrics['MCC']:.4f}")
    print("=" * 60)


def get_confusion_matrix(y_true, y_pred):
    """Get confusion matrix"""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    """Get classification report"""
    return classification_report(y_true, y_pred)


def metrics_to_dataframe(metrics_dict, model_name):
    """
    Convert metrics dictionary to DataFrame row

    Args:
        metrics_dict: Dictionary with metric values
        model_name: Name of the model

    Returns:
        DataFrame with one row of metrics
    """
    data = {
        'Model': model_name,
        'Accuracy': f"{metrics_dict['Accuracy']:.4f}",
        'AUC': f"{metrics_dict['AUC']:.4f}",
        'Precision': f"{metrics_dict['Precision']:.4f}",
        'Recall': f"{metrics_dict['Recall']:.4f}",
        'F1': f"{metrics_dict['F1']:.4f}",
        'MCC': f"{metrics_dict['MCC']:.4f}"
    }
    return pd.DataFrame([data])