"""Held-out hybrid alert metrics, explicitly based on synthetic labels."""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from .data_generator import FAULTS


def evaluate(audit):
    y, pred = audit.is_fault, audit.final_alert
    metrics = {'Precision': precision_score(y, pred, zero_division=0),
               'Recall': recall_score(y, pred, zero_division=0), 'F1-Score': f1_score(y, pred, zero_division=0),
               'ROC-AUC': roc_auc_score(y, audit.rf_fault_prob) if y.nunique() == 2 else None}
    rows = []
    for mode in FAULTS:
        subset = audit[audit.fault_type == mode]
        caught = int(subset.final_alert.sum())
        rows.append({'Fault type': mode, 'Injected': len(subset), 'Caught': caught, 'Missed': len(subset)-caught,
                     'Recall': caught/len(subset) if len(subset) else 0.})
    return {'metrics': metrics, 'confusion': confusion_matrix(y, pred, labels=[0, 1]),
            'faults': pd.DataFrame(rows), 'rows': len(audit),
            'clean_false_positive_rate': float(audit.loc[y == 0, 'final_alert'].mean())}
