"""Temporal training, validation-only calibration, and non-destructive inference."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from .data_generator import ROOT, ensure_dataset, inject_faults
from .features import FEATURES, extract_features, validate_data
from .rules import apply_rules

MODEL_PATH = ROOT / 'models/rf_quality_model.joblib'
VERSION = 1


def audit_data(df, artifact, threshold=None):
    raw = validate_data(df)
    features = extract_features(raw)
    probability = artifact['model'].predict_proba(features)[:, 1]
    rules = apply_rules(raw)
    out = raw.copy()
    for col in rules:
        out[col] = rules[col]
    out['rf_fault_prob'] = probability
    cutoff = artifact['threshold'] if threshold is None else float(threshold)
    if not 0 <= cutoff <= 1:
        raise ValueError('Threshold must be between zero and one.')
    out['ml_flag'] = probability >= cutoff
    out['final_alert'] = out.rule_flag | out.ml_flag
    scores = np.abs((features.to_numpy()-artifact['center']) / artifact['scale']) * np.sqrt(artifact['model'].feature_importances_)
    clues = []
    for i, flag in enumerate(out.final_alert):
        if not flag:
            clues.append('')
            continue
        top = np.argsort(scores[i])[-3:][::-1]
        text = '; '.join(f'{FEATURES[j].replace("_", " ")}: {features.iloc[i, j]:.2f}' for j in top)
        clues.append((out.rule_reasons.iloc[i] + '; ' if out.rule_flag.iloc[i] else '') + text)
    out['diagnostic_clues'] = clues
    return out


def train_and_save_model():
    baseline = ensure_dataset()
    # Extract features separately per split AND scenario: no adjacent-day leakage across splits.
    partitions = {}
    for name, lo, hi, seed in [('train', 2018, 2021, 42), ('calibration', 2022, 2022, 43), ('test', 2023, 2023, 44)]:
        part = baseline[baseline.date.dt.year.between(lo, hi)].reset_index(drop=True)
        partitions[name] = inject_faults(part, seed)
    train, calibration = partitions['train'], partitions['calibration']
    x = extract_features(train)
    model = RandomForestClassifier(n_estimators=240, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(x, train.is_fault)
    probabilities = model.predict_proba(extract_features(calibration))[:, 1]
    precision, recall, thresholds = precision_recall_curve(calibration.is_fault, probabilities)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], 1e-12)
    threshold = float(thresholds[np.argmax(f1)])
    normal = x.loc[train.is_fault == 0]
    artifact = {'version': VERSION, 'model': model, 'threshold': threshold, 'features': FEATURES,
                'center': normal.mean().to_numpy(), 'scale': normal.std().clip(lower=.25).to_numpy(),
                'split': {'train': '2018–2021', 'calibration': '2022', 'test': '2023'},
                'calibration_f1': float(f1.max()), 'test_data': partitions['test']}
    from .evaluator import evaluate
    artifact['evaluation'] = evaluate(audit_data(partitions['test'], artifact))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MODEL_PATH.with_suffix('.tmp')
    joblib.dump(artifact, temporary, compress=3)
    temporary.replace(MODEL_PATH)
    return artifact


def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    artifact = joblib.load(MODEL_PATH)
    if artifact.get('version') != VERSION or artifact.get('features') != FEATURES:
        return train_and_save_model()
    return artifact


if __name__ == '__main__':
    result = train_and_save_model()
    print('Calibrated threshold:', result['threshold'])
    print('2023 synthetic evaluation:', result['evaluation']['metrics'])
