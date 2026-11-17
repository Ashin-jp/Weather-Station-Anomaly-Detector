"""Physical guardrails operate exclusively on original measurements."""
import pandas as pd
from .data_generator import FIELDS


def apply_rules(df):
    checks = pd.DataFrame({
        'rule_missing': df[FIELDS].isna().any(axis=1),
        'rule_inconsistent_temp': df.tmin > df.tmax,
        'rule_negative_rain': df.rainfall < 0,
        'rule_physical_bounds': (df.tmax > 60) | (df.tmax < -60) | (df.tmin < -65) | (df.tmin > 60) | (df.rainfall > 500),
    }, index=df.index)
    labels = ['Missing telemetry', 'Minimum exceeds maximum', 'Negative rainfall', 'Outside physical bounds']
    checks['rule_flag'] = checks.any(axis=1)
    checks['rule_reasons'] = ['; '.join(label for label, flag in zip(labels, row) if flag)
                              for row in checks.iloc[:, :4].to_numpy()]
    return checks
