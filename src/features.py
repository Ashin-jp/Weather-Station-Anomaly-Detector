"""17 deterministic features; each scenario and temporal partition is isolated."""
import numpy as np
import pandas as pd
from .data_generator import FIELDS

FEATURES = ['tmax_raw', 'tmin_raw', 'rainfall_raw', 'diurnal_range', 'tmax_diff_prev1',
            'tmax_diff_next1', 'tmin_diff_prev1', 'tmax_rolling_mean_7d', 'tmax_rolling_std_7d',
            'tmax_zscore_7d', 'tmax_seasonal_zscore', 'tmax_consecutive_identical',
            'tmin_consecutive_identical', 'tmax_drift_slope_10d', 'neighbor_tmax_diff',
            'neighbor_tmin_diff', 'missing_field_count']


def validate_data(df):
    required = ['date', 'station_id', *FIELDS]
    absent = set(required) - set(df.columns)
    if absent:
        raise ValueError('Missing columns: ' + ', '.join(sorted(absent)))
    if df.empty:
        raise ValueError('CSV has no records.')
    out = df.copy().reset_index(drop=True)
    out['date'] = pd.to_datetime(out.date, errors='coerce')
    if out.date.isna().any() or out.station_id.isna().any() or out.station_id.astype(str).str.strip().eq('').any():
        raise ValueError('Every row needs a valid date and nonempty station_id.')
    if (out.date != out.date.dt.normalize()).any():
        raise ValueError('Use daily dates without intraday timestamps.')
    keys = (['scenario_id'] if 'scenario_id' in out else []) + ['station_id', 'date']
    if out.duplicated(keys).any():
        raise ValueError('Duplicate station/date records are ambiguous; resolve them before auditing.')
    for field in FIELDS:
        values = pd.to_numeric(out[field], errors='coerce')
        if (out[field].notna() & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f'{field} must contain finite numbers or empty values.')
        out[field] = values
    return out


def _scenario_features(raw):
    f = pd.DataFrame(index=raw.index, columns=FEATURES, dtype=float)
    filled = raw[FIELDS].copy()
    for _, group in raw.groupby('station_id', sort=False):
        group = group.sort_values('date')
        for field, default in zip(FIELDS, [15., 7., 0.]):
            s = group[field]
            seasonal = s.groupby(group.date.dt.month).transform('median')
            filled.loc[group.index, field] = s.fillna(s.rolling(7, min_periods=1).median()).fillna(seasonal).fillna(default)
    for _, group in raw.groupby('station_id', sort=False):
        group = group.sort_values('date')
        idx = group.index
        hi, lo = filled.loc[idx, 'tmax'], filled.loc[idx, 'tmin']
        # Calendar windows and adjacency checks prevent gaps from masquerading as consecutive days.
        daily = pd.Series(hi.to_numpy(), index=pd.DatetimeIndex(group.date))
        mean = pd.Series(daily.rolling('7D', min_periods=1).mean().to_numpy(), index=idx)
        std = pd.Series(daily.rolling('7D', min_periods=2).std().fillna(0).to_numpy(), index=idx)
        adjacent = group.date.diff().dt.days.eq(1)
        next_adjacent = group.date.diff(-1).dt.days.eq(-1)
        f.loc[idx, ['tmax_raw', 'tmin_raw', 'rainfall_raw']] = filled.loc[idx].to_numpy()
        f.loc[idx, 'diurnal_range'] = hi-lo
        f.loc[idx, 'tmax_diff_prev1'] = hi.diff().where(adjacent, 0)
        f.loc[idx, 'tmax_diff_next1'] = (hi-hi.shift(-1)).where(next_adjacent, 0)
        f.loc[idx, 'tmin_diff_prev1'] = lo.diff().where(adjacent, 0)
        f.loc[idx, 'tmax_rolling_mean_7d'], f.loc[idx, 'tmax_rolling_std_7d'] = mean, std
        f.loc[idx, 'tmax_zscore_7d'] = (hi-mean)/std.clip(lower=.5)
        # Fixed NY climatological curve: never fit seasonality to held-out years.
        seasonal = 16 + 14*np.sin(2*np.pi*(group.date.dt.dayofyear-105)/365.25)
        f.loc[idx, 'tmax_seasonal_zscore'] = (hi-seasonal)/4
        for field in ['tmax', 'tmin']:
            original = group[field]
            starts = original.ne(original.shift()) | original.isna() | ~adjacent
            runs = original.groupby(starts.cumsum()).cumcount()+1
            f.loc[idx, field+'_consecutive_identical'] = runs.where(original.notna(), 0)
        for field in ['tmax', 'tmin']:
            peers = raw.loc[raw.station_id != group.station_id.iloc[0]].groupby('date')[field].median()
            neighbor = group.date.map(peers)
            fallback = hi-mean if field == 'tmax' else pd.Series(0., index=idx)
            f.loc[idx, 'neighbor_'+field+'_diff'] = (filled.loc[idx, field]-neighbor).fillna(fallback)
        residual = f.loc[idx, 'neighbor_tmax_diff']
        segments = (~adjacent).cumsum()
        slopes = residual.groupby(segments).transform(lambda s: s.rolling(10, min_periods=4).apply(
            lambda a: np.polyfit(np.arange(len(a)), a, 1)[0], raw=True))
        f.loc[idx, 'tmax_drift_slope_10d'] = slopes.fillna(0)
    f['missing_field_count'] = raw[FIELDS].isna().sum(axis=1)
    return f.fillna(0).astype(float)


def extract_features(df):
    raw = validate_data(df)
    if 'scenario_id' in raw:
        result = pd.concat([_scenario_features(g) for _, g in raw.groupby('scenario_id')]).sort_index()
    else:
        result = _scenario_features(raw)
    result.index = df.index
    return result[FEATURES]
