"""Reproducible synthetic weather and independent fault scenarios."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATIONS = ['NY_CENTRAL_PARK', 'NY_JFK', 'NY_LAGUARDIA', 'NY_WHITE_PLAINS']
FIELDS = ['tmax', 'tmin', 'rainfall']
FAULTS = ['spike', 'stuck_sensor', 'drift', 'missing', 'inconsistent', 'neighbor_disagreement']


def generate_baseline(seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2018-01-01', '2023-12-31')
    n = len(dates)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = .78 * noise[i-1] + rng.normal(0, 1.7)
    regional = 16 + 14 * np.sin(2 * np.pi * (dates.dayofyear.to_numpy() - 105) / 365.25) + noise
    for year in range(2018, 2024):
        for month, delta in [(7, 9), (1, -10)]:
            mask = (dates.year == year) & (dates.month == month) & (dates.day >= 12) & (dates.day <= 18)
            regional[mask] += delta
    rain = np.where(rng.random(n) < .30, rng.gamma(1.4, 7, n), 0)
    frames = []
    for station, offset in zip(STATIONS, [0, -.8, .3, -2]):
        high = regional + offset + rng.normal(0, .65, n)
        low = high - np.clip(rng.normal(8, 1.4, n), 3, 14)
        frames.append(pd.DataFrame({'date': dates, 'station_id': station, 'tmax': high.round(1),
                                    'tmin': low.round(1), 'rainfall': (rain * rng.uniform(.7, 1.3, n)).round(1)}))
    return pd.concat(frames, ignore_index=True).sort_values(['date', 'station_id']).reset_index(drop=True)


def ensure_dataset():
    path = ROOT / 'data/ny_stations_raw.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        generate_baseline().to_csv(path, index=False)
    return pd.read_csv(path, parse_dates=['date'])


def apply_fault(df, station, start, mode, magnitude=25., duration=5):
    """Return a copy with a single labeled event; never mutate the input."""
    out = df.copy(deep=True)
    if 'is_fault' not in out:
        out['is_fault'], out['fault_type'] = 0, 'none'
    ids = out.loc[(out.station_id == station) & (out.date >= pd.Timestamp(start))].sort_values('date').index[:duration]
    if not len(ids):
        raise ValueError('Fault window contains no records.')
    if mode == 'spike':
        ids = ids[:1]
        out.loc[ids, 'tmax'] += magnitude
    elif mode == 'stuck_sensor':
        out.loc[ids, ['tmax', 'tmin']] = out.loc[ids[0], ['tmax', 'tmin']].to_numpy()
    elif mode == 'drift':
        out.loc[ids, 'tmax'] += np.arange(1, len(ids)+1) * magnitude
    elif mode == 'missing':
        out.loc[ids, FIELDS[int(abs(magnitude)) % 3]] = np.nan
    elif mode == 'inconsistent':
        if magnitude >= 0:
            out.loc[ids, 'tmin'] = out.loc[ids, 'tmax'] + max(1, magnitude)
        else:
            out.loc[ids, 'rainfall'] = magnitude
    elif mode == 'neighbor_disagreement':
        out.loc[ids, ['tmax', 'tmin']] += magnitude
    else:
        raise ValueError(f'Unknown fault mode: {mode}')
    out.loc[ids, ['is_fault', 'fault_type']] = [1, mode]
    return out


def make_upload_test_dataset():
    """Return an upload-safe sample and separate row-level ground truth.

    A new seed keeps this sample distinct from model evaluation data. Events
    affect one station at a time and leave the shared July heatwave intact.
    """
    clean = generate_baseline(seed=2026)
    clean = clean[clean.date.dt.year == 2023].reset_index(drop=True)
    sample = clean.copy()
    events = [
        ('2023-02-10', 'spike', 30., 1),
        ('2023-03-10', 'stuck_sensor', 0., 7),
        ('2023-04-10', 'drift', .4, 15),
        ('2023-05-10', 'missing', 0., 2),
        ('2023-05-15', 'missing', 1., 2),
        ('2023-05-20', 'missing', 2., 2),
        ('2023-06-10', 'inconsistent', 5., 1),
        ('2023-06-20', 'inconsistent', -5., 1),
        ('2023-08-10', 'neighbor_disagreement', 15., 3),
    ]
    for start, mode, magnitude, duration in events:
        sample = apply_fault(sample, 'NY_CENTRAL_PARK', start, mode, magnitude, duration)
    answer = sample[['date', 'station_id', 'is_fault', 'fault_type']].copy()
    for field in FIELDS:
        answer['original_' + field] = clean[field]
        answer['injected_' + field] = sample[field]
    # Never export duplicated scenarios or training-only columns for upload.
    return sample[['date', 'station_id', *FIELDS]], answer


def inject_faults(df, seed=42):
    """Duplicate the baseline into clean and corrupted scenarios, isolated by scenario_id."""
    rng = np.random.default_rng(seed)
    clean = df.copy().reset_index(drop=True)
    clean['is_fault'], clean['fault_type'], clean['scenario_id'] = 0, 'none', 'clean'
    dirty = clean.copy()
    dirty['scenario_id'] = 'injected'
    for station in clean.station_id.unique():
        dates = clean.loc[clean.station_id == station, 'date'].sort_values().tolist()
        for number, pos in enumerate(range(15, len(dates)-16, 23)):
            mode = FAULTS[number % len(FAULTS)]
            duration = {'spike': 1, 'stuck_sensor': int(rng.integers(4, 9)), 'drift': int(rng.integers(10, 16)),
                        'missing': 1, 'inconsistent': 1, 'neighbor_disagreement': int(rng.integers(2, 4))}[mode]
            magnitude = {'spike': rng.choice([-1, 1])*rng.uniform(15, 45), 'stuck_sensor': 0,
                         'drift': .4, 'missing': int(rng.integers(0, 3)),
                         'inconsistent': rng.choice([-5, 5]), 'neighbor_disagreement': rng.uniform(12, 18)}[mode]
            # Stagger station events so contemporaneous peers usually remain clean.
            pos += int(rng.integers(0, 6))
            dirty = apply_fault(dirty, station, dates[pos], mode, magnitude, duration)
    return pd.concat([clean, dirty], ignore_index=True)
