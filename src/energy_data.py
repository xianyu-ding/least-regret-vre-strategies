"""Validated repository-local inputs shared by the energy analysis scripts.

Nine-criterion mode uses supplied summary data; ten-criterion mode additionally
requires CO2. Neither mode recreates upstream TIAM, Antares or LCA models.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ['NDC-LTP', 'VRE-70%', 'VRE-75%', 'VRE-80%', 'VRE-85%', 'VRE-90%']
ENVIRONMENTAL = ['Freshwater ecotoxicity', 'Freshwater eutrophication',
                 'Human toxicity', 'Ionising radiation', 'Land occupation',
                 'Marine eutrophication', 'Mineral resource depletion']

def numeric(series, label):
    try:
        result = pd.to_numeric(series.astype(str).str.replace(',', '', regex=False).str.strip(), errors='raise')
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label}: expected numeric values, including optional thousands separators') from exc
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError(f'{label}: missing or non-finite values')
    if (result < 0).any():
        raise ValueError(f'{label}: negative values require an explicitly reviewed accounting/normalization rule')
    return result

def _read(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required input not found: {path}. Restore the real source file; do not substitute zeros.')
    return pd.read_csv(path)

def load_mcdm_data(repo_root=None, include_co2=False):
    root = Path(repo_root) if repo_root else ROOT
    base = root / 'data' / 'raw'
    result = pd.DataFrame(index=pd.Index(SCENARIOS, name='Scenario'))
    sources = [('LOLE', base/'reliability'/'lole_df.csv'), ('Cost', base/'economics'/'cost_df.csv')]
    if include_co2:
        sources.append(('CO2_emission', base/'environment'/'co2_df.csv'))
    for name, path in sources:
        df = _read(path)
        if not {'Scenario', 'Pv'}.issubset(df.columns):
            raise ValueError(f'{path}: required columns Scenario,Pv')
        df['Scenario'] = df['Scenario'].astype(str).str.strip()
        if df['Scenario'].duplicated().any():
            raise ValueError(f'{path}: duplicate Scenario rows')
        missing = set(SCENARIOS) - set(df['Scenario'])
        if missing:
            raise ValueError(f'{path}: missing scenarios {sorted(missing)}')
        df = df.set_index('Scenario').loc[SCENARIOS]
        result[name] = numeric(df['Pv'], name)
    for name in ENVIRONMENTAL:
        path = base/'environment'/f'{name}.csv'
        df = _read(path)
        df.columns = df.columns.str.strip()
        if 'Unit' not in df or not set(SCENARIOS).issubset(df.columns):
            raise ValueError(f'{path}: requires Unit and all six scenario columns')
        rows = df[df.iloc[:, 0].astype(str).str.strip().str.lower().eq('total')]
        if len(rows) != 1:
            raise ValueError(f'{path}: expected exactly one Total row, got {len(rows)}')
        row = rows.iloc[0]
        if pd.isna(row['Unit']) or not str(row['Unit']).strip():
            raise ValueError(f'{path}: Total unit is missing')
        result[name] = numeric(row[SCENARIOS], name)
    return result.reset_index()

def validate_combined(df):
    df = df.rename(columns={'Scenario': 'scenario'}).copy()
    if 'scenario' not in df:
        raise ValueError('Combined data requires Scenario or scenario column')
    if df['scenario'].duplicated().any() or set(df['scenario']) != set(SCENARIOS):
        raise ValueError('Combined data must contain each of the six supported scenarios exactly once')
    indicators = [c for c in df if c != 'scenario']
    if not indicators:
        raise ValueError('Combined data has no indicator columns')
    for col in indicators:
        df[col] = numeric(df[col], col)
    return df.set_index('scenario').loc[SCENARIOS].reset_index()

def is_cost_indicator(name):
    key = name.removesuffix('_indicator').removesuffix('_impact').lower().replace('_', ' ')
    known = {'lole', 'cost', 'co2 emission'} | {x.lower() for x in ENVIRONMENTAL}
    if key not in known:
        raise ValueError(f'Unknown criterion direction: {name}. Add explicit reviewed metadata first.')
    return True

def vector_normalize(values, cost=True):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Vector normalization requires finite values')
    norm = np.linalg.norm(values)
    # A constant-zero criterion has no discriminatory information.
    normalized = values/norm if norm else np.zeros_like(values)
    return 1-normalized if cost else normalized
