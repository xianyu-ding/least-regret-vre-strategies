#!/usr/bin/env python3
"""Build a validated six-scenario matrix from supplied summary data.

Default: nine criteria (LOLE, Cost, seven LCIA metrics). Use --include-co2
only after restoring data/raw/environment/co2_df.csv (Scenario,Pv).
"""
import argparse
from pathlib import Path
from energy_data import ROOT, load_mcdm_data

def load_specific_format_data(repo_root=None, include_co2=False):
    return load_mcdm_data(repo_root, include_co2=include_co2)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    parser.add_argument('--include-co2', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    data = load_specific_format_data(args.repo_root, args.include_co2)
    output = args.output or args.repo_root/'results'/'energy'/('combined_data_10criteria.csv' if args.include_co2 else 'combined_data.csv')
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output, index=False)
    print(f'Saved {len(data)} scenarios × {len(data.columns)-1} criteria to {output}')
    print('Summary-data analysis only; upstream TIAM/Antares/LCA reproduction is separate.')
    return data

if __name__ == '__main__':
    main()  # Missing/invalid required data must produce a nonzero exit code.
