# CLePS: Cross-site, Leakage-safe and Persistence-Skill evaluation of ML PV power forecasting

M.Tech AI&ML dissertation (SIT Pune), Niranjan Dhanyakumar Kolpe. Guide: Dr. Shivali Wagle; co-guide: Dr. Madhvi Saxena.

## What it does
CLePS evaluates machine-learning PV power forecasts with (1) a leakage-safe chronological split, (2) leave-one-plant-out
testing and (3) forecast skill against smart persistence. It measures how much four leakage paths inflate the skill (E1),
how much skill is lost on unseen plants (E2), and whether models still beat smart persistence (E3).

## Files
| File | Role |
|---|---|
| `convert.py` | raw datasets in `data/` -> standard files in `output/1_standard/` (one reader per dataset) |
| `cleps.py` | the framework: samples, sufficiency check, split, models, E1/E2 runs, report (E3 tables, figures) |
| `run_all.bat` | runs every step in order (Windows); progress in `output/run_all_status.txt` |
| `requirements.txt` | packages; `requirements-lock.txt` = exact versions of the last full run |

## Setup (Windows, Python 3.11)
```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Input data (not in git; download links in `docs/excel/Dataset Log.xlsx`)
```
data/nist/2017/MM/onemin-Ground-2017-MM-DD.csv     NIST ground array, 1 min (or data/nist/onemin-Ground-2017.zip)
data/dryad/LSK North_Inverter.csv                  HKUST, from Dryad Dataset.zip (Inverter level dataset)
data/zenodo/dt3_electrical_production_inverter_2.csv   La Reunion, inverter 2
data/pvdaq/9068_ac_power_data_20240101_20250430.csv    PVDAQ 9068
data/pvdaq/9068_acvolt_curr_data_20240101_20250430.csv
data/pvdaq/9068_system_metadata.json               source of the 1910 kW inverter capacity
data/pvdaq/2107_electrical_data_2024.csv           PVDAQ 2107 (5th plant), from
    https://oedi-data-lake.s3.amazonaws.com/pvdaq/2023-solar-data-prize/2107_OEDI/data/2107_electrical_data_2024.csv
data/pvdaq/2107_system_metadata.json               .../2107_OEDI/metadata/2107_system_metadata.json
```
`output/1_standard/raw_files.csv` lists every input file with its size and MD5, so a later run can check that the
inputs are identical.

## Run
```
python convert.py                                              # standard files + conversion report
python cleps.py run --exp e1 e2 --models ridge gbt --horizons 1h
python cleps.py report
```
or double-click `run_all.bat` for everything (ridge, GBT, MLP, LSTM; 15 min and 1 h; all plants).

**Stop and resume:** every finished run is appended to `results.csv` at once. Stopping (closing the window) loses at most
the run in progress; starting again skips finished runs. Each run has a `fingerprint` (input data of the plants used +
the code and settings that compute results); a run is redone automatically only when its fingerprint changes, e.g.
new raw data, a new plant (E2 uses all other plants), or a changed model setting. Report changes never trigger re-runs.
Options: `--plants`, `--paths` (E1 leakage paths), `--seeds` (default 5).
To add a plant: add its reader and entry in `convert.py`, then run `run_all.bat` (only the new and affected runs are done).

## Outputs (all in `output/`, not in git)
| Folder | Content |
|---|---|
| `1_standard/` | `<plant>.csv` (timestamp_utc, power_kw, current_a), `plants.csv`, `conversion_report.csv`, `raw_files.csv`, `log.txt` |
| `2_run/` | `sufficiency.csv`, `results.csv` (one row per run), `daily_errors.csv` (per run and test day: squared-error sums of model and persistence), `log.txt` |
| `3_report/` | `summary.csv`, `e1_skill_inflation.csv`, `e2_cross_site_loss.csv`, `e3_skill_vs_persistence.csv` (each with 95% confidence intervals and a `_significant` column), `figures/` (error bars = 95% CI), `log.txt` |

## Design choices (not from a source)
80% coverage per 5-min interval; stuck = power and current unchanged for 1 h; small negative power (> -2% of capacity)
set to 0; power above 105% of rated capacity removed; sun >= 5 deg; input gaps <= 3 steps filled; 60/20/20 split;
>= 6 months and >= 20 valid days per block (day valid if >= 70% of daytime samples valid); 3-value grids; 5 seeds.

## Confidence intervals
Skill, skill inflation (E1) and cross-site loss (E2) get a 95% percentile interval from a bootstrap over test days
(1000 resamples; whole days are resampled so errors within a day stay together; seeds are pooled per day). Two runs
with the same test days (leaks b-d, E2 vs within-site) are resampled together (paired); the random split (a) has
different test rows, so its days are resampled separately. `_significant` = the interval does not contain 0. Days are
treated as independent, which ignores weather spells lasting several days, so the intervals may be somewhat narrow.
