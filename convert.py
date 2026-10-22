"""CLePS step 1-2: raw PV datasets -> standardized dataset format (proposal Section 4.2, Table 3).

Input : data/<source>/<files>                        (download links: docs/excel/Dataset Log.xlsx)
Output: output/1_standard/<plant_id>.csv             timestamp_utc, power_kw, current_a  (5-min means, UTC)
        output/1_standard/plants.csv                 one row per plant: capacity, location, orientation, source
        output/1_standard/conversion_report.csv      what the converter did to each plant (for checking by eye)
        output/1_standard/log.txt                    console messages of the last run
        output/1_standard/raw_files.csv              every raw input file with size and MD5 (to check the inputs later)

To add a dataset: write one reader that returns UTC-indexed power_kw and current_a (phases summed),
then add one entry to PLANTS. Everything after that is the same for every plant.
"""
import glob
import hashlib
import math
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

RAW, OUT = Path("data"), Path("output/1_standard")
STEP = "5min"            # standard resolution
MIN_COVERAGE = 0.80      # a 5-min interval is kept only if >= 80% of its raw points exist (design choice)
STUCK_STEPS = 12         # power and current unchanged for >= 1 h while producing -> "stuck" logger (design choice)
NEG_TOLERANCE = 0.02     # small negative power (standby draw, > -2% of capacity) is set to 0; larger -> empty
OVER_TOLERANCE = 0.05    # power above 105% of rated AC capacity -> empty (5% margin for meter tolerance, design choice)


def log(msg):
    print(msg, flush=True)
    with open(OUT / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------- readers: one per dataset, each returns DataFrame(index=UTC time, power_kw, current_a) ----------
def phase_sum(df, cols):
    return df[cols].sum(axis=1, min_count=len(cols))          # empty if any phase is missing


def read_nist_ground():
    # NIST TN 1896: loggers use local standard time; the timestamps carry the -05:00 offset.
    zpath = RAW / "nist" / "onemin-Ground-2017.zip"               # either the zip or its extracted daily CSVs
    if zpath.exists():
        z = zipfile.ZipFile(zpath)
        files = [z.open(n) for n in sorted(z.namelist()) if n.endswith(".csv")]
    else:
        files = sorted(glob.glob(os.path.join(RAW, "nist", "**", "onemin-Ground-*.csv"), recursive=True))
    log(f"[convert] nist_ground: {len(files)} raw files found under {os.path.abspath(RAW / 'nist')}")
    if not files:
        raise FileNotFoundError(f"No NIST files (onemin-Ground-*.csv or onemin-Ground-2017.zip) in {os.path.abspath(RAW / 'nist')}")
    faults = ["InvMainFault_Max", "InvDriveFault_Max", "InvVoltageFault_Max",
              "InvGridFault_Max", "InvTempFault_Max", "InvSystemFault_Max"]
    cols = ["TIMESTAMP", "InvPAC_kW_Avg", "InvIa_Avg", "InvIb_Avg", "InvIc_Avg"] + faults
    d = pd.concat([pd.read_csv(f, usecols=cols) for f in files], ignore_index=True)
    bad = d[faults].fillna(0).ne(0).any(axis=1)                # source fault flags -> empty
    out = pd.DataFrame({"power_kw": d["InvPAC_kW_Avg"],
                        "current_a": phase_sum(d, ["InvIa_Avg", "InvIb_Avg", "InvIc_Avg"])})
    out[bad] = np.nan
    out.index = pd.to_datetime(d["TIMESTAMP"], utc=True)
    return out


def read_hkust_lsk_north():
    # Inverter-level file of station LSK North, taken from the Dryad Dataset.zip.
    # Naive timestamps = Hong Kong time (UTC+8, no DST).
    d = pd.read_csv(RAW / "dryad" / "LSK North_Inverter.csv",
                    usecols=["Time", "totalActivePower(W)", "L1_acCurrent(A)", "L2_acCurrent(A)", "L3_acCurrent(A)"])
    out = pd.DataFrame({"power_kw": d["totalActivePower(W)"] / 1000,
                        "current_a": phase_sum(d, ["L1_acCurrent(A)", "L2_acCurrent(A)", "L3_acCurrent(A)"])})
    out.index = pd.to_datetime(d["Time"]).dt.tz_localize("Etc/GMT-8").dt.tz_convert("UTC")
    return out


def read_la_reunion():
    # Data in Brief 46 (2023) 108723, dataset 3 (inverter 2, no faults). Timestamps carry +04:00.
    # Pg = power injected into the grid [W]; Ig = current injected into the grid [A] (single phase).
    d = pd.read_csv(RAW / "zenodo" / "dt3_electrical_production_inverter_2.csv", usecols=["time", "Pg", "Ig"])
    out = pd.DataFrame({"power_kw": d["Pg"] / 1000, "current_a": d["Ig"]})
    out.index = pd.to_datetime(d["time"], utc=True)
    return out


def read_pvdaq_9068():
    # PVDAQ 'measured_on' is local time and may include daylight saving (OEDI docs); Kersey, CO -> America/Denver.
    t = "measured_on"
    p = pd.read_csv(RAW / "pvdaq" / "9068_ac_power_data_20240101_20250430.csv", usecols=[t, "inverter_1_ac_power_(kw)_inv_150143"])
    phases = [f"inverter_1_ac_current_phase_{k}_(a)_inv_{150122 + k}" for k in (1, 2, 3)]
    c = pd.read_csv(RAW / "pvdaq" / "9068_acvolt_curr_data_20240101_20250430.csv", usecols=[t] + phases)
    d = p.merge(c, on=t, how="outer")
    out = pd.DataFrame({"power_kw": d["inverter_1_ac_power_(kw)_inv_150143"], "current_a": phase_sum(d, phases)})
    out.index = pd.to_datetime(d[t]).dt.tz_localize("America/Denver", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return out[out.index.notna()]


F_2107 = RAW / "pvdaq" / "2107_electrical_data_2024.csv"        # optional 5th plant; used only if present


def read_pvdaq_2107():
    # PVDAQ 2107 'Farm Solar Array', Arbuckle CA; 'measured_on' is local time, timezone_code PST8PDT (metadata)
    # -> America/Los_Angeles (same rules). Inverter 01 only; column names found by prefix (metric ids may differ).
    t = "measured_on"
    head = pd.read_csv(F_2107, nrows=0).columns
    pick = lambda pre: next(c for c in head if c.lower().startswith(pre))
    pc, cc = pick("inv_01_ac_power"), pick("inv_01_ac_current")
    d = pd.read_csv(F_2107, usecols=[t, pc, cc])
    out = pd.DataFrame({"power_kw": d[pc].to_numpy(), "current_a": d[cc].to_numpy()})
    out.index = pd.to_datetime(d[t]).dt.tz_localize("America/Los_Angeles", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return out[out.index.notna()]


# capacity_kw = AC rating of the device that measures power_kw. If it is not documented, it is left
# empty here and estimated as the highest 5-min mean power (marked "estimated" in plants.csv).
# tilt/azimuth only where documented (azimuth: degrees clockwise from north); otherwise CLePS uses horizontal.
PLANTS = [
    dict(plant_id="nist_ground", reader=read_nist_ground, capacity_kw=260.0, latitude=39.1319, longitude=-77.2141,
         tilt_deg=20, azimuth_deg=180,
         source="NIST TN 1896, doi:10.6028/NIST.TN.1896 (271 kW DC, 260 kW inverter, tilt 20, azimuth 180)"),
    dict(plant_id="hkust_lsk_north", reader=read_hkust_lsk_north, capacity_kw=55.0, latitude=22.3363, longitude=114.2634,
         tilt_deg=None, azimuth_deg=None,
         source="Lin et al., Dryad 2024, doi:10.5061/dryad.m37pvmd99 (CC0); README location (campus); metadata .ttl: "
                "LSK_North rated 55 kW (SolarEdge SE55K), tilt 10, azimuth '10/-153deg' (two faces -> horizontal used)"),
    dict(plant_id="la_reunion_inv2", reader=read_la_reunion, capacity_kw=None, latitude=-20.90126, longitude=55.48422,
         tilt_deg=None, azimuth_deg=None,
         source="Graillet et al., Data in Brief 2023, doi:10.1016/j.dib.2022.108723 (4 kW plant, 2 inverters; "
                "tilt 26, azimuth 11 deg from north, side not stated -> horizontal used)"),
    dict(plant_id="pvdaq_9068_inv1", reader=read_pvdaq_9068, capacity_kw=1910.0, latitude=40.3864, longitude=-104.5512,
         tilt_deg=None, azimuth_deg=None,
         source="PVDAQ (OEDI), doi:10.25984/1846021; system 9068 'SR_CO', 4738 kW DC, 2 inverters, single-axis tracker "
                "(not fixed -> horizontal used); 9068_system_metadata.json: inverter AC capacity 1910 kW"),
]
if F_2107.exists():
    PLANTS.append(dict(
        plant_id="pvdaq_2107_inv01", reader=read_pvdaq_2107, capacity_kw=30.0, latitude=38.996306, longitude=-122.134111,
        tilt_deg=25, azimuth_deg=180,
        source="PVDAQ (OEDI) 2023 Solar Data Prize, system 2107 'Farm Solar Array', Arbuckle CA, 893 kW DC, "
               "24 x ABB TRIO-27.6-TL-OUTD-S1B, fixed ground mount tilt 25 azimuth 180 (2107_system_metadata.json); "
               "inverter rated AC power 27.6 kW, maximum AC output power 30 kW (ABB TRIO-20.0/27.6 datasheet, -400 "
               "version; capacity = 30 kW because measured power reaches 30 kW)"))


# ---------- the same steps for every plant ----------
def standardize(raw, capacity_kw):
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated()]
    step_s = float(raw.index.to_series().diff().dt.total_seconds().median())
    if step_s > 300:
        raise ValueError(f"raw resolution {step_s:.0f} s is coarser than 5 min -> dataset rejected")
    need = math.ceil(MIN_COVERAGE * 300 / step_s)

    p, c = raw["power_kw"].copy(), raw["current_a"].copy()
    cap = capacity_kw if capacity_kw else p.max()
    over = (1 + OVER_TOLERANCE) * capacity_kw if capacity_kw else np.inf
    n_bad = int((p < -NEG_TOLERANCE * cap).sum() + (c < 0).sum() + (p > over).sum())
    p[(p < 0) & (p >= -NEG_TOLERANCE * cap)] = 0.0
    p[p < -NEG_TOLERANCE * cap] = np.nan
    p[p > over] = np.nan
    c[c < 0] = np.nan

    agg = pd.DataFrame({"power_kw": p, "current_a": c}).resample(STEP).agg(["mean", "count"])
    std = pd.DataFrame({k: agg[(k, "mean")].where(agg[(k, "count")] >= need) for k in ("power_kw", "current_a")})

    # stuck readings: power and current both unchanged for >= STUCK_STEPS steps while power > 0
    same = std["power_kw"].diff().eq(0) & std["current_a"].diff().eq(0) & std["power_kw"].gt(0)
    run_id = (~same).cumsum()
    run_len = same.groupby(run_id).transform("sum") + 1
    stuck = run_len.ge(STUCK_STEPS) & std["power_kw"].gt(0)
    std[stuck] = np.nan

    first, last = std.dropna(how="all").index[[0, -1]]
    std = std.loc[first:last]                                   # full 5-min grid: gaps stay visible as empty rows
    return std, dict(raw_rows=len(raw), raw_step_s=round(step_s, 1), min_points_per_5min=need,
                     raw_values_impossible=n_bad, slots_stuck=int(stuck.sum()))


def timezone_check(std, lon):
    """Minutes between the power-weighted centre of the day and solar noon (median over months).
    Near 0 means the UTC conversion is right; about +/-60 would mean a daylight-saving or zone error."""
    s = std["power_kw"].clip(lower=0).dropna()
    s = s[s > 0]
    mins = s.index.hour * 60 + s.index.minute + 2.5               # centre of the 5-min interval, UTC
    doy = s.index.dayofyear
    b = 2 * np.pi * (doy - 81) / 364
    eot = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    noon = 720 - 4 * lon - eot                                    # solar noon in UTC minutes
    diff = ((mins - noon + 720) % 1440) - 720
    w = pd.DataFrame({"d": diff * s.values, "w": s.values, "m": s.index.month})
    per_month = w.groupby("m")["d"].sum() / w.groupby("m")["w"].sum()
    return round(float(per_month.median()), 1), round(float(per_month.max() - per_month.min()), 1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "log.txt").unlink(missing_ok=True)
    plants, report = [], []
    for spec in PLANTS:
        pid = spec["plant_id"]
        log(f"[convert] {pid} ...")
        std, info = standardize(spec["reader"](), spec["capacity_kw"])
        cap, note = spec["capacity_kw"], "documented"
        if not cap:
            cap, note = math.ceil(std["power_kw"].max() * 100) / 100, "estimated (highest 5-min mean power)"
        out = std.reset_index().rename(columns={"index": "timestamp_utc"})
        out.columns = ["timestamp_utc", "power_kw", "current_a"]
        out["timestamp_utc"] = out["timestamp_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out.to_csv(OUT / f"{pid}.csv", index=False, float_format="%.4f")
        tz_med, tz_spread = timezone_check(std, spec["longitude"])
        plants.append({k: spec[k] for k in ("plant_id", "latitude", "longitude", "tilt_deg", "azimuth_deg")}
                      | {"capacity_kw": cap, "capacity_note": note, "source": spec["source"]})
        report.append({"plant_id": pid, **info,
                       "start_utc": out["timestamp_utc"].iloc[0], "end_utc": out["timestamp_utc"].iloc[-1],
                       "days": round(len(std) / 288, 1), "slots_5min": len(std),
                       "power_valid_pct": round(100 * float(std["power_kw"].notna().mean()), 1),
                       "current_valid_pct": round(100 * float(std["current_a"].notna().mean()), 1),
                       "tz_check_min": tz_med, "tz_check_spread_min": tz_spread})
        log(f"[convert] {pid}: {report[-1]}")
    raw = []
    for f in sorted(p for p in RAW.rglob("*") if p.is_file()):
        h = hashlib.md5()
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        raw.append({"file": f.as_posix(), "bytes": f.stat().st_size, "md5": h.hexdigest()})
    pd.DataFrame(raw).to_csv(OUT / "raw_files.csv", index=False)
    log(f"[convert] {len(raw)} raw files listed in raw_files.csv")
    cols = ["plant_id", "capacity_kw", "capacity_note", "latitude", "longitude", "tilt_deg", "azimuth_deg", "source"]
    pd.DataFrame(plants)[cols].to_csv(OUT / "plants.csv", index=False)
    pd.DataFrame(report).to_csv(OUT / "conversion_report.csv", index=False)
    log(f"[convert] done -> {OUT}/")


if __name__ == "__main__":
    main()
