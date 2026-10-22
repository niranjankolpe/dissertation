"""CLePS: Cross-site, Leakage-safe and Persistence-Skill evaluation of ML PV power forecasting.

Flow (proposal Section 4):  standard files -> samples -> sufficiency check -> leakage-safe split -> models
                            -> E1 leakage test / E2 cross-site test -> one results row per run -> report (E3).

    python cleps.py run --exp e1 e2 --models ridge gbt --horizons 1h       # writes results/results.csv
    python cleps.py report                                                 # tables + figures from results.csv
    python cleps.py run --exp e1 --models mlp lstm --plants la_reunion_inv2 --paths none --seeds 1   # quick check

Input : output/1_standard/<plant_id>.csv and plants.csv            (made by convert.py)
Output: output/2_run/sufficiency.csv     plants used / excluded, with sample counts and block periods
        output/2_run/results.csv         one row per run (all experiments, all seeds)
        output/2_run/daily_errors.csv    per run and test day: squared-error sums of model and persistence
        output/2_run/log.txt             console messages
        output/3_report/*.csv            summary, E1 skill inflation, E2 cross-site loss, E3 skill vs persistence
        output/3_report/figures/*.png    bar charts
        output/3_report/log.txt
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA, RES, REP = Path("output/1_standard"), Path("output/2_run"), Path("output/3_report")
LOG = None   # log file of the current command
STEP_MIN = 5
HORIZONS = {"15min": 3, "1h": 12}           # horizon -> number of 5-min steps
N_LAGS = 12                                 # normalised power at t-55 min ... t
SUN_MIN_DEG = 5.0                           # sample used only if sun >= 5 deg at t and t+h
MAX_FILL = 3                                # input gaps <= 3 steps filled from last value; targets never filled
SPLIT = (0.6, 0.2, 0.2)                     # chronological train / validation / test
MIN_MONTHS_DAYS, MIN_VALID_DAYS, VALID_DAY_SHARE = 182, 20, 0.70   # sufficiency rules (Section 4.2)
N_SEEDS = 5
RANDOM_MODELS = {"gbt", "mlp", "lstm"}      # models with random elements get N_SEEDS runs
GRIDS = {                                   # small grids of similar size, chosen on validation RMSE
    "ridge": [{"alpha": a} for a in (0.1, 1.0, 10.0)],
    "gbt": [{"num_leaves": n} for n in (15, 31, 63)],
    "mlp": [{"hidden": h} for h in ((32,), (64, 32), (128, 64))],
    "lstm": [{"hidden": h} for h in (16, 32, 64)],
}
PATHS = ["none", "a_random_split", "b_full_data_scaling", "c_test_set_tuning", "d_target_adjacent_current"]
KEY = ["experiment", "plant", "horizon", "model", "setting", "leakage_path", "seed"]
RES_COLS = KEY + ["fingerprint", "params", "n_train", "n_test", "rmse_kw", "mae_kw", "nrmse_pct",
                  "rmse_persistence_kw", "skill"]
DAY_COLS = KEY + ["fingerprint", "solar_day", "sse", "sse_persistence", "n"]
HEARTBEAT_S = 60
N_BOOT, CI = 1000, 0.95                     # day-block bootstrap for confidence intervals in the report


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# =============================== 1. standard files -> samples ===============================
def load_plants():
    meta = pd.read_csv(DATA / "plants.csv").set_index("plant_id")
    series = {}
    for pid in meta.index:
        d = pd.read_csv(DATA / f"{pid}.csv", parse_dates=["timestamp_utc"]).set_index("timestamp_utc")
        series[pid] = d.asfreq(f"{STEP_MIN}min")
    return meta, series


def sun_and_clearsky(index, m):
    """Sun position and clear-sky irradiance (pvlib). Plane of array if tilt/azimuth known, else horizontal."""
    import pvlib
    loc = pvlib.location.Location(m.latitude, m.longitude, tz="UTC")
    sp = loc.get_solarposition(index)
    cs = loc.get_clearsky(index, model="ineichen", solar_position=sp)
    if pd.notna(m.tilt_deg) and pd.notna(m.azimuth_deg):
        poa = pvlib.irradiance.get_total_irradiance(m.tilt_deg, m.azimuth_deg, sp["apparent_zenith"], sp["azimuth"],
                                                    cs["dni"], cs["ghi"], cs["dhi"])["poa_global"]
    else:
        poa = cs["ghi"]
    return sp["apparent_elevation"].to_numpy(), sp["azimuth"].to_numpy(), poa.fillna(0).to_numpy() / 1000.0


def fill_short_gaps(s, limit):
    gap = s.isna()
    run = gap.groupby((~gap).cumsum()).transform("sum")       # length of the gap each NaN belongs to
    return s.ffill().where(~gap | (run <= limit))


def build_samples(df, m, steps):
    """One row per issue time t: inputs known at t, target = normalised power at t+h (never filled)."""
    p = df["power_kw"] / m.capacity_kw
    pf = fill_short_gaps(p, MAX_FILL)
    elev, azi, cs = sun_and_clearsky(df.index, m)
    solar_hour = ((df.index.hour + df.index.minute / 60 + m.longitude / 15) % 24).to_numpy()

    X = pd.DataFrame(index=df.index)
    for k in range(N_LAGS - 1, -1, -1):                      # oldest -> newest (order used by the LSTM)
        X[f"p_lag{k}"] = pf.shift(k)
    for tag, sh in (("t", 0), ("th", -steps)):               # sun position and time at t and at t+h
        e, a, hr = pd.Series(elev).shift(sh), pd.Series(np.deg2rad(azi)).shift(sh), pd.Series(solar_hour).shift(sh)
        X[f"elev_{tag}"] = e.to_numpy()
        X[f"azi_sin_{tag}"], X[f"azi_cos_{tag}"] = np.sin(a).to_numpy(), np.cos(a).to_numpy()
        X[f"hour_sin_{tag}"] = np.sin(2 * np.pi * hr / 24).to_numpy()
        X[f"hour_cos_{tag}"] = np.cos(2 * np.pi * hr / 24).to_numpy()
    X["y"] = p.shift(-steps)                                  # target, raw (not filled)
    X["current_th"] = df["current_a"].shift(-steps)           # only for leakage path (d)
    X["cs_t"], X["cs_th"] = cs, pd.Series(cs).shift(-steps).to_numpy()
    X["persistence"] = X["p_lag0"] / X["cs_t"] * X["cs_th"]  # smart persistence: clear-sky index at t held to t+h
    X["solar_day"] = (df.index + pd.to_timedelta(m.longitude / 15, unit="h")).date
    daytime = (X["elev_t"] >= SUN_MIN_DEG) & (X["elev_th"] >= SUN_MIN_DEG)
    X["daytime"] = daytime
    X["valid"] = daytime & X.drop(columns=["solar_day", "daytime"]).notna().all(axis=1) & (X["cs_t"] > 0)
    return X


FEATURES = [f"p_lag{k}" for k in range(N_LAGS - 1, -1, -1)] + [
    f"{n}_{tag}" for tag in ("t", "th") for n in ("elev", "azi_sin", "azi_cos", "hour_sin", "hour_cos")]


# =============================== 2. sufficiency + split ===============================
def chrono_blocks(n):
    a, b = int(n * SPLIT[0]), int(n * (SPLIT[0] + SPLIT[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


def sufficiency(pid, horizon, X, df):
    span = (df["power_kw"].last_valid_index() - df["power_kw"].first_valid_index()).days
    S = X[X["valid"]]
    per_day = X[X["daytime"]].groupby("solar_day")["valid"].mean()
    good_days = set(per_day[per_day >= VALID_DAY_SHARE].index)
    row = dict(plant=pid, horizon=horizon, span_days=span, daytime_slots=int(X["daytime"].sum()),
               valid_samples=len(S), valid_pct=round(100 * len(S) / max(1, X["daytime"].sum()), 1))
    ok = span >= MIN_MONTHS_DAYS
    for name, idx in zip(("train", "val", "test"), chrono_blocks(len(S))):
        days = S.iloc[idx]["solar_day"]
        row[f"{name}_samples"] = len(idx)
        row[f"{name}_period"] = f"{days.iloc[0]} to {days.iloc[-1]}" if len(idx) else ""
        row[f"{name}_valid_days"] = len(set(days) & good_days)
        ok &= row[f"{name}_valid_days"] >= MIN_VALID_DAYS
    row["status"] = "used" if ok else "excluded"
    return row


# =============================== 3. models ===============================
class Model:
    """Common wrapper: fit(X, y) / predict(X) on scaled features."""

    def __init__(self, name, params, seed):
        self.name, self.params, self.seed = name, params, seed

    def fit(self, X, y):
        if self.name == "ridge":
            from sklearn.linear_model import Ridge
            self.m = Ridge(alpha=self.params["alpha"]).fit(X, y)
        elif self.name == "gbt":
            from lightgbm import LGBMRegressor
            self.m = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=self.params["num_leaves"],
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                   random_state=self.seed, verbose=-1).fit(X, y)
        elif self.name == "mlp":
            from sklearn.neural_network import MLPRegressor
            self.m = MLPRegressor(hidden_layer_sizes=self.params["hidden"], max_iter=200, early_stopping=True,
                                  random_state=self.seed).fit(X, y)
        elif self.name == "lstm":
            self.m = fit_lstm(X, y, self.params["hidden"], self.seed)
        return self

    def predict(self, X):
        return predict_lstm(self.m, X) if self.name == "lstm" else self.m.predict(X)


def fit_lstm(X, y, hidden, seed, epochs=15, batch=512, lr=1e-3):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)

    class Net(torch.nn.Module):
        def __init__(self, n_static):
            super().__init__()
            self.lstm = torch.nn.LSTM(1, hidden, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden + n_static, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))

        def forward(self, x):
            seq, static = x[:, :N_LAGS].unsqueeze(-1), x[:, N_LAGS:]
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], static], dim=1)).squeeze(-1)

    net = Net(X.shape[1] - N_LAGS)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    return net


def predict_lstm(net, X):
    import torch
    with torch.no_grad():
        return net(torch.tensor(X, dtype=torch.float32)).numpy()


# =============================== 4. one evaluation run ===============================
def daily(test, yhat):
    """Squared-error sums per test day (normalised units), so the report can bootstrap over days."""
    y, p = test["y"].to_numpy(), test["persistence"].to_numpy()
    d = pd.DataFrame({"solar_day": test["solar_day"].astype(str).to_numpy(),
                      "sse": (yhat - y) ** 2, "sse_persistence": (p - y) ** 2, "n": 1})
    return d.groupby("solar_day", as_index=False).sum()


def metrics(y, yhat, ypers, cap):
    rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
    rmse_p = float(np.sqrt(np.mean((ypers - y) ** 2)))
    return dict(rmse_kw=rmse * cap, mae_kw=float(np.mean(np.abs(yhat - y))) * cap, nrmse_pct=100 * rmse,
                rmse_persistence_kw=rmse_p * cap, skill=1 - rmse / rmse_p)


def evaluate(model, train, val, test, cap, path="none", seed=0, refit_on_train_val=False):
    """Scale (fit on train), choose params on validation RMSE, score once on test. `path` switches on one leak."""
    from sklearn.preprocessing import StandardScaler
    feats = FEATURES + (["current_th"] if path == "d_target_adjacent_current" else [])
    fit_rows = pd.concat([train, val, test]) if path == "b_full_data_scaling" else train
    sc = StandardScaler().fit(fit_rows[feats])
    Xtr, Xva, Xte = (sc.transform(d[feats]) for d in (train, val, test))
    choose_on = (Xte, test["y"].to_numpy()) if path == "c_test_set_tuning" else (Xva, val["y"].to_numpy())

    best = None
    for params in GRIDS[model]:
        m = Model(model, params, seed).fit(Xtr, train["y"].to_numpy())
        score = np.sqrt(np.mean((m.predict(choose_on[0]) - choose_on[1]) ** 2))
        if best is None or score < best[0]:
            best = (score, params, m)
    _, params, m = best
    if refit_on_train_val:                                    # E2: final model uses train + validation blocks
        tv = pd.concat([train, val])
        sc = StandardScaler().fit(tv[feats])
        Xte = sc.transform(test[feats])
        m = Model(model, params, seed).fit(sc.transform(tv[feats]), tv["y"].to_numpy())
    yhat = m.predict(Xte)
    return dict(params=json.dumps(params), n_train=len(train), n_test=len(test), _daily=daily(test, yhat),
                **metrics(test["y"].to_numpy(), yhat, test["persistence"].to_numpy(), cap))


def persistence_row(test, cap):
    y, p = test["y"].to_numpy(), test["persistence"].to_numpy()
    return dict(params="", n_train=0, n_test=len(test), _daily=daily(test, p), **metrics(y, p, p, cap))


# =============================== 5. experiments ===============================
def split(S, path, seed):
    if path == "a_random_split":                              # rows shuffled before a random 60/20/20 split
        idx = np.random.default_rng(seed).permutation(len(S))
        a, b = int(len(S) * SPLIT[0]), int(len(S) * (SPLIT[0] + SPLIT[1]))
        return S.iloc[idx[:a]], S.iloc[idx[a:b]], S.iloc[idx[b:]]
    return tuple(S.iloc[i] for i in chrono_blocks(len(S)))


# ---------- resumable storage: every finished run is appended at once; a restart skips finished runs ----------
def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def settings_id():
    """Changes when anything that affects a result changes (constants or the code that computes results)."""
    parts = [json.dumps([STEP_MIN, HORIZONS, N_LAGS, SUN_MIN_DEG, MAX_FILL, SPLIT, MIN_MONTHS_DAYS, MIN_VALID_DAYS,
                         VALID_DAY_SHARE, {k: [json.dumps(g) for g in v] for k, v in GRIDS.items()}])]
    for f in (sun_and_clearsky, fill_short_gaps, build_samples, chrono_blocks, Model, fit_lstm, predict_lstm,
              daily, metrics, evaluate, persistence_row, split):
        parts.append(inspect.getsource(f))
    return md5("".join(parts))


def repair(path, cols):
    """Make a csv safe to append to: header present and no half-written last line (from a stopped run)."""
    if path.exists() and path.stat().st_size:
        b = path.read_bytes()
        if not b.endswith(b"\n"):
            path.write_bytes(b[:b.rfind(b"\n") + 1])
        head = pd.read_csv(path, nrows=0).columns.tolist()
        if head != cols:                                       # older file layout -> rewrite with current columns
            old = pd.read_csv(path, on_bad_lines="skip")
            old.reindex(columns=cols).to_csv(path, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(path, index=False)


def load_results():
    r = pd.read_csv(RES / "results.csv", on_bad_lines="skip")
    r = r[r["skill"].notna()].drop_duplicates(KEY, keep="last")
    return r


def load_daily(r):
    p = RES / "daily_errors.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, on_bad_lines="skip").dropna(subset=["sse"])
    d = d.merge(r[KEY + ["fingerprint"]].dropna(subset=["fingerprint"]), on=KEY + ["fingerprint"])
    return d.drop_duplicates(KEY + ["solar_day"], keep="last")


def run(args):
    meta, series = load_plants()
    plants = args.plants or list(meta.index)
    samples, suff = {}, []
    for pid in (list(meta.index) if "e2" in args.exp else plants):   # E2 trains on all other plants
        for hz in args.horizons:
            X = build_samples(series[pid], meta.loc[pid], HORIZONS[hz])
            suff.append(sufficiency(pid, hz, X, series[pid]))
            samples[pid, hz] = X[X["valid"]]
    suff = pd.DataFrame(suff)
    sp = RES / "sufficiency.csv"
    if sp.exists():                                           # keep rows of plants/horizons not run now
        old = pd.read_csv(sp)
        old = old[~old.set_index(["plant", "horizon"]).index.isin(suff.set_index(["plant", "horizon"]).index)]
        suff = pd.concat([old, suff]).sort_values(["plant", "horizon"])
    suff.to_csv(sp, index=False)
    log(suff[["plant", "horizon", "span_days", "valid_samples", "train_valid_days", "val_valid_days",
                "test_valid_days", "status"]].to_string(index=False))
    used = {(r.plant, r.horizon) for r in suff.itertuples() if r.status == "used" and r.horizon in args.horizons
            and (r.plant, r.horizon) in samples}

    out, out_d = RES / "results.csv", RES / "daily_errors.csv"
    repair(out, RES_COLS)
    repair(out_d, DAY_COLS)
    done = set(map(tuple, load_results()[KEY + ["fingerprint"]].astype(str).to_numpy()))
    sid = settings_id()
    data_id = {pid: md5(json.dumps(meta.loc[pid].astype(str).to_dict()) + hashlib.md5(
        (DATA / f"{pid}.csv").read_bytes()).hexdigest()) for pid in meta.index}
    n_seeds, count = args.seeds, {"new": 0, "skipped": 0, "now": "starting"}
    order = sorted(u for u in used if u[0] in plants)
    total = 0                                                  # number of runs this command covers (done or not)
    for pid, hz in order:
        if "e1" in args.exp:
            for path in args.paths:
                for model in ["smart_persistence"] + args.models:
                    if model == "smart_persistence" and path not in ("none", "a_random_split"):
                        continue
                    total += n_seeds if (model in RANDOM_MODELS or path == "a_random_split") else 1
        if "e2" in args.exp and sum(1 for o in meta.index if o != pid and (o, hz) in used) >= 2:
            total += sum(n_seeds if m in RANDOM_MODELS else 1 for m in args.models)
    t_start = time.time()

    def heartbeat():                                           # one progress line per minute in the window
        try:
            con = open("CON", "w") if os.name == "nt" else sys.__stderr__
        except OSError:
            return
        while True:
            time.sleep(HEARTBEAT_S)
            done_n = count["new"] + count["skipped"]
            rate = count["new"] / max(1.0, time.time() - t_start)
            eta = f"{(total - done_n) / rate / 60:.0f} min" if rate > 0 else "?"
            con.write(f"[{time.strftime('%H:%M')}] {' '.join(args.models)}: {done_n}/{total} runs "
                      f"({100 * done_n / max(1, total):.0f}%), now {count['now']}, about {eta} left\n")
            con.flush()

    threading.Thread(target=heartbeat, daemon=True).start()

    def todo(exp, pid, hz, model, setting, path, seed, fp):
        count["now"] = f"{exp} {pid} {hz} {model} {path} s{seed}"
        if (exp, pid, hz, model, setting, path, str(seed), fp) in done:
            count["skipped"] += 1
            return False
        return True

    def add(exp, pid, hz, model, setting, path, seed, fp, res):
        k = dict(experiment=exp, plant=pid, horizon=hz, model=model, setting=setting, leakage_path=path, seed=seed,
                 fingerprint=fp)
        res.pop("_daily").assign(**k)[DAY_COLS].to_csv(out_d, mode="a", header=False, index=False,
                                                       float_format="%.6g")
        pd.DataFrame([dict(**k, **res)])[RES_COLS].to_csv(out, mode="a", header=False, index=False,
                                                         float_format="%.5f")   # written last = run finished
        count["new"] += 1
        log(f"  {exp} {pid:16s} {hz:5s} {model:17s} {setting:20s} {path:26s} s{seed} "
            f"skill={res['skill']:+.3f} nRMSE={res['nrmse_pct']:.2f}%")

    for pid, hz in order:
        S, cap = samples[pid, hz], meta.loc[pid, "capacity_kw"]
        if "e1" in args.exp:
            fp = md5(sid + data_id[pid])
            for path in args.paths:
                for model in ["smart_persistence"] + args.models:
                    if model == "smart_persistence" and path not in ("none", "a_random_split"):
                        continue                               # leaks b-d do not change persistence
                    seeds = range(n_seeds) if (model in RANDOM_MODELS or path == "a_random_split") else [0]
                    for seed in seeds:
                        if not todo("E1", pid, hz, model, "within-site", path, seed, fp):
                            continue
                        tr, va, te = split(S, path, seed)
                        res = persistence_row(te, cap) if model == "smart_persistence" else \
                            evaluate(model, tr, va, te, cap, path, seed)
                        add("E1", pid, hz, model, "within-site", path, seed, fp, res)
        if "e2" in args.exp:
            others = sorted(o for o in meta.index if o != pid and (o, hz) in used)
            if len(others) < 2:
                log(f"  E2 skipped for {pid} {hz}: needs >= 2 other usable plants")
                continue
            fp = md5(sid + data_id[pid] + "".join(data_id[o] for o in others))   # changes if the plant set changes
            jobs = [(m, s) for m in args.models for s in (range(n_seeds) if m in RANDOM_MODELS else [0])
                    if todo("E2", pid, hz, m, "leave-one-plant-out", "none", s, fp)]
            if jobs:
                blocks = {o: split(samples[o, hz], "none", 0) for o in others}
                tr = pd.concat([blocks[o][0] for o in others])
                va = pd.concat([blocks[o][1] for o in others])
                te = split(S, "none", 0)[2]                      # same test block as within-site
                for model, seed in jobs:
                    add("E2", pid, hz, model, "leave-one-plant-out", "none", seed, fp,
                        evaluate(model, tr, va, te, cap, "none", seed, refit_on_train_val=True))
        log(f"[run] {pid} {hz}: {count['new']} new runs so far, {count['skipped']} already done (skipped)")
    log(f"[run] finished: {count['new']} new runs, {count['skipped']} skipped -> {out}")


# =============================== 6. report (E3: skill tables and figures) ===============================
GRP = ["experiment", "plant", "horizon", "model", "setting", "leakage_path"]


def boot_days(dd):
    """Per configuration: test days x (sse, sse_persistence, n), summed over seeds."""
    if dd is None:
        return {}
    g = dd.groupby(GRP + ["solar_day"])[["sse", "sse_persistence", "n"]].sum()
    return {k: v.droplevel(GRP) for k, v in g.groupby(level=GRP)}


def skill_of(a):                                              # a: (..., days, 3) -> pooled skill
    s = a.sum(axis=-2)
    return 1 - np.sqrt(s[..., 0] / s[..., 2]) / np.sqrt(s[..., 1] / s[..., 2])


def ci(days, k1, k0=None, seed=0):
    """Day-block bootstrap CI of skill(k1) - skill(k0) (or of skill(k1) alone).
    Same test days -> paired resampling of days; different test days (random split) -> independent resampling."""
    if k1 not in days or (k0 is not None and k0 not in days):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    a = days[k1]
    if k0 is None:
        x = a.to_numpy()
        b = skill_of(x[rng.integers(0, len(x), (N_BOOT, len(x)))])
    else:
        z = days[k0]
        if a.index.equals(z.index):
            i = rng.integers(0, len(a), (N_BOOT, len(a)))
            b = skill_of(a.to_numpy()[i]) - skill_of(z.to_numpy()[i])
        else:
            b = (skill_of(a.to_numpy()[rng.integers(0, len(a), (N_BOOT, len(a)))])
                 - skill_of(z.to_numpy()[rng.integers(0, len(z), (N_BOOT, len(z)))]))
    q = (1 - CI) / 2
    return float(np.quantile(b, q)), float(np.quantile(b, 1 - q))


def add_ci(df, days, k1, k0, col):
    lo, hi = zip(*[ci(days, a, b) for a, b in zip(k1, k0)]) if len(df) else ((), ())
    df[col + "_ci_low"], df[col + "_ci_high"] = lo, hi
    df[col + "_significant"] = [None if np.isnan(l) else bool(l > 0 or h < 0) for l, h in zip(lo, hi)]
    return df


def report(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = load_results()
    r = r[r.plant.isin(pd.read_csv(DATA / "plants.csv")["plant_id"])]      # plants removed from plants.csv dropped
    figs = REP / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    grp = ["experiment", "plant", "horizon", "model", "setting", "leakage_path"]
    summ = r.groupby(grp).agg(runs=("seed", "count"), skill_mean=("skill", "mean"), skill_sd=("skill", "std"),
                              nrmse_pct_mean=("nrmse_pct", "mean"), rmse_kw_mean=("rmse_kw", "mean"),
                              mae_kw_mean=("mae_kw", "mean"), n_test=("n_test", "first")).reset_index()
    days = boot_days(load_daily(r))
    summ = add_ci(summ, days, [tuple(x) for x in summ[GRP].itertuples(index=False)], [None] * len(summ), "skill")
    summ.to_csv(REP / "summary.csv", index=False, float_format="%.4f")

    e1 = summ[(summ.experiment == "E1") & (summ.model != "smart_persistence")]
    safe = e1[e1.leakage_path == "none"].set_index(["plant", "horizon", "model"])["skill_mean"]
    infl = e1[e1.leakage_path != "none"].copy()
    infl["skill_safe"] = [safe.get((p, h, m), np.nan) for p, h, m in zip(infl.plant, infl.horizon, infl.model)]
    infl["skill_inflation"] = infl["skill_mean"] - infl["skill_safe"]
    infl = infl[["plant", "horizon", "model", "leakage_path", "skill_safe", "skill_mean", "skill_inflation"]]
    infl = add_ci(infl, days, [("E1", p, h, m, "within-site", l) for p, h, m, l in
                               zip(infl.plant, infl.horizon, infl.model, infl.leakage_path)],
                  [("E1", p, h, m, "within-site", "none") for p, h, m in zip(infl.plant, infl.horizon, infl.model)],
                  "skill_inflation")
    infl.to_csv(REP / "e1_skill_inflation.csv", index=False, float_format="%.4f")

    e2 = summ[summ.experiment == "E2"].set_index(["plant", "horizon", "model"])["skill_mean"]
    loss = safe.to_frame("skill_within_site").join(e2.rename("skill_leave_one_plant_out"), how="inner")
    loss["cross_site_loss"] = loss["skill_within_site"] - loss["skill_leave_one_plant_out"]
    ix = list(loss.index)
    loss = add_ci(loss, days, [("E1", p, h, m, "within-site", "none") for p, h, m in ix],
                  [("E2", p, h, m, "leave-one-plant-out", "none") for p, h, m in ix], "cross_site_loss")
    loss.reset_index().to_csv(REP / "e2_cross_site_loss.csv", index=False, float_format="%.4f")

    e3 = summ[(summ.model != "smart_persistence") & (summ.leakage_path == "none")]
    e3 = e3.pivot_table(index=["plant", "horizon", "model"], columns="setting", values="skill_mean")
    e3["beats_persistence"] = e3.apply(lambda r: bool((r.dropna() > 0).all()) if r.notna().any() else None, axis=1)
    low = summ[(summ.model != "smart_persistence") & (summ.leakage_path == "none")].pivot_table(
        index=["plant", "horizon", "model"], columns="setting", values="skill_ci_low")
    e3["beats_persistence_significant"] = (low > 0).all(axis=1).where(low.notna().all(axis=1)).reindex(e3.index) \
        if len(low) else None                                 # every setting's 95% CI of skill above 0
    e3.reset_index().to_csv(REP / "e3_skill_vs_persistence.csv", index=False, float_format="%.4f")

    def bars(d, index, columns, col, xlabel, title, png):     # bar chart with 95% bootstrap CI as error bars
        t = d.pivot_table(index=index, columns=columns, values=col)
        lo = d.pivot_table(index=index, columns=columns, values=col + "_ci_low", dropna=False).reindex_like(t)
        hi = d.pivot_table(index=index, columns=columns, values=col + "_ci_high", dropna=False).reindex_like(t)
        err = np.stack([np.stack([(t - lo)[c], (hi - t)[c]]) for c in t.columns]) if lo.notna().any().any() else None
        ax = t.plot.barh(figsize=(10, 0.5 * len(t) + 2), xerr=err, capsize=2, error_kw=dict(lw=0.7))
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(xlabel + ("; bars = 95% CI" if err is not None else ""))
        ax.set_title(title)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        plt.tight_layout(); plt.savefig(figs / png, dpi=150); plt.close()

    for hz in sorted(r.horizon.unique()):
        d = infl[infl.horizon == hz]
        if len(d):
            bars(d, ["plant", "model"], "leakage_path", "skill_inflation",
                 "Skill inflation (leaky skill - leakage-safe skill)", f"E1 leakage test, horizon {hz}",
                 f"e1_inflation_{hz}.png")
        d = loss.reset_index()
        d = d[d.horizon == hz]
        if len(d):
            bars(d, "plant", "model", "cross_site_loss",
                 "Cross-site loss (within-site skill - leave-one-plant-out skill)", f"E2 cross-site test, horizon {hz}",
                 f"e2_cross_site_loss_{hz}.png")
    log(f"[report] tables -> {REP}/, figures -> {figs}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--exp", nargs="+", default=["e1", "e2"], choices=["e1", "e2"])
    ap.add_argument("--models", nargs="+", default=["ridge", "gbt", "mlp", "lstm"], choices=list(GRIDS))
    ap.add_argument("--horizons", nargs="+", default=list(HORIZONS), choices=list(HORIZONS))
    ap.add_argument("--plants", nargs="+", default=None, help="plant_ids (default: all in plants.csv)")
    ap.add_argument("--paths", nargs="+", default=PATHS, choices=PATHS, help="E1 leakage paths (default: all)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS, help="seeds for models with random elements (default 5)")
    a = ap.parse_args()
    t0 = time.time()
    folder = RES if a.command == "run" else REP
    folder.mkdir(parents=True, exist_ok=True)
    LOG = folder / "log.txt"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  python cleps.py {' '.join(__import__('sys').argv[1:])}\n")
    run(a) if a.command == "run" else report(a)
    log(f"[done] {time.time() - t0:.0f} s")
