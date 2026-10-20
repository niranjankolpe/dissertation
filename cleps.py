"""CLePS: Cross-site, Leakage-safe and Persistence-Skill evaluation of ML PV power forecasting.

Flow (proposal Section 4):  standard files -> samples -> sufficiency check -> leakage-safe split -> models
                            -> E1 leakage test / E2 cross-site test -> one results row per run -> report (E3).

    python cleps.py run --exp e1 e2 --models ridge gbt --horizons 1h       # writes results/results.csv
    python cleps.py report                                                 # tables + figures from results.csv

Input : output/1_standard/<plant_id>.csv and plants.csv            (made by convert.py)
Output: output/2_run/sufficiency.csv     plants used / excluded, with sample counts and block periods
        output/2_run/results.csv         one row per run (all experiments, all seeds)
        output/2_run/log.txt             console messages
        output/3_report/*.csv            summary, E1 skill inflation, E2 cross-site loss, E3 skill vs persistence
        output/3_report/figures/*.png    bar charts
        output/3_report/log.txt
"""
import argparse
import json
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
    return dict(params=json.dumps(params), n_train=len(train), n_test=len(test),
                **metrics(test["y"].to_numpy(), yhat, test["persistence"].to_numpy(), cap))


def persistence_row(test, cap):
    y, p = test["y"].to_numpy(), test["persistence"].to_numpy()
    return dict(params="", n_train=0, n_test=len(test), **metrics(y, p, p, cap))


# =============================== 5. experiments ===============================
def split(S, path, seed):
    if path == "a_random_split":                              # rows shuffled before a random 60/20/20 split
        idx = np.random.default_rng(seed).permutation(len(S))
        a, b = int(len(S) * SPLIT[0]), int(len(S) * (SPLIT[0] + SPLIT[1]))
        return S.iloc[idx[:a]], S.iloc[idx[a:b]], S.iloc[idx[b:]]
    return tuple(S.iloc[i] for i in chrono_blocks(len(S)))


def run(args):
    meta, series = load_plants()
    plants = args.plants or list(meta.index)
    samples, suff = {}, []
    for pid in plants:
        for hz in args.horizons:
            X = build_samples(series[pid], meta.loc[pid], HORIZONS[hz])
            suff.append(sufficiency(pid, hz, X, series[pid]))
            samples[pid, hz] = X[X["valid"]]
    suff = pd.DataFrame(suff)
    suff.to_csv(RES / "sufficiency.csv", index=False)
    log(suff[["plant", "horizon", "span_days", "valid_samples", "train_valid_days", "val_valid_days",
                "test_valid_days", "status"]].to_string(index=False))
    used = {(r.plant, r.horizon) for r in suff.itertuples() if r.status == "used"}

    rows = []

    def add(exp, pid, hz, model, setting, path, seed, res):
        rows.append(dict(experiment=exp, plant=pid, horizon=hz, model=model, setting=setting,
                         leakage_path=path, seed=seed, **res))
        r = rows[-1]
        log(f"  {exp} {pid:16s} {hz:5s} {model:17s} {setting:20s} {path:26s} s{seed} "
              f"skill={r['skill']:+.3f} nRMSE={r['nrmse_pct']:.2f}%")

    for pid, hz in sorted(used):
        S, cap = samples[pid, hz], meta.loc[pid, "capacity_kw"]
        if "e1" in args.exp:
            for path in PATHS:
                for model in ["smart_persistence"] + args.models:
                    seeds = range(N_SEEDS) if (model in RANDOM_MODELS or path == "a_random_split") else [0]
                    for seed in seeds:
                        tr, va, te = split(S, path, seed)
                        if model == "smart_persistence":
                            if path in ("none", "a_random_split"):   # leaks b-d do not change persistence
                                add("E1", pid, hz, model, "within-site", path, seed, persistence_row(te, cap))
                            continue
                        add("E1", pid, hz, model, "within-site", path, seed, evaluate(model, tr, va, te, cap, path, seed))
        if "e2" in args.exp:
            others = [o for o in plants if o != pid and (o, hz) in used]
            if len(others) < 2:
                log(f"  E2 skipped for {pid} {hz}: needs >= 2 other usable plants")
                continue
            blocks = {o: split(samples[o, hz], "none", 0) for o in others}
            tr = pd.concat([blocks[o][0] for o in others])
            va = pd.concat([blocks[o][1] for o in others])
            te = split(S, "none", 0)[2]                          # same test block as within-site
            for model in args.models:
                for seed in (range(N_SEEDS) if model in RANDOM_MODELS else [0]):
                    add("E2", pid, hz, model, "leave-one-plant-out", "none", seed,
                        evaluate(model, tr, va, te, cap, "none", seed, refit_on_train_val=True))

    new = pd.DataFrame(rows)
    out = RES / "results.csv"
    if out.exists():                                          # keep earlier runs, replace re-run rows
        new = pd.concat([pd.read_csv(out), new]).drop_duplicates(KEY, keep="last")
    new.sort_values(KEY).to_csv(out, index=False, float_format="%.5f")
    log(f"[run] {len(rows)} runs -> {out}")


# =============================== 6. report (E3: skill tables and figures) ===============================
def report(_args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = pd.read_csv(RES / "results.csv")
    figs = REP / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    grp = ["experiment", "plant", "horizon", "model", "setting", "leakage_path"]
    summ = r.groupby(grp).agg(runs=("seed", "count"), skill_mean=("skill", "mean"), skill_sd=("skill", "std"),
                              nrmse_pct_mean=("nrmse_pct", "mean"), rmse_kw_mean=("rmse_kw", "mean"),
                              mae_kw_mean=("mae_kw", "mean"), n_test=("n_test", "first")).reset_index()
    summ.to_csv(REP / "summary.csv", index=False, float_format="%.4f")

    e1 = summ[(summ.experiment == "E1") & (summ.model != "smart_persistence")]
    safe = e1[e1.leakage_path == "none"].set_index(["plant", "horizon", "model"])["skill_mean"]
    infl = e1[e1.leakage_path != "none"].copy()
    infl["skill_safe"] = [safe.get((p, h, m), np.nan) for p, h, m in zip(infl.plant, infl.horizon, infl.model)]
    infl["skill_inflation"] = infl["skill_mean"] - infl["skill_safe"]
    infl = infl[["plant", "horizon", "model", "leakage_path", "skill_safe", "skill_mean", "skill_inflation"]]
    infl.to_csv(REP / "e1_skill_inflation.csv", index=False, float_format="%.4f")

    e2 = summ[summ.experiment == "E2"].set_index(["plant", "horizon", "model"])["skill_mean"]
    loss = safe.to_frame("skill_within_site").join(e2.rename("skill_leave_one_plant_out"), how="inner")
    loss["cross_site_loss"] = loss["skill_within_site"] - loss["skill_leave_one_plant_out"]
    loss.reset_index().to_csv(REP / "e2_cross_site_loss.csv", index=False, float_format="%.4f")

    e3 = summ[(summ.model != "smart_persistence") & (summ.leakage_path == "none")]
    e3 = e3.pivot_table(index=["plant", "horizon", "model"], columns="setting", values="skill_mean")
    e3["beats_persistence"] = e3.apply(lambda r: bool((r.dropna() > 0).all()) if r.notna().any() else None, axis=1)
    e3.reset_index().to_csv(REP / "e3_skill_vs_persistence.csv", index=False, float_format="%.4f")

    for hz in sorted(r.horizon.unique()):
        d = infl[infl.horizon == hz]
        if len(d):
            t = d.pivot_table(index=["plant", "model"], columns="leakage_path", values="skill_inflation")
            ax = t.plot.barh(figsize=(10, 0.5 * len(t) + 2))
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Skill inflation (leaky skill - leakage-safe skill)")
            ax.set_title(f"E1 leakage test, horizon {hz}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            plt.tight_layout(); plt.savefig(figs / f"e1_inflation_{hz}.png", dpi=150); plt.close()
        d = loss.reset_index()
        d = d[d.horizon == hz]
        if len(d):
            t = d.pivot_table(index="plant", columns="model", values="cross_site_loss")
            ax = t.plot.barh(figsize=(10, 0.5 * len(t) + 2))
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("Cross-site loss (within-site skill - leave-one-plant-out skill)")
            ax.set_title(f"E2 cross-site test, horizon {hz}")
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
            plt.tight_layout(); plt.savefig(figs / f"e2_cross_site_loss_{hz}.png", dpi=150); plt.close()
    log(f"[report] tables -> {REP}/, figures -> {figs}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["run", "report"])
    ap.add_argument("--exp", nargs="+", default=["e1", "e2"], choices=["e1", "e2"])
    ap.add_argument("--models", nargs="+", default=["ridge", "gbt", "mlp", "lstm"], choices=list(GRIDS))
    ap.add_argument("--horizons", nargs="+", default=list(HORIZONS), choices=list(HORIZONS))
    ap.add_argument("--plants", nargs="+", default=None, help="plant_ids (default: all in plants.csv)")
    a = ap.parse_args()
    t0 = time.time()
    folder = RES if a.command == "run" else REP
    folder.mkdir(parents=True, exist_ok=True)
    LOG = folder / "log.txt"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  python cleps.py {' '.join(__import__('sys').argv[1:])}\n")
    run(a) if a.command == "run" else report(a)
    log(f"[done] {time.time() - t0:.0f} s")
