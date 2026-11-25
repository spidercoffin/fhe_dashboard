"""
Clean + optimize cs-training.csv (Give Me Some Credit) for the FHE credit-scoring project.

Outputs (written next to this script's --out dir):
  cs-training-clean.csv        human-readable cleaned data (original units + flags)
  cs-training-model-ready.csv  scaled, bounded features (what the trusted client encrypts)
  model_artifacts.json         preprocessing parameters + logistic-regression weights/bias
  cleaning_report.json         what was changed and why (row counts, metrics)

Usage: python prepare_dataset.py --in cs-training.csv --out ./out
"""
import argparse, json, os
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss, log_loss

RENAME = {
    "SeriousDlqin2yrs": "default",
    "RevolvingUtilizationOfUnsecuredLines": "util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late30",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "late90",
    "NumberRealEstateLoansOrLines": "re_loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "late60",
    "NumberOfDependents": "dependents",
}


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


# Score scale. 400/850 and the band edges are the publicly reported i-Score ones; the two anchors below are OUR design
# choice: a customer at 5% calibrated default probability scores 700, at 1% scores 800.
LO, HI, ANCHORS = 400, 850, ((800, 0.01), (700, 0.05))
logit_ = lambda p: float(np.log(p / (1 - p)))
_pdo = (ANCHORS[0][0] - ANCHORS[1][0]) * np.log(2) / (logit_(ANCHORS[1][1]) - logit_(ANCHORS[0][1]))
SCORE = {"lo": LO, "hi": HI, "base_score": ANCHORS[1][0], "reference_pd": ANCHORS[1][1], "pdo": float(_pdo)}
BANDS = [("Weak", 400, 520), ("Unsatisfactory", 521, 625), ("Satisfactory", 626, 699), ("Good", 700, 799), ("Excellent", 800, 850)]


def to_score(z):
    k = SCORE["pdo"] / np.log(2)
    return np.clip(np.round(SCORE["base_score"] - k * (z - logit_(SCORE["reference_pd"]))), LO, HI)


def calib(y, p, bins=10):
    """Brier, log-loss, expected calibration error and a per-decile predicted-vs-observed table."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop"); d = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return {"brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
            "ece": float((d.n * (d.pred - d.obs).abs()).sum() / d.n.sum()),
            "deciles_pred_obs": [[round(a, 4), round(b, 4)] for a, b in zip(d.pred, d.obs)]}


def clean(df, rep):
    df = df.copy()
    rep["rows_in"] = len(df)

    # 1. drop invalid / duplicate rows
    bad_age = df["age"] < 18
    rep["dropped_age_lt_18"] = int(bad_age.sum())
    df = df[~bad_age]
    # NOTE: no drop_duplicates(): the original file has a unique row id, so identical
    # feature rows are different customers with the same profile, not duplicate records.

    # 2. 96/98 are placeholder codes in all three late-payment columns (same 269 rows,
    #    ~55% default rate) -> keep the signal as a flag, neutralise the fake counts.
    sent = (df[["late30", "late60", "late90"]] >= 96).any(axis=1)
    df["late_code_flag"] = sent.astype(int)
    for c in ["late30", "late60", "late90"]:
        df.loc[sent, c] = 0
        df[c] = df[c].clip(0, 10)
    rep["late_placeholder_rows_flagged"] = int(sent.sum())

    # 3. income missing/zero: DebtRatio is then a raw debt *amount*, not a ratio.
    inc_missing = df["income"].isna() | (df["income"] <= 0)
    df["income_missing"] = inc_missing.astype(int)
    rep["income_missing_or_zero"] = int(inc_missing.sum())
    med_ratio = float(df.loc[~inc_missing, "debt_ratio"].median())
    df.loc[inc_missing, "debt_ratio"] = med_ratio          # not a true ratio -> neutral value
    med_income = float(df.loc[~inc_missing, "income"].median())
    df.loc[inc_missing, "income"] = med_income
    rep["impute_debt_ratio_median"] = med_ratio
    rep["impute_income_median"] = med_income

    # 4. dependents: missing exactly where income is missing -> impute 0 (mode)
    rep["dependents_missing"] = int(df["dependents"].isna().sum())
    df["dependents"] = df["dependents"].fillna(0)

    # 5. cap outliers at sensible bounds (not blind percentile chopping)
    caps = {"util": (0, 2.0), "debt_ratio": (0, 5.0), "age": (18, 100),
            "open_lines": (0, 40), "re_loans": (0, 10), "dependents": (0, 8)}
    inc_hi = float(df.loc[df.income_missing == 0, "income"].quantile(0.995))
    caps["income"] = (0, inc_hi)
    for c, (lo, hi) in caps.items():
        rep.setdefault("clipped", {})[c] = int(((df[c] < lo) | (df[c] > hi)).sum())
        df[c] = df[c].clip(lo, hi)
    rep["caps"] = {k: list(map(float, v)) for k, v in caps.items()}

    # 6. downcast dtypes
    ints = ["default", "age", "late30", "late60", "late90", "open_lines", "re_loans",
            "dependents", "late_code_flag", "income_missing"]
    df[ints] = df[ints].astype("int16")
    df["default"] = df["default"].astype("int8")
    df[["util", "debt_ratio", "income"]] = df[["util", "debt_ratio", "income"]].round(6).astype("float32")
    return df.reset_index(drop=True)


FEATURES = ["util", "age", "log_income", "log_debt_ratio", "open_lines", "re_loans",
            "dependents", "late_total", "any_late90", "late_code_flag", "income_missing"]


def engineer(df):
    X = pd.DataFrame(index=df.index)
    X["util"] = df["util"]
    X["age"] = df["age"]
    X["log_income"] = np.log1p(df["income"])
    X["log_debt_ratio"] = np.log1p(df["debt_ratio"])
    X["open_lines"] = df["open_lines"]
    X["re_loans"] = df["re_loans"]
    X["dependents"] = df["dependents"]
    X["late_total"] = (df["late30"] + df["late60"] + df["late90"]).clip(0, 10)
    X["any_late90"] = (df["late90"] > 0).astype(int)
    X["late_code_flag"] = df["late_code_flag"]
    X["income_missing"] = df["income_missing"]
    return X[FEATURES].astype(float)


def main(inp, out):
    os.makedirs(out, exist_ok=True)
    raw = pd.read_csv(inp, index_col=0).rename(columns=RENAME)
    rep = {}

    # ---- baseline on the untouched data (median impute only) for an honest before/after
    rb = raw.copy()
    rb["income"] = rb["income"].fillna(rb["income"].median())
    rb["dependents"] = rb["dependents"].fillna(0)
    Xb, yb = rb.drop(columns="default"), rb["default"]
    Xtr, Xte, ytr, yte = train_test_split(Xb, yb, test_size=0.2, stratify=yb, random_state=42)
    mu, sd = Xtr.mean(), Xtr.std()
    m0 = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    p0 = m0.predict_proba((Xte - mu) / sd)[:, 1]
    rep["baseline_raw_lr"] = {"auc": float(roc_auc_score(yte, p0)), "ks": ks_stat(yte, p0)}

    # ---- clean + engineer
    df = clean(raw, rep)
    X, y = engineer(df), df["default"]
    # 60 / 20 / 20 split: train, calibration, test (the test set is the same 20% as before)
    Xrest, Xte, yrest, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xtr, Xcal, ytr, ycal = train_test_split(Xrest, yrest, test_size=0.25, stratify=yrest, random_state=42)
    mu, sd = Xtr.mean(), Xtr.std()
    Z = lambda A: ((A - mu) / sd).clip(-4, 4)       # bounded inputs -> predictable CKKS range
    m = LogisticRegression(max_iter=1000).fit(Z(Xtr), ytr)
    # Platt scaling on the calibration split: z' = a*z + c. It is a LINEAR map of the logit, so it folds into the
    # weights and bias (w' = a*w, b' = a*b + c) and costs nothing under FHE. (Isotonic would not stay linear.)
    zc = Z(Xcal).values @ m.coef_[0] + m.intercept_[0]
    pl = LogisticRegression(C=1e6, max_iter=1000).fit(zc.reshape(-1, 1), ycal)
    pa, pc = float(pl.coef_[0][0]), float(pl.intercept_[0])
    coef, icpt = m.coef_[0] * pa, float(m.intercept_[0] * pa + pc)
    sg = lambda t: 1 / (1 + np.exp(-t))
    zt_raw, zt = Z(Xte).values @ m.coef_[0] + m.intercept_[0], Z(Xte).values @ coef + icpt
    rep["calibration"] = {"platt_a": pa, "platt_c": pc, "test_before": calib(yte.values, sg(zt_raw)),
                          "test_after": calib(yte.values, sg(zt))}
    p = sg(zt)
    rep["clean_lr"] = {"auc": float(roc_auc_score(yte, p)), "ks": ks_stat(yte, p)}

    # ---- FHE feasibility: logit range + low-degree polynomial sigmoid
    logit = zt
    lo, hi = float(np.percentile(logit, 0.1)), float(np.percentile(logit, 99.9))
    xs = np.linspace(-8, 8, 2001)
    sig = lambda t: 1 / (1 + np.exp(-t))
    poly = {}
    for deg in (3, 5):
        pcf = np.polyfit(xs, sig(xs), deg)
        pp = np.polyval(pcf, np.clip(logit, -8, 8))
        poly[f"deg{deg}"] = {"coeffs_high_to_low": pcf.tolist(),
                             "auc_after_approx": float(roc_auc_score(yte, pp)),
                             "max_abs_err_vs_sigmoid": float(np.max(np.abs(pp - sig(np.clip(logit, -8, 8)))))}
    rep["logit_range_p0.1_p99.9"] = [lo, hi]
    rep["poly_sigmoid_on_[-8,8]"] = poly

    zall = Z(X).values @ coef + icpt
    sc = to_score(zall); k = SCORE["pdo"] / np.log(2)
    rep["score_bands"] = [{"band": n, "from": a, "to": b, "share_pct": round(float(((sc >= a) & (sc <= b)).mean() * 100), 1),
                           "observed_default_pct": round(float(y.values[(sc >= a) & (sc <= b)].mean() * 100), 2),
                           "implied_pd_at_lower_edge_pct": round(100 * float(1 / (1 + np.exp(-(logit_(SCORE["reference_pd"]) - (a - SCORE["base_score"]) / k)))), 1)}
                          for n, a, b in BANDS]
    rep["score"] = SCORE
    # ---- write outputs
    df.to_csv(os.path.join(out, "cs-training-clean.csv"), index=False)
    mr = Z(X).round(5)
    mr.insert(0, "default", y.values)
    mr.to_csv(os.path.join(out, "cs-training-model-ready.csv"), index=False)
    art = {
        "features": FEATURES,
        "mean": mu.to_dict(), "std": sd.to_dict(), "clip_z": [-4, 4],
        "caps": rep["caps"], "impute": {"income": rep["impute_income_median"],
                                        "debt_ratio": rep["impute_debt_ratio_median"], "dependents": 0},
        "weights": dict(zip(FEATURES, coef.tolist())), "bias": icpt, "score": SCORE, "poly_range": [lo - 0.5, hi + 0.5],
        "sigmoid_poly": poly["deg3"]["coeffs_high_to_low"], "sigmoid_poly_range": [-8, 8],
    }
    json.dump(art, open(os.path.join(out, "model_artifacts.json"), "w"), indent=2)
    json.dump(rep, open(os.path.join(out, "cleaning_report.json"), "w"), indent=2)
    rep["rows_out"] = len(df)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="cs-training.csv")
    ap.add_argument("--out", default="./out")
    a = ap.parse_args()
    main(a.inp, a.out)
