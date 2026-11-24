"""i-Score employee console: real CKKS (TenSEAL) behind a FastAPI gateway.
Custodian = trusted key owner (secret key; releases only the final score).
Worker    = scoring server (public context only; never sees plaintext or the secret key).
Gateway   = what the employee UI calls (consent gate + audit log, stand-in for Fabric)."""
import hashlib, json, pathlib, time
from datetime import datetime, timezone
import numpy as np, pandas as pd, tenseal as ts
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent
D = next(p for p in [HERE / "data", HERE.parent, HERE] if (p / "model_artifacts.json").exists())
ART = json.load(open(D / "model_artifacts.json"))
F = ART["features"]; IDX = {f: i for i, f in enumerate(F)}
MU = np.array([ART["mean"][f] for f in F]); SD = np.array([ART["std"][f] for f in F])
W = [ART["weights"][f] for f in F]; B = ART["bias"]
MODEL_HASH = hashlib.sha256(json.dumps(ART, sort_keys=True).encode()).hexdigest()[:12]
_x = np.linspace(-6, 3, 901)
C3, C2, C1, C0 = np.polyfit(_x, 1 / (1 + np.exp(-_x)), 3)   # deg-3 sigmoid, fit on observed logit range
Z_REF = np.log(0.0668 / 0.9332)                             # default rate of the training data
LO, HI, BASE_SCORE, PDO = 400, 850, 700, 50                 # LO/HI/bands are published; BASE_SCORE and PDO are OUR assumption (i-Score doesn't publish its calibration)
FACTORS = [  # key, name, published weight % (third-party article, not i-Score), our model features that feed it
    ("pay", "Payment history", 35, ["late_total", "any_late90", "late_code_flag"]),
    ("debt", "Outstanding debt", 30, ["util", "log_debt_ratio"]),
    ("hist", "Credit history length", 15, []),               # not in Give Me Some Credit
    ("new", "New credit / inquiries", 10, []),               # not in Give Me Some Credit
    ("mix", "Types of credit", 10, ["open_lines", "re_loans"]),
    ("oth", "Other model inputs", None, ["age", "log_income", "dependents", "income_missing"])]
GW = {k: [W[IDX[f]] if f in fs else 0.0 for f in F] for k, _, _, fs in FACTORS if fs}


def to_score(z):
    return int(np.clip(round(BASE_SCORE - PDO / np.log(2) * (z - Z_REF)), LO, HI))


def band(s):                                                # bands as published by Banky (Jul 2026)
    return "Weak" if s < 521 else "Unsatisfactory" if s < 626 else "Satisfactory" if s < 700 else "Good" if s < 800 else "Excellent"


class Custodian:
    def __init__(s):
        s.ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[40, 30, 30, 30, 30, 40])
        s.ctx.global_scale = 2 ** 30; s.ctx.generate_galois_keys()   # 200 of 218 bits; 4 levels = depth of the circuit
        s.plain = {}                                         # demo-only mirror, used to measure CKKS error

    def public(s): return s.ctx.serialize(save_secret_key=False)

    def enroll(s, cid, x):
        s.plain[cid] = x.copy(); return ts.ckks_vector(s.ctx, x.tolist()).serialize()

    def release(s, cid, enc_z, enc_p, enc_g):                       # only the score leaves; features never do
        z = ts.ckks_vector_from(s.ctx, enc_z).decrypt()[0]
        p = float(np.clip(ts.ckks_vector_from(s.ctx, enc_p).decrypt()[0], 0, 1))
        zp = float(np.dot(s.plain[cid], W) + B)
        k = PDO / np.log(2)                                  # points per unit of log-odds; factor points are relative to an average profile
        fac = [dict(key=key, name=name, weight=wt, features=fs,
                    points=round(-k * ts.ckks_vector_from(s.ctx, enc_g[key]).decrypt()[0], 1) if fs else None)
               for key, name, wt, fs in FACTORS]
        return dict(score=to_score(z), band=band(to_score(z)), z=z, pd=p, abs_err=abs(z - zp), plain_score=to_score(zp),
                    baseline=round(BASE_SCORE - k * (B - Z_REF)), factors=fac)


class Worker:
    def __init__(s, pub): s.ctx = ts.context_from(pub)       # no secret key in here

    def score(s, ct):
        v = ts.ckks_vector_from(s.ctx, ct); z = v.dot(W) + B  # w.x + b on ciphertext
        z2 = z * z; t = z2 * C3 + C1
        p = z * t + z2 * C2 + C0                             # polynomial sigmoid on ciphertext
        return z.serialize(), p.serialize(), {k: v.dot(g).serialize() for k, g in GW.items()}   # per-factor partial sums

    def update(s, ct, delta): return (ts.ckks_vector_from(s.ctx, ct) + delta.tolist()).serialize()


def event_delta(ev, p):
    d = np.zeros(len(F))
    if ev == "credit_card": d[IDX["open_lines"]] = 1; d[IDX["util"]] = p.get("util_pp", 0) / 100
    elif ev == "loan":
        d[IDX["open_lines"]] = 1; d[IDX["re_loans"]] = 1 if p.get("kind") == "mortgage" else 0
        d[IDX["log_debt_ratio"]] = p.get("payment_pct", 0) / 100 / 1.29   # first-order log1p at median ratio
    elif ev == "late_payment": d[IDX["late_total"]] = 1      # 30-89 days only (90+ needs an encrypted comparison)
    elif ev == "close_account": d[IDX["open_lines"]] = -1
    elif ev == "dependent": d[IDX["dependents"]] = 1
    else: raise HTTPException(400, "unknown event")
    return d / SD                                            # features are stored standardized


cust, worker = Custodian(), None
worker = Worker(cust.public())
vault, consent, version, audit = {}, {}, {}, []


def seed(n=32):
    df = pd.read_csv(D / "cs-training-clean.csv").sample(n, random_state=7).reset_index(drop=True)
    X = pd.DataFrame({"util": df.util, "age": df.age, "log_income": np.log1p(df.income),
                      "log_debt_ratio": np.log1p(df.debt_ratio), "open_lines": df.open_lines, "re_loans": df.re_loans,
                      "dependents": df.dependents, "late_total": (df.late30 + df.late60 + df.late90).clip(0, 10),
                      "any_late90": (df.late90 > 0).astype(int), "late_code_flag": df.late_code_flag,
                      "income_missing": df.income_missing})[F].astype(float).values
    Z = np.clip((X - MU) / SD, -4, 4)
    for i, z in enumerate(Z):
        cid = f"IS-{1001 + i}"; vault[cid] = cust.enroll(cid, z); consent[cid] = True; version[cid] = 1
        log("ENROLL", cid, vault[cid])


def log(ev, cid, ct=None, **x):
    audit.insert(0, dict(ts=datetime.now(timezone.utc).strftime("%H:%M:%S"), event=ev, customer=cid, model=MODEL_HASH,
                         ct_hash=hashlib.sha256(ct).hexdigest()[:12] if ct else "-", **x))
    del audit[200:]


def guard(cid):
    if cid not in vault: raise HTTPException(404, "unknown customer")
    if not consent[cid]: log("DENIED", cid, vault[cid], note="consent revoked"); raise HTTPException(403, "Consent revoked: computation blocked")


def run_score(cid):
    t = time.perf_counter(); ez, ep, eg = worker.score(vault[cid]); t1 = time.perf_counter()
    r = cust.release(cid, ez, ep, eg); r["ms_encrypted_compute"] = round((t1 - t) * 1000); r["ms_decrypt"] = round((time.perf_counter() - t1) * 1000)
    return r


app = FastAPI(title="i-Score encrypted scoring console"); seed()


class Upd(BaseModel):
    event: str; params: dict = {}


class Cons(BaseModel):
    granted: bool


@app.get("/api/info")
def info(): return dict(model=MODEL_HASH, scheme="CKKS", poly_mod=8192, levels=4, features=len(F), customers=len(vault), lo=LO, hi=HI, base=BASE_SCORE, pdo=PDO)


@app.get("/api/customers")
def customers():
    return [dict(id=c, consent=consent[c], version=version[c], kb=round(len(vault[c]) / 1024), preview=vault[c][40:64].hex(),
                 hash=hashlib.sha256(vault[c]).hexdigest()[:12]) for c in vault]


@app.post("/api/score/{cid}")
def score(cid: str):
    guard(cid); r = run_score(cid); log("SCORE", cid, vault[cid], score=r["score"]); return r


@app.post("/api/update/{cid}")
def update(cid: str, u: Upd):
    guard(cid); before = run_score(cid); d = event_delta(u.event, u.params)
    vault[cid] = worker.update(vault[cid], d); cust.plain[cid] += d; version[cid] += 1
    after = run_score(cid); log("UPDATE", cid, vault[cid], note=u.event, score=after["score"])
    return dict(before=before, after=after, change=after["score"] - before["score"], version=version[cid],
                changes={F[i]: round(float(d[i] * SD[i]), 3) for i in range(len(F)) if d[i]})


@app.post("/api/consent/{cid}")
def set_consent(cid: str, c: Cons):
    if cid not in vault: raise HTTPException(404, "unknown customer")
    consent[cid] = c.granted; log("CONSENT_GRANT" if c.granted else "CONSENT_REVOKE", cid, vault[cid]); return dict(consent=c.granted)


@app.get("/api/audit")
def get_audit(): return audit[:60]


@app.get("/")
def root(): return FileResponse(HERE / "static" / "index.html")
