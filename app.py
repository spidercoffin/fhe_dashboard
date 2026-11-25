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
D = next(p for p in [HERE, HERE / "data", HERE.parent]
         if (p / "model_artifacts.json").exists()
         and all(k in json.loads((p / "model_artifacts.json").read_text()) for k in ("features", "weights", "bias", "score", "poly_range")))
ART = json.load(open(D / "model_artifacts.json"))
DATA = next(p for p in [HERE / "data", HERE, HERE.parent]
            if (p / "cs-training-model-ready.csv").exists())
F = ART["features"]; IDX = {f: i for i, f in enumerate(F)}
MU = np.array([ART["mean"][f] for f in F]); SD = np.array([ART["std"][f] for f in F])
W = [ART["weights"][f] for f in F]; B = ART["bias"]
MODEL_HASH = hashlib.sha256(json.dumps(ART, sort_keys=True).encode()).hexdigest()[:12]
_x = np.linspace(*ART["poly_range"], 901)
C3, C2, C1, C0 = np.polyfit(_x, 1 / (1 + np.exp(-_x)), 3)   # deg-3 sigmoid, fit on observed logit range
SC = ART["score"]                                           # written by prepare_dataset.py
LO, HI, BASE_SCORE, PDO, REF_PD = SC["lo"], SC["hi"], SC["base_score"], SC["pdo"], SC["reference_pd"]
Z_REF = np.log(REF_PD / (1 - REF_PD))                       # LO/HI/bands are published; the anchors (700 = 5% PD, 800 = 1% PD) are OUR design choice
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


POP = pd.read_csv(DATA / "cs-training-model-ready.csv")     # already standardized + clipped: exactly the vectors that get encrypted
POP_SC = np.clip(np.round(BASE_SCORE - PDO / np.log(2) * (POP[F].values @ np.array(W) + B - Z_REF)), LO, HI)   # every dataset customer's score


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
    Z = POP.sample(n, random_state=7).reset_index(drop=True)[F].values     # label column is never used
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
    r = cust.release(cid, ez, ep, eg); r["percentile"] = round(float((POP_SC < r["score"]).mean() * 100), 1); r["ms_encrypted_compute"] = round((t1 - t) * 1000); r["ms_decrypt"] = round((time.perf_counter() - t1) * 1000)
    return r


app = FastAPI(title="i-Score encrypted scoring console"); seed()


class Upd(BaseModel):
    event: str; params: dict = {}


class Cons(BaseModel):
    granted: bool


@app.get("/api/info")
def info(): return dict(model=MODEL_HASH, scheme="CKKS", poly_mod=8192, levels=4, features=len(F), customers=len(vault), lo=LO, hi=HI, base=BASE_SCORE, pdo=round(PDO, 2), ref_pd=REF_PD,
                                   pop=dict(n=len(POP), median=int(np.median(POP_SC)), p25=int(np.percentile(POP_SC, 25)), p75=int(np.percentile(POP_SC, 75)),
                                            good_share=round(float((POP_SC >= 700).mean() * 100)), default_rate=round(float(POP["default"].mean() * 100), 2)))


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