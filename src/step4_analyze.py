"""
Step 4 - Analysis and readout.

Implements the pre-registered analysis plan:
  primary    amount_recovered_ratio   Welch t-test, covariate-adjusted
  secondary  cured_30d                two-proportion z, Wilson CI
  guardrail  escalation_30d           one-sided non-inferiority check
  segments   by bounce vintage        Benjamini-Hochberg corrected

OUTCOME DATA
------------
Real 30-day outcomes are not yet available - the intervention has not
been run as a controlled test. This script therefore SIMULATES outcomes
under a stated true effect so the analysis pipeline is complete and
testable end to end. Every simulated figure is labelled. To run for
real, drop a CSV with columns
    customer_id, amount_recovered_ratio, cured_30d, escalation_30d
into data/outcomes.csv and this script will use it instead.

Run:  python src/step4_analyze.py
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import (
    proportion_confint, proportions_ztest,
)

import config as C

warnings.filterwarnings("ignore")

# Ground truth used only when simulating. Chosen deliberately near the
# detection boundary so the readout is realistic rather than flattering.
SIM_TRUE_EFFECT_D = 0.30
SIM_BASELINE_RECOVERY = 0.52
SIM_RECOVERY_SD = 0.31


def simulate_outcomes(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    lift = SIM_TRUE_EFFECT_D * SIM_RECOVERY_SD
    mu = np.where(df.arm == "treatment",
                  SIM_BASELINE_RECOVERY + lift, SIM_BASELINE_RECOVERY)

    # Recovery ratio is bounded [0, 1] and lumpy at both ends, so a plain
    # normal draw would misstate the variance. Beta keeps it in range.
    mu = np.clip(mu, 0.02, 0.98)
    conc = (mu * (1 - mu) / SIM_RECOVERY_SD**2) - 1
    conc = np.maximum(conc, 0.6)
    ratio = rng.beta(mu * conc, (1 - mu) * conc)

    out = df.copy()
    out["amount_recovered_ratio"] = ratio
    out["cured_30d"] = (ratio >= 0.95).astype(int)   # full EMI cleared
    # Guardrail: escalations driven mostly by contact intensity, with a
    # small genuine treatment cost baked in.
    p_esc = np.where(df.arm == "treatment", 0.055, 0.045)
    out["escalation_30d"] = rng.binomial(1, p_esc)
    return out


def welch(t: np.ndarray, c: np.ndarray) -> dict:
    res = stats.ttest_ind(t, c, equal_var=False)
    nt, nc = len(t), len(c)
    pooled = np.sqrt(((nt - 1) * t.var(ddof=1) + (nc - 1) * c.var(ddof=1))
                     / (nt + nc - 2))
    d = (t.mean() - c.mean()) / pooled
    se = np.sqrt(t.var(ddof=1) / nt + c.var(ddof=1) / nc)
    dfree = se**4 / ((t.var(ddof=1) / nt)**2 / (nt - 1)
                     + (c.var(ddof=1) / nc)**2 / (nc - 1))
    crit = stats.t.ppf(0.975, dfree)
    diff = t.mean() - c.mean()
    return {"diff": diff, "ci": (diff - crit * se, diff + crit * se),
            "p": res.pvalue, "d": d}


def report(df: pd.DataFrame, label: str) -> dict:
    t = df[df.arm == "treatment"]
    c = df[df.arm == "control"]
    print("\n" + "=" * 68)
    print(f"READOUT - {label}")
    print("=" * 68)
    print(f"n = {len(t)} treatment / {len(c)} control")

    # --- Primary ----------------------------------------------------------
    r = welch(t.amount_recovered_ratio.values, c.amount_recovered_ratio.values)
    sig = r["p"] < C.ALPHA
    print("\nPRIMARY - amount recovered ratio (30d)")
    print(f"  treatment {t.amount_recovered_ratio.mean():.4f}   "
          f"control {c.amount_recovered_ratio.mean():.4f}")
    print(f"  difference {r['diff']:+.4f}  "
          f"95% CI [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]")
    print(f"  p = {r['p']:.4f}   Cohen's d = {r['d']:+.3f}   "
          f"-> {'SIGNIFICANT' if sig else 'not significant'}")

    # Covariate adjustment buys precision without touching the estimand.
    X = pd.get_dummies(df[["emi_vintage", "city_tier"]], drop_first=True).astype(float)
    X["emi_amount_z"] = ((df.emi_amount - df.emi_amount.mean())
                         / df.emi_amount.std())
    X["treated"] = (df.arm == "treatment").astype(int)
    model = sm.OLS(df.amount_recovered_ratio.values,
                   sm.add_constant(X, has_constant="add")).fit(cov_type="HC3")
    print(f"  covariate-adjusted effect {model.params['treated']:+.4f} "
          f"(p = {model.pvalues['treated']:.4f})")

    # --- Secondary --------------------------------------------------------
    ct, cc = int(t.cured_30d.sum()), int(c.cured_30d.sum())
    zstat, pz = proportions_ztest([ct, cc], [len(t), len(c)])
    lo_t, hi_t = proportion_confint(ct, len(t), method="wilson")
    lo_c, hi_c = proportion_confint(cc, len(c), method="wilson")
    print("\nSECONDARY - 30-day cure rate  [underpowered by design, not decisive]")
    print(f"  treatment {ct}/{len(t)} = {ct/len(t):.1%}  "
          f"Wilson CI [{lo_t:.1%}, {hi_t:.1%}]")
    print(f"  control   {cc}/{len(c)} = {cc/len(c):.1%}  "
          f"Wilson CI [{lo_c:.1%}, {hi_c:.1%}]")
    print(f"  p = {pz:.4f}")

    # --- Guardrail --------------------------------------------------------
    et, ec = t.escalation_30d.mean(), c.escalation_30d.mean()
    breach = (et - ec) > C.GUARDRAIL_MAX_INCREASE
    print("\nGUARDRAIL - escalation rate (30d)")
    print(f"  treatment {et:.1%}   control {ec:.1%}   "
          f"delta {et-ec:+.1%}  (limit +{C.GUARDRAIL_MAX_INCREASE:.0%})")
    print(f"  -> {'BREACHED - stop' if breach else 'within tolerance'}")

    # --- Segments ---------------------------------------------------------
    print("\nSEGMENTS - by bounce vintage (exploratory, BH-corrected)")
    seg_rows, pvals = [], []
    for v in ["FEMI", "SEMI", "TEMI"]:
        st, sc = t[t.emi_vintage == v], c[c.emi_vintage == v]
        if len(st) < 5 or len(sc) < 5:
            seg_rows.append({"vintage": v, "n_t": len(st), "n_c": len(sc),
                             "diff": None, "p_raw": None})
            continue
        rr = welch(st.amount_recovered_ratio.values,
                   sc.amount_recovered_ratio.values)
        seg_rows.append({"vintage": v, "n_t": len(st), "n_c": len(sc),
                         "diff": round(rr["diff"], 4), "p_raw": round(rr["p"], 4)})
        pvals.append(rr["p"])
    if pvals:
        adj = multipletests(pvals, method="fdr_bh")[1]
        it = iter(adj)
        for row in seg_rows:
            row["p_bh"] = round(next(it), 4) if row["p_raw"] is not None else None
    print(pd.DataFrame(seg_rows).to_string(index=False))
    print("  Segment results are hypothesis-generating only. They were not")
    print("  pre-registered and each cell is far below the powered sample size.")

    return {"label": label, "n": len(df), "p": r["p"], "diff": r["diff"],
            "d": r["d"], "significant": sig, "guardrail_breach": breach}


def decision(res: dict):
    print("\n" + "=" * 68)
    print("DECISION")
    print("=" * 68)
    if res["guardrail_breach"]:
        print("STOP. Guardrail breached - customer escalations rose beyond the")
        print("pre-registered tolerance. Roll back regardless of the primary.")
    elif res["significant"] and res["diff"] > 0:
        print("SHIP. Primary metric improved significantly, guardrail held.")
        print("Note the effect is an intent-to-treat lower bound: 81% of control")
        print("sits with a dealer who received alerts about other accounts, so")
        print("the true effect is likely larger than measured.")
    elif res["significant"]:
        print("ROLL BACK. Primary moved significantly in the wrong direction.")
    else:
        print("INCONCLUSIVE - not the same as 'no effect'.")
        print(f"The observed difference is {res['diff']:+.4f} (d = {res['d']:+.3f})")
        print("but the confidence interval spans zero at this sample size.")
        print("Continue accruing to the pre-registered n before deciding.")


def main():
    df = pd.read_csv(C.DATA / "cohort_assigned.csv")
    outcomes_path = C.DATA / "outcomes.csv"

    if outcomes_path.exists():
        print("Using REAL outcomes from data/outcomes.csv")
        df = df.merge(pd.read_csv(outcomes_path), on="customer_id", how="inner")
        res = report(df, f"real outcomes, n={len(df)}")
        decision(res)
        return

    print("!" * 68)
    print("SIMULATED OUTCOMES - the controlled test has not been run.")
    print(f"True effect injected: d = {SIM_TRUE_EFFECT_D}. Figures below")
    print("validate the analysis pipeline; they are not business results.")
    print("!" * 68)

    # Month 1 - the cohort we actually have.
    m1 = simulate_outcomes(df, seed=C.RANDOM_SEED)
    res1 = report(m1, "Month 1 only (n=94, as-is cohort)")
    decision(res1)

    # Accrued to the pre-registered sample size.
    reps = int(np.ceil(260 / len(df)))
    pooled = pd.concat(
        [simulate_outcomes(df.assign(customer_id=df.customer_id + f"_M{i}"),
                           seed=C.RANDOM_SEED + 100 + i) for i in range(reps)],
        ignore_index=True,
    ).head(260)
    res2 = report(pooled, "Accrued to pre-registered n=260 (~2.8 months)")
    decision(res2)

    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([res1, res2]).to_csv(C.OUTPUTS / "readout_summary.csv", index=False)
    print(f"\nWritten: {C.OUTPUTS/'readout_summary.csv'}")


if __name__ == "__main__":
    main()
