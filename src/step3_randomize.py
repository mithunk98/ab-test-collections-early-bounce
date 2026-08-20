"""
Step 3 - Stratified randomisation and balance diagnostics.

Assigns each account to control or treatment, forcing balance on the
covariates most predictive of the outcome, then checks that the draw
we actually got is balanced. Also quantifies the contamination risk
created by a dealer-level treatment on a customer-level randomisation.

Run:  python src/step3_randomize.py
"""

import warnings

import numpy as np
import pandas as pd

import config as C

warnings.filterwarnings("ignore")


def stratified_assign(df: pd.DataFrame, strata, p_treat, seed) -> pd.Series:
    """Within each stratum, permute and split. Guarantees near-exact
    balance per cell rather than relying on the law of large numbers,
    which does not apply kindly at n=94."""
    rng = np.random.default_rng(seed)
    arm = pd.Series(index=df.index, dtype=object)
    for _, idx in df.groupby(strata, observed=True).groups.items():
        idx = np.array(idx)
        rng.shuffle(idx)
        n_treat = int(round(len(idx) * p_treat))
        arm.loc[idx[:n_treat]] = "treatment"
        arm.loc[idx[n_treat:]] = "control"
    return arm


def smd(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference. The convention is that |SMD| < 0.10
    counts as balanced; a p-value on a balance test is the wrong tool
    because it conflates imbalance with sample size."""
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return 0.0 if pooled == 0 else (a.mean() - b.mean()) / pooled


def main():
    df = pd.read_csv(C.DATA / "cohort_anonymised.csv")
    print("Step 3 - stratified randomisation")
    print(f"Cohort: {len(df)} accounts | strata: {C.STRATA} | seed: {C.RANDOM_SEED}\n")

    # --- Rerandomisation --------------------------------------------------
    # Stratifying on vintage and city does not constrain EMI size, and at
    # n=94 a single draw can easily land imbalanced on it. Rerandomisation
    # (Morgan & Rubin) redraws until the acceptance criterion is met. It is
    # valid ONLY because no outcome data exists yet and the criterion is
    # fixed in advance - otherwise it is p-hacking by another name.
    BALANCE_COVARIATES = ["emi_amount", "total_outstanding"]
    ACCEPT_SMD = 0.10
    MAX_DRAWS = 5000

    accepted_at = None
    for draw in range(MAX_DRAWS):
        arm = stratified_assign(df, C.STRATA, C.ALLOCATION, C.RANDOM_SEED + draw)
        t_, c_ = df[arm == "treatment"], df[arm == "control"]
        if all(abs(smd(t_[col], c_[col])) < ACCEPT_SMD for col in BALANCE_COVARIATES):
            df["arm"] = arm
            accepted_at = draw
            break
    else:
        raise RuntimeError(f"No balanced allocation found in {MAX_DRAWS} draws")

    print(f"Rerandomisation: accepted draw {accepted_at + 1} "
          f"(criterion |SMD| < {ACCEPT_SMD} on {BALANCE_COVARIATES})\n")

    t = df[df.arm == "treatment"]
    c = df[df.arm == "control"]
    print(f"Assigned: {len(t)} treatment / {len(c)} control\n")

    # --- Balance on stratifying variables ---------------------------------
    print("Balance by stratum:")
    cross = pd.crosstab([df.emi_vintage, df.city_tier], df.arm)
    print(cross.to_string())

    # --- Balance on covariates not used for stratification ----------------
    print("\nBalance on continuous covariates (|SMD| < 0.10 = balanced):")
    rows = []
    for col in ["emi_amount", "total_outstanding"]:
        d = smd(t[col], c[col])
        rows.append({
            "covariate": col,
            "treatment_mean": round(t[col].mean(), 1),
            "control_mean": round(c[col].mean(), 1),
            "smd": round(d, 3),
            "balanced": "yes" if abs(d) < 0.10 else "NO - flag",
        })
    bal = pd.DataFrame(rows)
    print(bal.to_string(index=False))

    # --- Contamination diagnostic -----------------------------------------
    # The treatment is an email to a DEALER, but we randomise CUSTOMERS.
    # A dealer holding both arms is told about their treated accounts and
    # may act on their control accounts too, biasing the effect toward zero.
    print("\n" + "-" * 62)
    print("CONTAMINATION RISK - dealer-level treatment, customer-level draw")
    print("-" * 62)
    per_dealer = df.groupby("dealer_id")["arm"].nunique()
    mixed = (per_dealer > 1).sum()
    exposed = df[df.dealer_id.isin(per_dealer[per_dealer > 1].index)]
    exposed_ctrl = (exposed.arm == "control").sum()
    print(f"Dealers holding both arms: {mixed} of {df.dealer_id.nunique()}")
    print(f"Control accounts at a mixed dealer: {exposed_ctrl} "
          f"({exposed_ctrl/len(c):.0%} of control)")
    print("\nThese control accounts sit with a dealer who IS receiving alerts")
    print("about their other customers. Any spillover attenuates the measured")
    print("effect, so a null result cannot be read as 'the alert does nothing'.")
    print("\nMitigation options:")
    print("  a) Randomise at DEALER level instead - removes spillover entirely,")
    print(f"     but with only {df.dealer_id.nunique()} dealers the effective sample")
    print("     collapses and power gets far worse. Rejected on those grounds.")
    print("  b) Keep the customer-level draw, pre-register the estimate as an")
    print("     INTENT-TO-TREAT lower bound, and report the contamination rate")
    print("     alongside it. <- chosen")

    df.to_csv(C.DATA / "cohort_assigned.csv", index=False)
    bal.to_csv(C.OUTPUTS / "balance_check.csv", index=False)
    print(f"\nWritten: {C.DATA/'cohort_assigned.csv'}")


if __name__ == "__main__":
    main()
